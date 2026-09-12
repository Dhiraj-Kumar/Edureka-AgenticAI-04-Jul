import os
import time
from functools import wraps
from typing import Callable
from typing_extensions import TypedDict

from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode

from prometheus_client import Counter, Histogram, start_http_server

load_dotenv()

# ------------------------------------------------------------
# OpenTelemetry configuration
# ------------------------------------------------------------

resource = Resource.create(
    {
        "service.name": "langgraph-observability-agent",
        "service.version": "1.0.0",
        "deployment.environment": "demo",
    }
)

provider = TracerProvider(resource=resource)

otlp_exporter = OTLPSpanExporter(
    endpoint=os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://localhost:4318/v1/traces",
    )
)

provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("langgraph-observability")


# ------------------------------------------------------------
# Prometheus metrics
# ------------------------------------------------------------

REQUESTS = Counter(
    "agent_requests_total",
    "Total number of agent requests.",
    ["status"],
)

REQUEST_LATENCY = Histogram(
    "agent_request_latency_seconds",
    "End-to-end request latency.",
)

NODE_LATENCY = Histogram(
    "agent_node_latency_seconds",
    "Execution latency for each LangGraph node.",
    ["node"],
)

TOKENS = Counter(
    "agent_tokens_total",
    "Total tokens used by the agent.",
    ["type"],
)

ERRORS = Counter(
    "agent_errors_total",
    "Total number of agent errors.",
    ["node", "error_type"],
)

ESTIMATED_COST = Counter(
    "agent_estimated_cost_usd_total",
    "Estimated model cost in USD.",
)


# ------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------

model = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=500,
)

INPUT_COST_PER_MILLION = float(
    os.getenv("INPUT_TOKEN_COST_PER_MILLION", "3.00")
)

OUTPUT_COST_PER_MILLION = float(
    os.getenv("OUTPUT_TOKEN_COST_PER_MILLION", "15.00")
)


# ------------------------------------------------------------
# LangGraph state
# ------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    plan: str
    context: str
    answer: str
    input_tokens: int
    output_tokens: int


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def extract_usage(response) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None) or {}

    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))

    return input_tokens, output_tokens


def record_usage(input_tokens: int, output_tokens: int) -> None:
    TOKENS.labels(type="input").inc(input_tokens)
    TOKENS.labels(type="output").inc(output_tokens)

    cost = (
        input_tokens / 1_000_000 * INPUT_COST_PER_MILLION
        + output_tokens / 1_000_000 * OUTPUT_COST_PER_MILLION
    )

    ESTIMATED_COST.inc(cost)


def instrument_node(node_name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: AgentState) -> dict:
            started = time.perf_counter()

            with tracer.start_as_current_span(
                f"langgraph.node.{node_name}"
            ) as span:
                span.set_attribute("langgraph.node", node_name)

                try:
                    result = func(state)

                    span.set_attribute("node.status", "success")
                    span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as exc:
                    ERRORS.labels(
                        node=node_name,
                        error_type=type(exc).__name__,
                    ).inc()

                    span.record_exception(exc)
                    span.set_attribute("node.status", "error")
                    span.set_status(
                        Status(StatusCode.ERROR, str(exc))
                    )

                    raise

                finally:
                    elapsed = time.perf_counter() - started

                    NODE_LATENCY.labels(
                        node=node_name
                    ).observe(elapsed)

                    span.set_attribute(
                        "node.duration_seconds",
                        elapsed,
                    )

        return wrapper

    return decorator


# ------------------------------------------------------------
# LangGraph nodes
# ------------------------------------------------------------

@instrument_node("planner")
def planner(state: AgentState) -> dict:
    response = model.invoke(
        "Create a concise two-step plan for answering this question:\n\n"
        f"{state['question']}"
    )

    input_tokens, output_tokens = extract_usage(response)
    record_usage(input_tokens, output_tokens)

    span = trace.get_current_span()
    span.set_attribute("llm.input_tokens", input_tokens)
    span.set_attribute("llm.output_tokens", output_tokens)

    return {
        "plan": response.content,
        "input_tokens": state["input_tokens"] + input_tokens,
        "output_tokens": state["output_tokens"] + output_tokens,
    }


@instrument_node("retriever")
def retriever(state: AgentState) -> dict:
    question = state["question"].lower()

    if "simulate error" in question:
        raise RuntimeError("Simulated retriever failure")

    knowledge_base = {
        "langgraph": (
            "LangGraph builds stateful workflows using nodes, "
            "edges, shared state, and conditional routing."
        ),
        "opentelemetry": (
            "OpenTelemetry provides vendor-neutral APIs and SDKs "
            "for collecting traces, metrics, and logs."
        ),
        "prometheus": (
            "Prometheus stores time-series metrics by scraping "
            "application HTTP endpoints."
        ),
        "grafana": (
            "Grafana visualises monitoring data through dashboards, "
            "panels, queries, and alerts."
        ),
    }

    matches = [
        content
        for keyword, content in knowledge_base.items()
        if keyword in question
    ]

    context = "\n".join(matches)

    if not context:
        context = (
            "No direct knowledge-base match was found. "
            "Answer carefully and state uncertainty when needed."
        )

    trace.get_current_span().set_attribute(
        "retrieval.matches",
        len(matches),
    )

    return {"context": context}


@instrument_node("answer_generator")
def answer_generator(state: AgentState) -> dict:
    response = model.invoke(
        f"Question:\n{state['question']}\n\n"
        f"Plan:\n{state['plan']}\n\n"
        f"Context:\n{state['context']}\n\n"
        "Provide a clear and concise answer grounded in the context."
    )

    input_tokens, output_tokens = extract_usage(response)
    record_usage(input_tokens, output_tokens)

    span = trace.get_current_span()
    span.set_attribute("llm.input_tokens", input_tokens)
    span.set_attribute("llm.output_tokens", output_tokens)
    span.set_attribute("answer.length", len(response.content))

    return {
        "answer": response.content,
        "input_tokens": state["input_tokens"] + input_tokens,
        "output_tokens": state["output_tokens"] + output_tokens,
    }


# ------------------------------------------------------------
# Build the LangGraph workflow
# ------------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("planner", planner)
builder.add_node("retriever", retriever)
builder.add_node("answer_generator", answer_generator)

builder.add_edge(START, "planner")
builder.add_edge("planner", "retriever")
builder.add_edge("retriever", "answer_generator")
builder.add_edge("answer_generator", END)

graph = builder.compile()


# ------------------------------------------------------------
# Request-level tracing
# ------------------------------------------------------------

def run_agent(question: str) -> dict:
    started = time.perf_counter()

    with tracer.start_as_current_span(
        "langgraph.agent.run"
    ) as root_span:
        root_span.set_attribute("agent.question", question)
        root_span.set_attribute(
            "agent.workflow",
            "plan-retrieve-answer",
        )

        try:
            result = graph.invoke(
                {
                    "question": question,
                    "plan": "",
                    "context": "",
                    "answer": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            )

            REQUESTS.labels(status="success").inc()

            root_span.set_attribute("agent.status", "success")
            root_span.set_attribute(
                "agent.total_input_tokens",
                result["input_tokens"],
            )
            root_span.set_attribute(
                "agent.total_output_tokens",
                result["output_tokens"],
            )

            root_span.set_status(Status(StatusCode.OK))

            return result

        except Exception as exc:
            REQUESTS.labels(status="error").inc()

            root_span.record_exception(exc)
            root_span.set_attribute("agent.status", "error")
            root_span.set_status(
                Status(StatusCode.ERROR, str(exc))
            )

            raise

        finally:
            elapsed = time.perf_counter() - started

            REQUEST_LATENCY.observe(elapsed)

            root_span.set_attribute(
                "agent.duration_seconds",
                elapsed,
            )


# ------------------------------------------------------------
# Run the command-line application
# ------------------------------------------------------------

if __name__ == "__main__":
    metrics_server, metrics_thread = start_http_server(8000)

    print("LangGraph Observability Agent")
    print("Metrics:    http://localhost:8000")
    print("Jaeger:     http://localhost:16686")
    print("Prometheus: http://localhost:9090")
    print("Grafana:    http://localhost:3000")
    print("Type 'quit' to exit.")
    print("Type 'simulate error' to create a failed trace.\n")

    try:
        while True:
            question = input("Ask a question: ").strip()

            if question.lower() in {"quit", "exit"}:
                print("Goodbye!")
                break

            if not question:
                continue

            try:
                result = run_agent(question)

                print("\n" + "=" * 65)
                print("PLAN")
                print("=" * 65)
                print(result["plan"])

                print("\n" + "=" * 65)
                print("ANSWER")
                print("=" * 65)
                print(result["answer"])

                print(
                    f"\nTokens: "
                    f"{result['input_tokens']} input, "
                    f"{result['output_tokens']} output"
                )

                print("=" * 65 + "\n")

            except Exception as exc:
                print(f"\nAgent request failed: {exc}\n")

    finally:
        provider.force_flush()
        metrics_server.shutdown()
        metrics_server.server_close()
        metrics_thread.join(timeout=5)