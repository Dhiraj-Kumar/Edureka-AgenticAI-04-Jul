import json
import sqlite3
from typing import TypedDict, List, Dict, Literal, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------- Structured planning models ----------
class Step(BaseModel):
    id: int
    description: str
    depends_on: List[int] = Field(default_factory=list)
    sensitive: bool = Field(default=False, description="Requires human approval")


class Plan(BaseModel):
    steps: List[Step]


class FinalAnswer(BaseModel):
    response: str
    confidence: float = Field(ge=0.0, le=1.0)


planner_llm = llm.with_structured_output(Plan)
final_llm = llm.with_structured_output(FinalAnswer)


# ---------- State ----------
def merge_results(a: Dict, b: Dict) -> Dict:
    return {**a, **b}


class State(TypedDict):
    objective: str
    plan: List[dict]                       # serialized Steps
    results: Annotated[Dict[int, str], merge_results]
    skipped: Annotated[Dict[int, str], merge_results]
    final: dict


# ---------- Step with retry ----------
def run_step(description: str, context: str, retries: int = 2) -> str:
    last_err = ""
    for attempt in range(retries + 1):
        try:
            msg = llm.invoke(
                f"Execute this step and return only the result.\n"
                f"Step: {description}\nContext:\n{context}"
            )
            if not msg.content.strip():
                raise ValueError("empty tool response")
            return msg.content.strip()
        except Exception as e:      # noqa
            last_err = str(e)
    return f"[FAILED after {retries + 1} attempts: {last_err}]"


# ---------- Nodes ----------
def plan_node(state: State) -> State:
    if state.get("plan"):
        return {}
    plan = planner_llm.invoke(
        f"Break this objective into minimal dependent steps. Mark any step "
        f"that deletes data, sends messages, or spends money as sensitive.\n\n"
        f"Objective: {state['objective']}"
    )
    return {"plan": [s.model_dump() for s in plan.steps]}


def ready_steps(state: State) -> List[dict]:
    done = set(state.get("results", {})) | set(state.get("skipped", {}))
    ready = []
    for s in state["plan"]:
        if s["id"] in done:
            continue
        if all(d in done for d in s["depends_on"]):
            ready.append(s)
    return ready


def dispatch(state: State):
    ready = ready_steps(state)
    if not ready:
        return "finalize"
    return [Send("execute", {"step": s, "state_results": state.get("results", {})})
            for s in ready]


def execute(payload: dict) -> State:
    step = payload["step"]
    context = json.dumps(payload["state_results"], indent=2)

    if step["sensitive"]:
        decision = interrupt({
            "prompt": f"Approve sensitive step #{step['id']}?",
            "description": step["description"],
        })
        if str(decision).strip().lower() not in ("yes", "approve", "y", "true"):
            return {"skipped": {step["id"]: step["description"]}}

    result = run_step(step["description"], context)
    return {"results": {step["id"]: result}}


def finalize(state: State) -> State:
    completed = "\n".join(
        f"[{i}] {r}" for i, r in sorted(state.get("results", {}).items())
    )
    ans = final_llm.invoke(
        f"Objective: {state['objective']}\n\nCompleted step results:\n"
        f"{completed}\n\nWrite the final response and a confidence score "
        f"(0-1) reflecting how fully the objective was met."
    )
    return {"final": ans.model_dump()}


# ---------- Graph ----------
def build_app(checkpointer):
    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute)
    g.add_node("finalize", finalize)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", dispatch, ["execute", "finalize"])
    # after each execution stage, loop back to re-dispatch (replan ready steps)
    g.add_conditional_edges("execute", dispatch, ["execute", "finalize"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


# ---------- CLI ----------
def main():
    objective = input("Enter objective:\n> ").strip()
    thread = {"configurable": {"thread_id": "run-1"}}

    # conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
        app = build_app(saver)

        state = app.invoke({"objective": objective}, thread)

        # Handle any interrupts (approval loop) until the graph finishes
        while "__interrupt__" in state:
            intr = state["__interrupt__"][0].value
            print(f"\n⚠  APPROVAL NEEDED — step #{intr['prompt']}")
            print(f"   {intr['description']}")
            ans = input("   Approve? [yes/no] > ").strip()
            state = app.invoke(Command(resume=ans), thread)

    print("\n" + "=" * 60)
    print("COMPLETED STEPS")
    print("=" * 60)
    for i, r in sorted(state.get("results", {}).items()):
        print(f"[{i}] {r}")

    if state.get("skipped"):
        print("\n" + "-" * 60)
        print("SKIPPED (approval denied)")
        print("-" * 60)
        for i, d in sorted(state["skipped"].items()):
            print(f"[{i}] {d}")

    final = state.get("final", {})
    print("\n" + "#" * 60)
    print("FINAL RESPONSE")
    print("#" * 60)
    print(final.get("response", "(none)"))
    print(f"\nConfidence: {final.get('confidence', 0.0):.2f}")


if __name__ == "__main__":
    main()