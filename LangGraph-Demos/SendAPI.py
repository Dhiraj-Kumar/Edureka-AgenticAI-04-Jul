from langchain_openrouter import ChatOpenRouter
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send
from dotenv import load_dotenv
from typing import TypedDict
from typing import Annotated
import time
import operator

load_dotenv()

llm = ChatOpenRouter(model="gpt-4o-mini")

TOPICS = ["Cyber Security", "Quantum Computing", "AI", "Vector Databases"]

class ResearchState(TypedDict):
    topics: list[str]
    articles: Annotated[list[str], operator.add]
    final_report: str

class TopicState(TypedDict):
    topic: str

def fan_out(state: ResearchState):
    sends = []
    for topic in state['topics']:
        sends.append(Send("research_topic", {"topic": topic}))
    return sends

def research_topic(state: TopicState):
    topic = state["topic"]
    print(f"Worker Staretd: {topic}")

    response = llm.invoke(f"In 2 sentences, explain: {topic}")
    content= response.content
    print(f"\nWorker Finished: {topic}")
    print(f"{content}\n")
    return {"articles": [f"## {topic}\n\n{content}"]}

def compile_report(state: ResearchState):
    return {"final_report": "\n\n".join(state["articles"])}

builder = StateGraph(ResearchState)

builder.add_node("research_topic", research_topic)
builder.add_node("compile_report", compile_report)

builder.add_conditional_edges(START, fan_out, ["research_topic"])
builder.add_edge("research_topic", "compile_report")
builder.add_edge("compile_report", END)

graph = builder.compile()

started = time.time()
result = graph.invoke({"topics": TOPICS})
elapsed = time.time() - started

print("="*60)
print("FINAL REPORT")
print("="*60)
print(result["final_report"])
print("="*60)
print(f"{len(TOPICS)} topics completed in {elapsed:.1f}s")

