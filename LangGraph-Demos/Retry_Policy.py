from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from dotenv import load_dotenv
from typing import TypedDict
import time
import json

load_dotenv()

llm = ChatOpenRouter(model="gpt-4o-mini")

start = time.time()

def log(message: str):
    print(f"[{time.time() - start:5.1f}s] {message}")

outage = {"remaining": 2}

class ReviewState(TypedDict):
    review: str
    sentiment: str
    score: int


def classify_sentiment(state: ReviewState):
    if outage["remaining"] > 0:
        outage["remaining"] -= 1
        log("classify_sentiment: connection reset")
        raise ConnectionError("openrouter: connection reset")

    log("classify_sentiment: calling LLM")
    reply = llm.invoke(f"Reply with exactly one word - positive, negative or neutral:\n\n{state['review']}")
    return {"sentiment": reply.content.strip().lower()}

def extract_score(state: ReviewState):
    log("extract_score: calling LLM")
    reply = llm.invoke(f'Rate this review from 1 to 5. Reply with JSON only, no markdown: '
                       '{"score": <integer>}\n\n' + state["review"])
    data = json.loads(reply.content.strip())
    return {"score": int(data["score"])}

builder = StateGraph(ReviewState)

builder.add_node("classify_sentiment", classify_sentiment, retry_policy=RetryPolicy(
    max_attempts=4,
    initial_interval=1.0,
    retry_on=ConnectionError,
    backoff_factor=2.0
))

builder.add_node("extract_score", extract_score, retry_policy=RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    retry_on=ConnectionError,
))

builder.add_edge(START, "classify_sentiment")
builder.add_edge("classify_sentiment", "extract_score")
builder.add_edge("extract_score", END)

graph = builder.compile()

review = "Delivery took three weeks and the box arrived crushed, but the product itself works fine."

result = graph.invoke({"review": review})

print(f"Sentiment: {result['sentiment']}")
print(f"Score: {result['score']}/5")

