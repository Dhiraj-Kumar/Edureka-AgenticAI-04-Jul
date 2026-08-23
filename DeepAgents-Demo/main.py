from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenRouter(model="gpt-4o-mini")

class Critique(BaseModel):
    clarity: int=Field(ge=1, le=10)
    coherence: int=Field(ge=1, le=10)
    engagement: int=Field(ge=1, le=10)
    feedback: str=Field(description="Spcific, actionable improvements")

    @property
    def total(self) -> int:
        return self.clarity + self.coherence + self.engagement

class Revision(TypedDict):
    round: int
    draft: str
    critique: Critique

class State(TypedDict):
    task: str
    draft: str
    critique: Critique
    history: List[Revision]
    round: int

THRESHOLD = 27
MAX_ROUNDS = 4

critic_llm = llm.with_structured_output(Critique)

def generate(state: State) -> State:
    if not state["history"]:
        prompt = f"Write a draft for this task:\n\n{state['task']}"
    else:
        prompt= f"Task:\n{state['task']}\n\n Previous draft:\n{state['critique'].feedback}\n\nRewrite an improved draft"
    draft = llm.invoke(prompt).content
    return {"draft": draft, "round": state["round"]+1}

def critique(state: State) -> State:
    result = critic_llm.invoke(
        f"Critique this draft against the task. Score clarity, coherence, engagement (1-10) and give feedback. Task:\n\n{state['task']}\n\nDraft:\n{state['draft']}"
    )

    entry: Revision = {
        "round": state["round"],
        "draft": state["draft"],
        "critique": result
    }
    return {"critique": result, "history": state["history"] + [entry]}

def should_continue(state: State) -> Literal["generate", "__end__"]:
    if state["critique"].total >= THRESHOLD or state["round"] >= MAX_ROUNDS:
        return END
    return "generate"

graph = StateGraph(State)
graph.add_node("generate", generate)
graph.add_node("critique", critique)
graph.add_edge(START, "generate")
graph.add_edge("generate", "critique")
graph.add_conditional_edges("critique", should_continue)
app = graph.compile()

def main():
    task = input("Enter your writing task: ")
    final = app.invoke({"task": task, "history": [], "round": 0})

    for rev in final["history"]:
        c = rev["critique"]
        print(f"\n{'='*60}\nROUND {rev['round']}")
        print(f"{'='*60}\n{rev['draft']}")
        print(f"\nScores  clarity={c.clarity}  coherence={c.coherence}  "
              f"engagement={c.engagement}  total={c.total}/30")
        print(f"Feedback: {c.feedback}")

    print(f"\n{'#'*60}\nFINAL DRAFT (round {final['round']})\n{'#'*60}")
    print(final["draft"])

if __name__ == "__main__":
    main()
