from langgraph.graph import START, END, StateGraph
from typing import TypedDict
from mcp_crew.crew import McpCrew

class GraphState(TypedDict):
    result: str
    city: str

def call_crew(state: GraphState) -> GraphState:
    crew_result = McpCrew().crew().kickoff(inputs={'city': state['city']})
    return {'result': str(crew_result)}

graph = StateGraph(GraphState)
graph.add_node("call_crew", call_crew)
graph.add_edge(START, "call_crew")
graph.add_edge("call_crew", END)

workflow = graph.compile()
result = workflow.invoke({'city': 'Mumbai'})
print(result)