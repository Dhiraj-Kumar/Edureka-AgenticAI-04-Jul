from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenRouter(
    model="gpt-4o-mini"
)

def ChatbotNode(state: MessagesState) -> MessagesState:
    result = llm.invoke(state["messages"])
    return {'messages': [result]}


graph = StateGraph(MessagesState)
graph.add_node('ChatbotNode', ChatbotNode)
graph.add_edge(START, 'ChatbotNode')
graph.add_edge('ChatbotNode', END)

workflow = graph.compile(checkpointer=InMemorySaver())

# image = workflow.get_graph().draw_mermaid_png()
# with open("chatbot_graph.png", mode="wb") as f:
#     f.write(image)

while True:
    user_input=input("You: ")
    if user_input.lower()=="exit":
        break
    response = workflow.invoke({'messages': HumanMessage(content=user_input)}, config={'configurable': {'thread_id': 1}})
    print(response['messages'][-1].content)