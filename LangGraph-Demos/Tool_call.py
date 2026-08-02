from langgraph.graph import START, END, StateGraph
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode, tools_condition
import requests
import os


load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@tool
def call_stock_api(symbol: str):
    """
    Gets the stock data from an online API based on stock symbol
    """
    result = requests.get(f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.getenv('STOCK_API_KEY')}")
    return result.json()

@tool
def make_payment():
    """
    Fake payment simulation for developmnent environment.
    Makes payment for the required stock and quantity. Return the proper message based on the transaction status.
    """
    decision = interrupt("Do you want to make a payment (yes/no)")
    if decision.lower() == "yes":
        return {
            "status": "success",
            "message": "Order placed! Payment successful"
        }
    else:
        return {
           "status": "cancelled",
           "message": "Payment denied! User cancelled the transaction"
        }

tools = [call_stock_api, make_payment]
llm_with_tools = llm.bind_tools(tools=tools)

def agent_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {'messages': [response]}

tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node('agent_node', agent_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'agent_node')
graph.add_conditional_edges('agent_node', tools_condition)
graph.add_edge('tools', 'agent_node')
workflow = graph.compile(checkpointer=InMemorySaver())

# image = workflow.get_graph().draw_mermaid_png()
# with open("tool_call.png", mode="wb") as f:
#     f.write(image)

config={'configurable': {'thread_id': 1}}

while True:
    user_input=input("You: ")
    if user_input.lower()=="exit":
        break
    response = workflow.invoke({'messages': HumanMessage(content=user_input)}, config=config)
    interrupt_response = response.get("__interrupt__", [])
    if interrupt_response:
        user_decision = input("Your response: ").lower()
        response = workflow.invoke(
            Command(resume=user_decision),
            config=config
        )

    print(f"AI: {response['messages'][-1].content}")