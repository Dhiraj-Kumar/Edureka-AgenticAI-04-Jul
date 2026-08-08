from langgraph.graph import StateGraph, START, END
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm = ChatOpenRouter(
    model="gpt-4o-mini"
)

# Subgraph Code

class TranslateAgentState(TypedDict):
    js_code: str
    python_code: str

def translate_code(state: TranslateAgentState):
    prompt = f"""
    You are an AI coding assistant whose expertise is to translate code from javascript to python. Translate the following javascript code to python. ONLY translate the code to python without adding any additional information or approaches.
    JavaScript Code:
    {state['js_code']}
    """
    python_code = llm.invoke(prompt).content
    return {'python_code': python_code}

subgraph = StateGraph(TranslateAgentState)
subgraph.add_node('translate_code', translate_code)
subgraph.add_edge(START, 'translate_code')
subgraph.add_edge('translate_code', END)

subgraph_workflow = subgraph.compile()

# Main Graph Code

class MainAgentState(TypedDict):
    query: str
    js_code: str
    python_code: str

def generate_code(state: MainAgentState):
    prompt = f"""
    You are an AI coding assistant whose expertise it to generate javascript code based on the following user query. ONLY generate a code without adding additonal information or multiple approaches.
    \n
    User Query:
    {state['query']}
    """
    result = llm.invoke(prompt).content
    return {'js_code': result}

def convert_code(state: MainAgentState):
    result = subgraph_workflow.invoke({'js_code': state['js_code']})
    return {'python_code': result['python_code']}

maingraph = StateGraph(MainAgentState)
maingraph.add_node('generate_code', generate_code)
maingraph.add_node('convert_code', convert_code)

maingraph.add_edge(START, 'generate_code')
maingraph.add_edge('generate_code', 'convert_code')
maingraph.add_edge('convert_code', END)

maingraph_workflow = maingraph.compile()

response = maingraph_workflow.invoke({'query': 'Write a code to convert json data into csv'})
print(response['js_code'])
print(response['python_code'])