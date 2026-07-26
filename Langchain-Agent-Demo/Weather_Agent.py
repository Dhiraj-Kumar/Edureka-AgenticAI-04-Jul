from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch
from langchain.tools import tool
from dotenv import load_dotenv
import requests
import os

load_dotenv()

llm = ChatOpenRouter(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

@tool
def getWeatherInfo(city: str):
    """
    Get the current weather for a city using openweathermap api
    """
    url="https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": os.getenv("WEATHER_API_KEY"),
        "units": "metric"
    }

    response = requests.get(url, params)

    if response.status_code != 200:
        return f"Unable to fetch data from weather api"
    return response.json()

tavily_tool = TavilySearch(max_results=3)

agent = create_agent(
    model=llm,
    system_prompt="""
    You are a a helpful AI agent who answers queries with a bit of humour. Use emojis to decorate your responses. Be polite and professional.
    """,
    checkpointer=InMemorySaver(),
    tools=[tavily_tool]
)

while True:
    user_input = input("You: ")
    if user_input.lower()=="exit":
        break
    response = agent.invoke(
        {
            "messages": [
                {"role":"user", "content": user_input}
            ]
        }, config={"configurable": {"thread_id": 1}}
    )

    print(response["messages"][-1].content)