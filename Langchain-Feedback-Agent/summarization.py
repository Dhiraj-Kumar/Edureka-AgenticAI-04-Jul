from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware

load_dotenv()

llm = ChatOpenRouter(
    model="gpt-4o-mini"
)

# llm2 = ChatOpenRouter(
#     model="gemini"
# )

agent = create_agent(
    model=llm,
    system_prompt="You are an AI chatbot assistant",
    checkpointer=InMemorySaver(),
    middleware=SummarizationMiddleware(
        model=llm,
        # trigger=("tokens", 1000)
        trigger=("fraction", 0.6),
        keep=("messages", 10),
    )
)

while True:
    user_input=input("You: ")
    if user_input.lower()=="exit":
        break
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": user_input}
        ]
    })