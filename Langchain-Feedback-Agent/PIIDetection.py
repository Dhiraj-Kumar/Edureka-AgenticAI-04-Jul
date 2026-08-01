from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import PIIMiddleware

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
    middleware=PIIMiddleware(
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware("api_key", detector="sk-[a-zA-Z0-9][20,]", strategy="redact", apply_to_input=True)
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