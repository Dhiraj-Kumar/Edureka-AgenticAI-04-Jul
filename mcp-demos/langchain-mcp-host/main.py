import asyncio
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenRouter(
    model="gpt-4o-mini"
)

client1 = MultiServerMCPClient(
    {
        "TaskPlanner": {
            "command": "uv",
            "transport": "stdio",
            "args": [
                "run",
                "--with",
                "mcp[cli]==2.0.0",
                "mcp",
                "run",
                "F:\\My Batches\\Edureka-AgenticAI-04-Jul\\mcp-demos\\mcp-task-planner\\main.py"
            ]
        }
    }
)

client2 = MultiServerMCPClient(
    {
        "TaskPlanner": {
            "command": "uv",
            "transport": "stdio",
            "args": [
                "run",
                "--with",
                "mcp[cli]==2.0.0",
                "mcp",
                "run",
                "F:\\My Batches\\Edureka-AgenticAI-04-Jul\\mcp-demos\\mcp-task-planner\\main2.py"
            ]
        }
    }
)

async def main():
    tools = await client.get_tools()
    agent = create_agent(
        model=llm,
        tools=tools
    )

    while True:
        user_input = input("You: ")
        if user_input.lower()=="exit":
            break
        response = await agent.ainvoke({"messages": [{"role": "user", "content": user_input}]})
        print(f"AI: {response['messages'][-1].content}")

if __name__ == "__main__":
    asyncio.run(main())