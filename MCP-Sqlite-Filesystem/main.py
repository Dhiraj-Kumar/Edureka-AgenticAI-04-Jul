from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import shutil
import os
import asyncio

load_dotenv()

UVX = shutil.which("uvx.cmd") or shutil.which("uvx")
NPX = shutil.which("npx.cmd") or shutil.which("npx")

async def main():
    outputdir = os.path.abspath("output")
    os.makedirs(outputdir, exist_ok=True)

    client = MultiServerMCPClient({
        "sqlite": {
            "transport": "stdio",
            "command": UVX,
            "args": [
                "--with", "mcp==1.3.0",
                "mcp-server-sqlite",
                "--db-path", "shopease.db"
            ]
        },
        "filesystem": {
            "transport": "stdio",
            "command": NPX,
            "args": ["-y", "@modelcontextprotocol/server-filesystem", outputdir]
        }
    })

    print("loading tools...")
    tools = await client.get_tools()

    print("tools loaded: ", [t.name for t in tools])

    llm = ChatOpenRouter(model="gpt-4o-mini")
    agent = create_agent(
        model=llm,
        tools=tools
    )

    prompt = """
        You have access to a SQLite database (table: orders) and a filesystem restricted to the output folder.

        Do the following:
        1. Query revenue by category: SUM(quantity * price) grouped by category.
        2. Identify the top-selling product by total quantity sold.
        3. Summarize order counts by city.
        4. Write a clean Markdown report with these three sections and save it as 'sales_report.md' in the output folder.

        Use the SQLite tools to run queries and the filesystem tools to write the file.
        Write the file to 'sales_report.md' (the filesystem is already rooted at the output folder — do not prefix the path with 'output/').
    """

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    print("DONE")

asyncio.run(main())