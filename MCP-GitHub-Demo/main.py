import asyncio
import os
from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

load_dotenv()

GITHUB_PAT = os.getenv("GITHUB_PAT")
TOOLSETS = "context,issues,pull_requests,repos"

client = MultiServerMCPClient({
    "github": {
        "transport": "streamable_http",
        "url": "https://api.githubcopilot.com/mcp/",
        "headers": {
            "Authorization": f"Bearer {GITHUB_PAT}",
            "X-MCP_Toolsets": TOOLSETS,
            "X-MCP-Readonly": "true"
        }
    }
})

SYSTEM_PROMPT = """You are a repository triage assistant for open-source maintainers.
You have read-only access to GitHub via MCP tools. You never modify anything.

You can:
- List and group open issues by theme; label each as likely BUG or FEATURE REQUEST with a one-line reason.
- Suggest a priority (High / Medium / Low) with brief justification.
- Summarize open pull requests: what each changes, its status, and anything blocking it.
- Describe a repository using its README and file tree.

Always fetch real data with the tools before answering — never guess.
Be concise and well-structured. Group related items instead of dumping raw lists.
When you make a judgment (bug vs feature, priority), state the one-line reason so a maintainer can trust it."""


DEMOS = {
    "1": (
        "Triage board",
        "For fastapi/fastapi: pull the open issues, group them by theme, mark each group "
        "as mostly BUGS or mostly FEATURE REQUESTS, and give me a High/Medium/Low priority "
        "for each group with a one-line reason. End with the 3 issues you'd tackle first.",
    ),
    "2": (
        "PR review queue",
        "Summarize the open pull requests on pallets/flask. For each, tell me in one line what it "
        "changes and whether it looks ready to merge, needs review, or is stalled. "
        "Then tell me which one a maintainer should look at first and why.",
    ),
    "3": (
        "Repo onboarding",
        "I'm new to psf/requests. Describe what this repo is using its README and top-level file "
        "tree, point me to where the core logic lives, and tell me where I'd start if I wanted to "
        "fix a small bug.",
    ),
    "4": (
        "Bug vs feature split",
        "Look at the 15 most recent open issues on tiangolo/typer. Split them into likely bugs vs "
        "feature requests, and flag any that are actually questions/support rather than either. "
        "Give me the counts and the reasoning behind a few of the trickier calls.",
    ),
}

async def main():
    print("loading tools...")
    tools = await client.get_tools()
    print("tools: ", [t.name for t in tools])

    llm = ChatOpenRouter(model="gpt-4o-mini")
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT
    )

    choice = input("Pick an option from 1 to 4")
    number, prompt = DEMOS.get(choice)

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()


asyncio.run(main())