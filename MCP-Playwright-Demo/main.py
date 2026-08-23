import asyncio, json
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_openrouter import ChatOpenRouter
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

class Book(BaseModel):
    title: str = Field(description="Product title exactly as shown")
    price: float = Field(description="Numeric price, currency symbol stripped")
    availability: Optional[str] = Field(default=None, description="Stock/availability text if present")
    rating: Optional[str] = Field(default=None, description="Star rating text if shown, e.g. 'Three'")


class ScrapeResult(BaseModel):
    source_url: str = Field(description="Page the data was extracted from")
    product_count: int = Field(description="Number of products captured")
    products: List[Book] = Field(description="Extracted products")
    notes: Optional[str] = Field(default=None, description="Anything notable, e.g. pagination present")

llm = ChatOpenRouter(model="gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a web data-extraction agent. Use browser_navigate once to open the "
    "supplied page. Then use browser_snapshot to inspect the same open page. "
    "Extract every product tile on that single page: title, price as a number "
    "without the currency symbol, availability text, and star rating if shown. "
    "Do not follow pagination. Do not open individual product pages. "
    "If the page is already open, do not navigate again. "
    "Set product_count to the number of products captured and source_url to the supplied URL."
)

async def scrape(url: str) -> ScrapeResult:
    client = MultiServerMCPClient(
        {
            "playwright": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@playwright/mcp@latest", "--browser", "chromium", "--isolated", "--headless"]
            }
        }
    )

    async with client.session("playwright") as session:
        tools = await load_mcp_tools(session)
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            response_format=ScrapeResult
        )

        result = await asyncio.wait_for(
            agent.ainvoke(
                {"messages": [{"role": "user", "content": f"Extract the books catalogue from this page: {url}"}]},
                config={"recursion_limit": 30}
            ),
            timeout=180
        )

        snapshot: ScrapeResult = result["structured_response"]
        snapshot.product_count = len(snapshot.products)
        snapshot.source_url = url
        return snapshot

if __name__ == "__main__":
    url = input("Enter URL: ")
    result = asyncio.run(scrape(url))
    for i, p in enumerate(result.products, 1):
        print(f"{i}. {p.title} — £{p.price:.2f} | {p.availability or '—'} | {p.rating or 'no'} rating")
    print(f"\n{result.product_count} products from {result.source_url}")
    if result.notes:
        print(f"Notes: {result.notes}")