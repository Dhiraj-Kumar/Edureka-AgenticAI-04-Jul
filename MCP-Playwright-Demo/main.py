import asyncio, json
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openrouter import ChatOpenRouter
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

class Book(BaseModel):
    title: str = Field(description="string title only")
    price: float = Field(description="numeric price only, no currency symbol")
    availability: str = Field(description="e.g. 'In stock', 'Out of stock'")
    rating: Optional[float] = Field(default=None, description="numeric rating, null if none")


class ScrapeResult(BaseModel):
    source_url: str = Field(description="The URL of the page that was scraped")
    product_count: int = Field(description="Total number of books extracted from the page")
    products: list[Book] = Field(description="List of all books found on the page")
    notes: Optional[str] = Field(default=None, description="Optional notes about anything unusual, e.g. missing fields or pagination")

llm = ChatOpenRouter(model="gpt-4o-mini")

PROMPT = """You extract book data from a web page using the `fetch` tool.

1. Call fetch with the given url and max_length=50000 to get the page as markdown.
2. If content looks truncated, call fetch again with start_index past where you stopped.
3. Each book tile has a title, a price (like £51.77), and an availability label ('In stock').
   Ratings appear as a word (One, Two, Three, Four, Five) — convert to the number 1-5, or null if absent.
   Ignore nav links, the sidebar categories, and pagination.
4. Extract title, price (number only), availability, rating (1-5 or null) for every book.

Reply with ONLY this JSON, no prose/fences:
{"source_url":"...","product_count":N,"products":[
 {"title":"...","price":51.77,"availability":"In stock","rating":3}],"notes":null}
price is a number (no symbol); rating is 1-5 or null. Extract ALL books on the page."""

async def main():
    client = MultiServerMCPClient({
        "playwright": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@playwright/mcp@latest", "--browser", "chromium", "--isolated"]
        }
    })

    agent = create_agent(
        model=llm,
        tools=await client.get_tools(),
        system_prompt=PROMPT
    )

    async for chunk in agent.astream({"messages": [{"role": "user", "content": "Extract all books from: https://books.toscrape.com/"}]}, stream_mode="values", config={"recursion_limit": 10}):
        chunk["messages"][-1].pretty_print()

    # res = await agent.ainvoke({"messages": [{"role": "user", "content": "Extract all books from: https://books.toscrape.com/"}]})

    # text = res["messages"][-1].content

    # if isinstance(text, list):
    #     text = "".join(b.get("text", "") for b in text if isinstance(b, dict))

    #     start = text.find("{")
    #     end = text.rfind("}") + 1
    #     json_text = text[start:end]

    #     data = json.loads(json_text)
    #     data["source_url"]="https://books.toscrape.com/"
    #     data["product_count"] = len(data.get("products"), [])
    #     return ScrapeResult.model_validate(data)


if __name__ == "__main__":
    result = asyncio.run(main())
    # print(result)
    # for i, p in enumerate(result.products, 1):
    #     print(f"{i}. {p.title} — ${p.price:.2f} | {p.availability} | {p.rating or 'no rating'}")
    # print(f"\n{result.product_count} products", f"| {result.notes}" if result.notes else "")

