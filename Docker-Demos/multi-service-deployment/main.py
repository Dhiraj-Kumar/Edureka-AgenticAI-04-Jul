import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.agents import create_agent
# from langchain_openrouter import ChatOpenRouter
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

DB_URI = os.getenv("DB_URI")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    # api_key=os.getenv("OPENROUTER_API_KEY")
)

SYSTEM_PROMPT = """
You are an AI chat assitant who answers queries with a bit of humour. Use emojis to decorate your responses. Be polite and professional.
"""

app = FastAPI(title="Chat Agent API")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        agent = create_agent(
            model=llm,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer
        )
        response = agent.invoke(
            {"messages": [{"role": "user", "content": req.message}]},
            config={"configurable": {"thread_id": req.thread_id}}
        )
    return ChatResponse(reply=response["messages"][-1].content, thread_id=req.thread_id)


@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)