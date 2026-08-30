from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openrouter import ChatOpenRouter
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()

pdf_loader = PyPDFLoader("./Documents/HR-Policy.pdf")
pdf_documents = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
)
document_chunks = text_splitter.split_documents(pdf_documents)

embedding_model = OpenAIEmbeddings(base_url="https://openrouter.ai/api/v1", model="text-embedding-3-small")

vectore_store = Chroma.from_documents(
    documents=document_chunks,
    embedding=embedding_model,
    collection_name="rag_collection",
    persist_directory="./chroma_db"
)

retriever = vectore_store.as_retriever(search_kwargs={"k":5})

@tool
def search_company_policy(query: str) -> str:
    """Search the company policy document for information relevant to the query.
    Use this whenever the user asks about company policies, leave, benefits,
    or any topic that might be covered in the policy document."""
    retrieved_docs = retriever.invoke(query)
    combined_text = ""
    for doc in retrieved_docs:
        combined_text = combined_text + doc.page_content + "\n\n"
    return combined_text.strip()

llm = ChatOpenRouter(model="openai/gpt-4o-mini")

agent = create_agent(
    model=llm,
    system_prompt=(
        "You are a helpful assistant. Use the search_company_policy tool "
        "whenever the question requires information from the company policy document. "
        "For general questions that do not need the document, answer directly "
        "without calling the tool."
    ),
    tools=[search_company_policy]
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "Who is the prime minister of India"}
    ]
})
# What is the company's leave policy for new employees?
final_answer = result["messages"][-1].content
print(final_answer)