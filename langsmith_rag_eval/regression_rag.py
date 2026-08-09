from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1. Loading PDF document
loader = PyPDFLoader("./Documents/HR-Policy.pdf")
documents = loader.load()

# Step 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

# Step 3. Create instance for an Embedding model
embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# Step 4. Store embeddings into Vector DB
PERSIST_DIR = "./chroma_db"
if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    vectorestore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    vectorestore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR
    )

# Step 5. Create a retriever
retriever = vectorestore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 5}
)

# Step 6. Augmentation
PROMPTS = {
    "v1_basic": PromptTemplate(
        template="""You are a helpful AI assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"],
    ),
    "v2_strict": PromptTemplate(
        template="""You are an HR policy assistant.
Answer using ONLY the context below. Follow these rules:
- Quote all figures, dates, times and names exactly as written.
- If the context does not contain the answer, reply exactly:
  "The HR policy document does not specify this."
- Answer in 1-3 sentences. No preamble, no markdown.

Context:
{context}

Question:
{question}

Answer:""",
        input_variables=["context", "question"],
    ),
}

# Step 7. Initialize the LLM
llm =ChatOpenRouter(
    model="gpt-4o-mini"    
)

def make_pipeline(version: str):
    """Returns a target function bound to one prompt version."""
    selected = PROMPTS[version]

    def pipeline(inputs: dict) -> dict:
        question = inputs["question"]
        docs = retriever.invoke(question)
        context = "\n".join(d.page_content for d in docs)
        response = (selected | llm).invoke({"context": context, "question": question})
        return {"answer": response.content, "context": context}

    return pipeline


# Guard the demo call — otherwise it fires on every import from evaluate_rag.py
if __name__ == "__main__":
    rag_pipeline = make_pipeline("v2_strict")
    result = rag_pipeline({"question": "How many leaves an employee can take in a year?"})
    print(result["answer"])