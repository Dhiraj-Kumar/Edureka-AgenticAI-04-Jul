from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langsmith import traceable
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# Step 1. Loading PDF document
@traceable(name="load_documents")
def load_documents(doc_path):
    loader = PyPDFLoader(doc_path)
    return loader.load()
    

# Step 2. Split the document into chunks
@traceable(name="split_text")
def split_text(chunk_size, chunk_overlap, documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)

# Step 3. Create instance for an Embedding model

@traceable(name="generate_vectors")
def generate_vectors(chunks):
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )


    PERSIST_DIR = "./chroma_db"
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectorestore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        vectorestore = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=PERSIST_DIR
        )
    return vectorestore

@traceable(name="build_pipeline")
def build_pipeline(doc_path: str):
    docs = load_documents(doc_path)
    splitted_text = split_text(500, 100, docs)
    return generate_vectors(splitted_text)

@traceable(name="complete_rag_flow")
def complete_rag_flow():
    vectorestore = build_pipeline("./Documents/HR-Policy.pdf")
# Step 5. Create a retriever
    retriever = vectorestore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 5}
    )

# Step 6. Augmentation
    prompt = PromptTemplate(
        template="""
        You are a helpful AI assistant.
        Answer the question using ONLY the context below
        
        Context:
        {context}
        
        Question:
        {question}
        """,
        input_variables=["context", "question"]
    )

    # Step 7. Initialize the LLM
    llm =ChatOpenRouter(
        model="gpt-4o-mini"    
    )

    rag_chain = (RunnableParallel(context=retriever, question=RunnablePassthrough()) | prompt | llm | StrOutputParser())

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        answer = rag_chain.invoke(user_input, {'run_name':'rag_workflow'})
        print(f"AI: {answer}")


# Guard the demo call — otherwise it fires on every import from evaluate_rag.py
if __name__ == "__main__":
    complete_rag_flow()
