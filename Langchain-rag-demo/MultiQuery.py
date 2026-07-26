# from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
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

# Step 3. Create instance for an embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# Step 4. Store Embeddings into vector db
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

llm = ChatOpenAI(
    model="gpt-4o-mini"
)

# Step 5. Create a retriver
retriever = MultiQueryRetriever.from_llm(
    retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k": 5}
    ),
    llm=llm
)


# Step 6. Augmentation
prompt = PromptTemplate(
    template = """
    You are an AI assistant.
    Answer the question using ONLY the context below

    Context:
    {context}

    Question:
    {question}
    """,
    input_variables=["context", "question"]
)

# Step 7. Initialize LLM

# Step 8. Create a pipeline chain
rag_chain = (RunnableParallel(context=retriever, question=RunnablePassthrough())) | prompt | llm |StrOutputParser()

while True:
    user_input = input("You: ")
    if user_input.lower()=="exit":
        break
    answer = rag_chain.invoke(user_input)
    print(f"AI: {answer}")