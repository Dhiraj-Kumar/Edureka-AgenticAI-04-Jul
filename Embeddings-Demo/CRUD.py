from openrouter import OpenRouter
from dotenv import load_dotenv
import os
import chromadb

load_dotenv()


# Funtion to generate embeddings
def generate_embeddings(text: str):
    client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
    response = client.embeddings.generate(
        model="openai/text-embedding-3-small",
        input=text,
        dimensions=534
    )
    return response.data[0].embedding


# Path of the chromadb database file
CHROMA_PATH = "./chroma_storage"

# Create an instance of Chroma Client
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Create a new collection in db
collection = client.get_or_create_collection(
    name="travel_knowledge"
)

# Example Documents
docs = [
    "The Eiffel Tower in Paris is a wrought-iron structure known for its iconic design and panoramic city views.",
    "The Great Wall of China is an ancient series of fortifications built to protect against invasions and spans thousands of miles.",
    "The Statue of Liberty in New York symbolizes freedom and democracy, and was a gift from France to the United States."
]

# Generate embeddings for each document
doc_emeddings = [generate_embeddings(doc) for doc in docs]


# Add document embeddings in vector database
def create_documents():
    collection.add(
        documents=docs,
        embeddings=doc_emeddings,
        ids=["doc1", "doc2", "doc3"],
        metadatas=[
            {"source": "sample1"},
            {"source": "sample2"},
            {"source": "sample3"},
        ]
    )
    print("Documents successfully added to ChromaDB")


# Retrieve all documents
def get_documents():
    result = collection.get(include=['embeddings'])
    print(result['embeddings'])


# Update the documents based on document id
def update_documents(id: str):
    collection.update(
        ids=[id],
        embeddings=[generate_embeddings(
            "The Taj Mahal is located in Agra, India, and is a UNESCO World Heritage Site."
        )],
        documents=[
            "The Taj Mahal is located in Agra, India, and is a UNESCO World Heritage Site."
        ]
    )
    print("Document Updated")


# Delete document based on Id
def delete_documents(id: str):
    collection.delete(ids=[id])
    print("Document Delete")


create_documents()
