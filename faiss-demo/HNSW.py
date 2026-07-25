from openrouter import OpenRouter
import faiss
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

# Funtion to generate embeddings
def generate_embeddings(text: str):
    client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
    response = client.embeddings.generate(
        model="openai/text-embedding-3-small",
        input=text
    )
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype='float32')

# Sample documents
documents = [
    "The cat sat on the mat in the living room.",
    "A dog was resting on the rug near the fireplace.",
    "The stock market crashed sharply today.",
    "Shares fell across all major indices this afternoon.",
    "Python is a popular programming language for data science.",
    "Machine learning models require large amounts of training data.",
]

# Get embeddings from OpenAI
embeddings = generate_embeddings(documents)

# Build FAISS index
M = 32 #(typical range is 16-64)
index = faiss.IndexHNSWFlat(1536, M)
# Store embeddings in index
index.add(embeddings)
print("Vectors stored: ", index.ntotal)

# Embed the query the same way, then search
query = "A pet was lying near the fire"
query_vector = generate_embeddings([query])

distances, indices = index.search(query_vector, 3)

for i in range(3):
    index = indices[0][i]
    dist = distances[0][i]
    print(f"{i+1}. {documents[index]} (distance={dist:.3f})")