from openrouter import OpenRouter
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

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


# Documents (some text, each sentence is considered as single document)
docs = [
    "The Eiffel Tower in Paris is a wrought-iron structure known for its iconic design and panoramic city views.",
    "The Great Wall of China is an ancient series of fortifications built to protect against invasions and spans thousands of miles.",
    "The Statue of Liberty in New York symbolizes freedom and democracy, and was a gift from France to the United States."
]

# This is query based on which we want to perform similarity search
# query = "What monument represents freedom in the united states?"
query = "Which one is the national monument in US?"

# Generate the embeddings for documents and query
doc_emeddings = [generate_embeddings(doc) for doc in docs]
query_embedding = generate_embeddings(query)

scores = cosine_similarity([query_embedding], doc_emeddings)[0]

print(scores)

# Retrieve the document based in the best match
# best_index, best_score = max(enumerate(scores), key=lambda x: x[1])

# print("Most similar document:")
# print(docs[best_index])
# print(f"Similarity Score is: {best_score}")
