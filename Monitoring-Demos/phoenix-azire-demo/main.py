import numpy as np
from langchain_core.documents import Document
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openrouter import ChatOpenRouter
from langchain_openai import OpenAIEmbeddings
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from dotenv import load_dotenv
import os
load_dotenv()


# Step 1. Connect to Phoenix
tracer_provider = register(project_name="rag-drift-demo", auto_instrument=True)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
tracer = tracer_provider.get_tracer(__name__)

#  Step 2. Create RAG Pipeline
model = ChatOpenRouter(model="openai/gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="openai/text-embedding-3-small", 
                              api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1")

KNOWLEDGE_BASE = [
    "NimbusCloud's free tier includes 5GB storage and 100 API calls/day.",
    "NimbusCloud's Pro tier costs $29/month: 500GB storage, unlimited API calls.",
    "Refunds are available within 14 days of purchase.",
    "Two-factor authentication is mandatory for Enterprise accounts.",
]

docs = [Document(page_content=t) for t in KNOWLEDGE_BASE]
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_drift")
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

REFERENCE_QUERIES = [
    "How much does the Pro tier cost?",
    "What's included in the free tier?",
    "How do I get a refund?",
    "Is 2FA required for Enterprise?",
]

reference_embeddings = np.array(embeddings.embed_documents(REFERENCE_QUERIES))
reference_centroid = reference_embeddings.mean(axis=0)

def compute_drift_score(query: str) -> float:
    """Cosine distance between a new query and the reference centroid.
    Higher = more different from typical traffic = more likely drifted."""
    query_embedding = np.array(embeddings.embed_query(query))
    cosine_similarity = np.dot(query_embedding, reference_centroid) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(reference_centroid)
    )
    return 1 - cosine_similarity


DRIFT_ALERT_THRESHOLD = 0.35

def rag_answer_with_drift_check(question: str) -> tuple[str, float]:
    with tracer.start_as_current_span("rag_query") as span:
        drift_score = compute_drift_score(question)
        span.set_attribute("drift.score", float(drift_score))
        span.set_attribute("drift.is_alert", bool(drift_score > DRIFT_ALERT_THRESHOLD))
        span.set_attribute("input.value", question)

        docs = retriever.invoke(question)
        context = "\n".join(d.page_content for d in docs)
        response = model.invoke(f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely.")

    return response.content, drift_score

while True:
    question = input("Ask a question: ").strip()
    if question.lower() in ("quit", "exit"):
        break
    if not question:
        continue

    answer, drift = rag_answer_with_drift_check(question)

    print(f"\nAgent: {answer}")
    alert = " 🚨 DRIFT ALERT — this query is unusual vs. reference traffic" if drift > DRIFT_ALERT_THRESHOLD else ""
    print(f"Drift score: {drift:.3f} (threshold: {DRIFT_ALERT_THRESHOLD}){alert}\n")