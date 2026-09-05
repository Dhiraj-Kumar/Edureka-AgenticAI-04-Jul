import re
from typing import TypedDict, List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

pdf_loader = PyPDFLoader("./Documents/HR-Policy.pdf")
pdf_pages = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100)
document_chunks = text_splitter.split_documents(pdf_pages)

embedding_model = OpenAIEmbeddings(base_url="https://openrouter.ai/api/v1", model="text-embedding-3-small")

vector_store = Chroma.from_documents(
    documents=document_chunks,
    embedding=embedding_model,
    collection_name="self_rag_pdf",
    persist_directory="./chroma_db"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenRouter(model="gpt-4o-mini")

# Graph Starts Here
class SelfRagState(TypedDict):
    question: str
    documents: List[str]
    scored_documents: List[Tuple[str, int]]
    relevant_documents: List[str]
    answer: str
    regeneration_count: int

RELEVANCE_THRESHOLD = 6 # out of 10

def retrieve_node(state: SelfRagState) -> SelfRagState:
    retrieved_docs = retriever.invoke(state["question"])
    document_texts = []
    for doc in retrieved_docs:
        document_texts.append(doc.page_content)
    state["documents"] = document_texts
    return state

relevance_scoring_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rate how relevant the document is to answering the question, "
               "on a scale from 0 to 10, where 0 means completely irrelevant "
               "and 10 means directly answers the question. "
               "Respond with only the number, nothing else."),
    ("human", "Question: {question}\n\nDocument:\n{document}")
])

def parse_score(raw_text: str) -> int:
    match = re.search(r"\d+", raw_text)
    if match:
        score = int(match.group())
        return max(0, min(score, 10))
    return 0

def grade_documents_node(state: SelfRagState) -> SelfRagState:
    scored_documents = []
    scoring_chain = relevance_scoring_prompt | llm

    for document_text in state["documents"]:
        scoring_result = scoring_chain.invoke({
            "question": state["question"],
            "document": document_text
        })
        score = parse_score(scoring_result.content)
        scored_documents.append((document_text, score))

    scored_documents.sort(key=lambda pair: pair[1], reverse=True)
    state["scored_documents"] = scored_documents

    relevant_documents = []
    for document_text, score in scored_documents:
        if score >= RELEVANCE_THRESHOLD:
            relevant_documents.append(document_text)
    state["relevant_documents"] = relevant_documents

    return state

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the provided context. "
               "If the context does not contain the answer, say you don't know."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

def generate_node(state: SelfRagState) -> SelfRagState:
    context_text = "\n\n".join(state["relevant_documents"])
    generation_chain = generate_prompt | llm
    generation_result = generation_chain.invoke({
        "context": context_text,
        "question": state["question"]
    })
    state["answer"] = generation_result.content
    state["regeneration_count"] = state.get("regeneration_count", 0) + 1
    return state

groundedness_prompt = ChatPromptTemplate.from_messages([
    ("system", "Check if the answer is fully supported by the context. "
               "Answer with only one word: 'grounded' or 'not_grounded'."),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
])

def check_groundedness(state: SelfRagState) -> str:
    if len(state["relevant_documents"]) == 0:
        return "no_documents"

    context_text = "\n\n".join(state["relevant_documents"])
    grading_chain = groundedness_prompt | llm
    grading_result = grading_chain.invoke({
        "context": context_text,
        "answer": state["answer"]
    })
    verdict = grading_result.content.strip().lower()

    if verdict.startswith("grounded"):
        return "grounded"

    if state["regeneration_count"] >= 2:
        return "give_up"

    return "not_grounded"

def no_documents_node(state: SelfRagState) -> SelfRagState:
    state["answer"] = "I could not find sufficiently relevant information in the document to answer this question."
    return state

graph_builder = StateGraph(SelfRagState)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("grade_documents", grade_documents_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_node("no_documents", no_documents_node)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "grade_documents")

def route_after_grading(state: SelfRagState) -> str:
    if len(state["relevant_documents"]) == 0:
        return "no_documents"
    return "generate"

graph_builder.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "no_documents": "no_documents",
        "generate": "generate"
    }
)

graph_builder.add_conditional_edges(
    "generate",
    check_groundedness,
    {
        "grounded": END,
        "not_grounded": "generate",
        "give_up": END,
        "no_documents": "no_documents"
    }
)
graph_builder.add_edge("no_documents", END)

self_rag_graph = graph_builder.compile()

image = self_rag_graph.get_graph().draw_mermaid_png()
with open("self_graph.png", mode="wb") as f:
    f.write(image)

final_state = self_rag_graph.invoke(
    {
        "question": "What is the company's leave policy for new employees?",
        "documents": [],
        "scored_documents": [],
        "relevant_documents": [],
        "answer": "",
        "regeneration_count": 0
    }
)

print("Document relevance scores:")
for document_text, score in final_state["scored_documents"]:
    print(f"  Score {score}/10: {document_text}...")

print("\nFinal Answer:")
print(final_state["answer"])
print(final_state["regeneration_count"])