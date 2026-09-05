from typing import TypedDict, List
from langchain_openai import OpenAIEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
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
    collection_name="crag_hr_policy",
    persist_directory="./chroma_db"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenRouter(model="gpt-4o-mini")

web_search_tool = TavilySearch(max_results=5, topic="general", include_answer=True)

class CragState(TypedDict):
    question: str
    documents: List[str]
    grade: str
    refined_context: str
    web_results: str
    answer: str

def retrieve_node(state: CragState) -> CragState:
    retrieved_docs = retriever.invoke(state["question"])
    document_texts = []
    for doc in retrieved_docs:
        document_texts.append(doc.page_content)
    state["documents"] = document_texts
    return state

grading_prompt = ChatPromptTemplate.from_messages([
    ("system", "You evaluate whether the retrieved documents are sufficient to answer "
               "the question. Respond with only one word:\n"
               "'correct' - the documents clearly and fully answer the question\n"
               "'ambiguous' - the documents are partially relevant but incomplete\n"
               "'incorrect' - the documents do not help answer the question at all"),
    ("human", "Question: {question}\n\nRetrieved documents:\n{documents}")
])

def grade_documents_node(state: CragState) -> CragState:
    combined_documents = "\n\n".join(state["documents"])
    grading_chain = grading_prompt | llm
    grading_result = grading_chain.invoke({
        "question": state["question"],
        "documents": combined_documents
    })
    verdict = grading_result.content.strip().lower()

    if "correct" in verdict and "incorrect" not in verdict:
        state["grade"] = "correct"
    elif "ambiguous" in verdict:
        state["grade"] = "ambiguous"
    else:
        state["grade"] = "incorrect"

    return state

refine_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract and rewrite the parts of the following documents that provide "
               "useful background for answering the question, even if they don't fully "
               "answer it on their own — for example, keep specific numbers, policies, "
               "or facts that could be compared against other information. "
               "Only respond with 'NOTHING_RELEVANT' if the documents are entirely unrelated "
               "to the topic of the question."),
    ("human", "Question: {question}\n\nDocuments:\n{documents}")
])

def refine_knowledge_node(state: CragState) -> CragState:
    combined_documents = "\n\n".join(state["documents"])
    refine_chain = refine_prompt | llm
    refine_result = refine_chain.invoke({
        "question": state["question"],
        "documents": combined_documents
    })
    refined_text = refine_result.content.strip()

    if refined_text == "NOTHING_RELEVANT":
        state["refined_context"] = ""
    else:
        state["refined_context"] = refined_text

    return state

def web_search_node(state: CragState) -> CragState:
    search_response = web_search_tool.invoke({"query": state["question"]})

    result_parts = []
    if isinstance(search_response, dict) and "results" in search_response:
        for result_item in search_response["results"]:
            title = result_item.get("title", "")
            content = result_item.get("content", "")
            result_parts.append(f"{title}: {content}")
    else:
        result_parts.append(str(search_response))

    state["web_results"] = "\n\n".join(result_parts)
    return state

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using the provided context. The context may combine "
               "internal policy details with external legal/factual information — use both "
               "together and reason across them to form your answer, even if no single "
               "passage states the answer directly. Only say you don't know if the context "
               "is genuinely insufficient to reason from."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

def generate_node(state: CragState) -> CragState:
    context_parts = []

    if state.get("refined_context"):
        context_parts.append(
            "From company policy document:\n" + state["refined_context"])

    if state.get("web_results"):
        context_parts.append("From web search:\n" + state["web_results"])

    full_context = "\n\n".join(context_parts)

    generation_chain = generate_prompt | llm
    generation_result = generation_chain.invoke({
        "context": full_context,
        "question": state["question"]
    })
    state["answer"] = generation_result.content
    return state

graph_builder = StateGraph(CragState)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("grade_documents", grade_documents_node)
graph_builder.add_node("refine_knowledge", refine_knowledge_node)
graph_builder.add_node("web_search", web_search_node)
graph_builder.add_node("generate", generate_node)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "grade_documents")

def route_after_grading(state: CragState) -> str:
    if state["grade"] == "correct":
        return "refine_only"
    if state["grade"] == "ambiguous":
        return "refine_and_search"
    return "search_only"

graph_builder.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "refine_only": "refine_knowledge",
        "refine_and_search": "refine_knowledge",
        "search_only": "web_search"
    }
)


def route_after_refining(state: CragState) -> str:
    if state["grade"] == "ambiguous":
        return "also_search"
    return "generate"

graph_builder.add_conditional_edges(
    "refine_knowledge",
    route_after_refining,
    {
        "also_search": "web_search",
        "generate": "generate"
    }
)

graph_builder.add_edge("web_search", "generate")
graph_builder.add_edge("generate", END)

crag_graph = graph_builder.compile()

image = crag_graph.get_graph().draw_mermaid_png()
with open("crag_graph.png", mode="wb") as f:
    f.write(image)

initial_state = {
    "question": "Does Rikalp Capital's 18 earned leaves per year comply with the current Factories Act or Shops and Establishment Act requirements in Rajasthan?",
    "documents": [],
    "grade": "",
    "refined_context": "",
    "web_results": "",
    "answer": ""
}

final_state = crag_graph.invoke(initial_state)

print("Grade:", final_state["grade"])
print("\nRefined context:\n", final_state["refined_context"])
print("\nWeb results:\n", final_state["web_results"])
print("\nAnswer:\n", final_state["answer"])