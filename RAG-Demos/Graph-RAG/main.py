import re
import networkx as nx
import matplotlib.pyplot as plt
from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

pdf_loader = PyPDFLoader("./Documents/HR-Policy.pdf")
pdf_pages = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=100)
document_chunks = text_splitter.split_documents(pdf_pages)

llm = ChatOpenRouter(model="gpt-4o-mini")

graph_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Role", "Policy",
                   "Benefit", "Amount", "Duration", "Department"],
    allowed_relationships=["REPORTS_TO", "ESCALATES_TO",
                           "ENTITLES_TO", "REQUIRES", "APPLIES_TO", "HAS_LIMIT"]
)

graph_documents = graph_transformer.convert_to_graph_documents(document_chunks)

print("KNOWLEDGE GRAPH CONSTRUCTION — EXTRACTION RESULTS")

for chunk_index in range(len(graph_documents)):
    graph_document = graph_documents[chunk_index]
    if len(graph_document.nodes) == 0:
        continue

    print(f"\n--- Chunk {chunk_index} ---")
    print("Extracted nodes:")
    for node in graph_document.nodes:
        print(f"  ({node.type}) {node.id}")

    print("Extracted relationships:")
    for relationship in graph_document.relationships:
        print(
            f"  {relationship.source.id} --{relationship.type}--> {relationship.target.id}")

knowledge_graph = nx.DiGraph()

for graph_document in graph_documents:
    for node in graph_document.nodes:
        knowledge_graph.add_node(node.id, type=node.type)

    for relationship in graph_document.relationships:
        knowledge_graph.add_edge(
            relationship.source.id,
            relationship.target.id,
            relation=relationship.type
        )
print(f"\nFinal graph: {knowledge_graph.number_of_nodes()} nodes, "
      f"{knowledge_graph.number_of_edges()} edges")

def find_matching_nodes(question: str, graph: nx.DiGraph) -> list:
    question_lower = question.lower()
    matched_nodes = []
    for node_id in graph.nodes:
        node_words = re.findall(r"\w+", node_id.lower())
        for word in node_words:
            if len(word) > 3 and word in question_lower:
                matched_nodes.append(node_id)
                break
    return matched_nodes

def multi_hop_traversal(matched_nodes: list, graph: nx.DiGraph, max_hops: int = 3) -> list:
    all_facts = []
    visited_nodes = set(matched_nodes)
    current_layer = set(matched_nodes)

    for hop_number in range(1, max_hops + 1):
        hop_facts = []
        next_layer = set()

        for node_id in current_layer:
            if node_id not in graph:
                continue

            for _, target_id, edge_data in graph.out_edges(node_id, data=True):
                fact = f"{node_id} --{edge_data['relation']}--> {target_id}"
                hop_facts.append(fact)
                if target_id not in visited_nodes:
                    next_layer.add(target_id)
                    visited_nodes.add(target_id)

            for source_id, _, edge_data in graph.in_edges(node_id, data=True):
                fact = f"{source_id} --{edge_data['relation']}--> {node_id}"
                hop_facts.append(fact)
                if source_id not in visited_nodes:
                    next_layer.add(source_id)
                    visited_nodes.add(source_id)

        print(f"\nHop {hop_number} facts found:")
        if len(hop_facts) == 0:
            print("  (none)")
        for fact in hop_facts:
            print(f"  {fact}")

        all_facts.extend(hop_facts)
        current_layer = next_layer

        if len(current_layer) == 0:
            break

    return all_facts

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the following facts extracted from a "
               "knowledge graph via multi-hop traversal. Each fact is a relationship "
               "between two entities. Chain facts together across hops if needed to "
               "reach the full answer. If the facts do not contain enough information, "
               "say you don't know."),
    ("human",
     "Facts (collected across multiple hops):\n{facts}\n\nQuestion: {question}")
])

def answer_with_multi_hop_graphrag(question: str, max_hops: int = 3) -> str:
    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)

    matched_nodes = find_matching_nodes(question, knowledge_graph)
    print(f"\nMatched starting nodes: {matched_nodes}")

    if len(matched_nodes) == 0:
        return "I could not find any matching entities in the knowledge graph for this question."

    all_facts = multi_hop_traversal(
        matched_nodes, knowledge_graph, max_hops=max_hops)

    if len(all_facts) == 0:
        return "Found matching entities, but no connected facts in the graph."

    facts_text = "\n".join(all_facts)
    generation_chain = generate_prompt | llm
    generation_result = generation_chain.invoke({
        "facts": facts_text,
        "question": question
    })
    return generation_result.content

question = ("If an employee raises a complaint with their reporting manager and it is not "
            "resolved, who does it escalate to next, and who is the final authority after that?")

answer = answer_with_multi_hop_graphrag(question, max_hops=3)
print("\nFINAL ANSWER:")
print(answer)