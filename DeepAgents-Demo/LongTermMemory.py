import uuid
from langchain_openrouter import ChatOpenRouter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

emb = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenRouter(model="gpt-4o-mini")

episodic = Chroma(collection_name="episodic", embedding_function=emb, persist_directory="./memory", collection_metadata={"hnsw:space": "cosine"})
semantic = Chroma(collection_name="sematic", embedding_function=emb, persist_directory="./memory", collection_metadata={"hnsw:space": "cosine"})

THRESHOLD = 0.2
CONSOLIDATE_AFTER = 5

def add_episode(user, text):
    episodic.add_documents([Document(
        page_content=text,
        metadata={"user": user, "id": str(uuid.uuid4())},
    )])

def add_facts(user, facts):
    if facts:
        episodic
        semantic.add_documents([
            Document(page_content=f, metadata={"user": user}) for f in facts
        ])

def search(store, user, query, k=4):
    # similarity search with score, filtered by user and threshold
    hits = store.similarity_search_with_relevance_scores(
        query, k=k, filter={"user": user}
    )
    return [doc.page_content for doc, score in hits if score >= THRESHOLD]

def consolidate(user):
    # pull all raw episodes for this user
    got = episodic.get(where={"user": user})
    ids, docs = got["ids"], got["documents"]
    if len(ids) < CONSOLIDATE_AFTER:
        return

    joined = "\n".join(docs)
    resp = llm.invoke(
        f"Extract durable, general facts about the user from these exchanges.One fact per line, no numbering. Skip anything trivial or one-off.\n\n {joined}"
    ).content
    facts = [line.strip("-• ").strip() for line in resp.splitlines() if line.strip()]

    add_facts(user, facts)
    episodic.delete(ids=ids)          # delete raw episodes once distilled
    print(f"[consolidated {len(ids)} episodes -> {len(facts)} facts]")

def answer(user, query):
    # All known facts for this user (direct lookup by id — always available)
    facts = semantic.get(where={"user": user})["documents"]
    # Topically relevant past exchanges (similarity search)
    recent = search(episodic, user, query)

    context = ""
    if facts:
        context += "Known facts about the user:\n" + "\n".join(f"- {f}" for f in facts) + "\n\n"
    if recent:
        context += "Relevant past exchanges:\n" + "\n".join(f"- {r}" for r in recent) + "\n\n"

    reply = llm.invoke(
        f"{context}The user's id is '{user}'. The known facts above are about "
        f"this user — treat 'me', 'I', and 'my' as referring to them.\n\n"
        f"User says: {query}\n\nRespond helpfully using any relevant memory above."
    ).content

    add_episode(user, f"User: {query}\nAssistant: {reply}")
    consolidate(user)
    return reply

def main():
    user = input("User id: ").strip() or "default"
    print("Chat (blank line to quit). Memory persists across runs.\n")
    while True:
        q = input("> ").strip()
        if not q:
            break
        print(answer(user, q), "\n")


if __name__ == "__main__":
    main()