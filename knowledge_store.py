"""
Thin wrapper around a persistent Chroma collection, used as the
customer-support agent's knowledge base.

Two kinds of entries live in the same collection:
  - seeded FAQ docs, loaded once from data/seed_docs.json
  - learned entries, added at runtime after a human resolves an escalated
    query -- this is the "loop learns without retraining" mechanic.
"""

import json
import os
import chromadb
from llm_client import embed_text

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_store")
SEED_PATH = os.path.join(os.path.dirname(__file__), "data", "seed_docs.json")
COLLECTION_NAME = "support_knowledge"


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def seed_if_empty():
    """Load the FAQ seed docs into the collection, once."""
    collection = get_collection()
    if collection.count() > 0:
        return collection

    with open(SEED_PATH) as f:
        docs = json.load(f)

    for doc in docs:
        embedding = embed_text(doc["question"])
        collection.add(
            ids=[doc["id"]],
            embeddings=[embedding],
            documents=[doc["answer"]],
            metadatas=[{"question": doc["question"], "source": "seed"}],
        )
    print(f"[knowledge_store] Seeded {len(docs)} FAQ entries.")
    return collection


def query(question: str, n_results: int = 1):
    """Return the closest matching entries with a similarity confidence
    score in [0, 1], derived from Chroma's L2 distance (lower is closer)."""
    collection = get_collection()
    embedding = embed_text(question)
    results = collection.query(query_embeddings=[embedding], n_results=n_results)

    if not results["ids"] or not results["ids"][0]:
        return []

    matches = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        # crude but workable: map L2 distance to a 0-1 confidence score
        confidence = max(0.0, 1.0 - distance)
        matches.append({
            "id": results["ids"][0][i],
            "answer": results["documents"][0][i],
            "question": results["metadatas"][0][i].get("question", ""),
            "confidence": confidence,
        })
    return matches


def add_learned_entry(question: str, answer: str, entry_id: str):
    """Embed and store a human-validated answer so future semantically
    similar queries can be answered directly by the AI."""
    collection = get_collection()
    embedding = embed_text(question)
    collection.add(
        ids=[entry_id],
        embeddings=[embedding],
        documents=[answer],
        metadatas=[{"question": question, "source": "learned"}],
    )
    print(f"[knowledge_store] Learned new entry: '{question[:60]}...'")
