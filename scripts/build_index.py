from pathlib import Path
import json
import re

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
BM25_TEXTS_PATH = PROJECT_ROOT / "data" / "bm25_texts.json"
COLLECTION_NAME = "guidelines"


def tokenize(text: str):
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def load_chunks():
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_indexes():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"pdf_name": c["pdf_name"]} for c in chunks]

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted old Chroma collection: {COLLECTION_NAME}")

    collection = client.create_collection(COLLECTION_NAME)

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        metadatas=metadatas,
        documents=texts,
        embeddings=embeddings,
    )
    print("Chroma index built cleanly.")

    tokenized_texts = [tokenize(t) for t in texts]
    with BM25_TEXTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "texts": texts,
                "tokenized_texts": tokenized_texts,
            },
            f,
            ensure_ascii=False,
        )
    print(f"BM25 base texts saved to {BM25_TEXTS_PATH}")


if __name__ == "__main__":
    build_indexes()
