from pathlib import Path
import json

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

def load_chunks():
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def build_indexes():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    # Semantic index (Chroma + embeddings)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("guidelines")

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [{"pdf_name": c["pdf_name"]} for c in chunks]

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    collection.upsert(
        ids=ids,
        metadatas=metadatas,
        documents=texts,
        embeddings=embeddings,
    )
    print("Chroma index built.")

    # BM25 index
    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    bm25_path = PROJECT_ROOT / "data" / "bm25_texts.json"
    with bm25_path.open("w", encoding="utf-8") as f:
        json.dump({
            "texts": texts,
        }, f)
    print(f"BM25 base texts saved to {bm25_path}")
    print("You will recreate BM25Okapi in your query script from these texts.")

if __name__ == "__main__":
    build_indexes()
