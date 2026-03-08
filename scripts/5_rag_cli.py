from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import chromadb
from openai import OpenAI
from openai import APIConnectionError, APIError
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
MODEL_NAME = os.getenv("LMSTUDIO_MODEL", "qwen2.5-0.5b-instruct")

# LM Studio model ids usually look like "qwen2.5-0.5b-raft" (no .gguf).
# If someone passes a .gguf filename, strip the extension for robustness.
if MODEL_NAME.lower().endswith(".gguf"):
    MODEL_NAME = MODEL_NAME[:-5]

TOP_K_SEM = 4
TOP_K_BM25 = 4
FINAL_CONTEXT_K = 4  # how many chunks we actually feed to the model

# -----------------------------
# Helpers
# -----------------------------
def tokenize(text: str) -> List[str]:
    # simple tokenizer good enough for BM25
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)

@dataclass
class Chunk:
    id: str
    pdf_name: str
    text: str

def load_chunks() -> List[Chunk]:
    chunks: List[Chunk] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(Chunk(id=obj["id"], pdf_name=obj["pdf_name"], text=obj["text"]))
    return chunks

def build_bm25(chunks: List[Chunk]) -> BM25Okapi:
    tokenized = [tokenize(c.text) for c in chunks]
    return BM25Okapi(tokenized)

def open_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection("guidelines")

def rrf_fusion(sem_ids: List[str], bm25_ids: List[str], k: int = 60) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion: score(doc) = sum(1 / (k + rank))
    """
    scores: Dict[str, float] = {}
    for rank, doc_id in enumerate(sem_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(bm25_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores

def hybrid_retrieve(
    query: str,
    chunks: List[Chunk],
    bm25: BM25Okapi,
    collection,
    embedder: SentenceTransformer,
    k_sem: int = TOP_K_SEM,
    k_bm25: int = TOP_K_BM25,
    final_k: int = FINAL_CONTEXT_K,
) -> List[Chunk]:
    # --- semantic (Chroma) ---
    q_emb = embedder.encode([query]).tolist()
    sem = collection.query(query_embeddings=q_emb, n_results=k_sem)
    sem_ids = sem["ids"][0]  # list of ids

    # --- keyword (BM25) ---
    q_tok = tokenize(query)
    bm25_scores = bm25.get_scores(q_tok)
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_bm25]
    bm25_ids = [chunks[i].id for i in bm25_ranked]

    # --- fuse results ---
    fused = rrf_fusion(sem_ids, bm25_ids)
    ranked_ids = sorted(fused.keys(), key=lambda doc_id: fused[doc_id], reverse=True)[:final_k]

    # map id -> chunk
    by_id = {c.id: c for c in chunks}
    return [by_id[i] for i in ranked_ids if i in by_id]


def build_prompt(question: str, contexts: List[Chunk]) -> str:
    # Label sources so the model can cite them.
    source_blocks = []
    for i, c in enumerate(contexts, start=1):
        source_blocks.append(
            f"[S{i}] pdf={c.pdf_name} chunk_id={c.id}\n{c.text}"


        )
    sources_text = "\n\n---\n\n".join(source_blocks)
    return f"""You are a medical document QA assistant.

RULES:
- Use ONLY the SOURCES below.
- If the answer is not clearly supported by the sources, say: "I don't know based on the provided documents."
- In your answer, cite sources like [S1], [S2] next to the claims they support.
- Keep the answer concise and factual.

QUESTION:
{question}

SOURCES:
{sources_text}
"""

def ask_lmstudio(prompt: str) -> str:
    try:
        lm_client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")
        resp = lm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=160,
        )
        return resp.choices[0].message.content
    except APIConnectionError:
        return "ERROR: Could not connect to LM Studio. Make sure the LM Studio local server is running."
    except APIError as e:
        return f"ERROR: LM Studio API error: {e}"
    except Exception as e:
        return f"ERROR: Unexpected error while calling LM Studio: {e}"

def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_PATH}")
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Missing Chroma DB dir: {CHROMA_DIR}")

    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    print("Loading Chroma collection...")
    collection = open_chroma_collection()

    print("Loading embedder...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Building BM25 (in-memory)...")
    bm25 = build_bm25(chunks)

    print(f"\n✅ Using model: {MODEL_NAME}")
    print("\n✅ Ready. Type questions. Type 'exit' to quit.")

    while True:
        q = input("\nQuestion> ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        contexts = hybrid_retrieve(q, chunks, bm25, collection, embedder)
        prompt = build_prompt(q, contexts)
        answer = ask_lmstudio(prompt)

        print("\n=== ANSWER ===")
        print(answer)

        print("\n=== SOURCES USED ===")
        for i, c in enumerate(contexts, start=1):
            snippet = c.text.replace("\n", " ")
            print(f"[S{i}] pdf={c.pdf_name} chunk_id={c.id} :: {snippet[:220]}...")
        print("=" * 80)

if __name__ == "__main__":
    main()