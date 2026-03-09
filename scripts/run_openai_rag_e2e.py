from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "raw" / "questions.txt"
OUT_DIR = PROJECT_ROOT / "data" / "eval" / "out"

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TOP_K_SEM = 4
TOP_K_BM25 = 4
FINAL_CONTEXT_K = 4

def tokenize(text: str) -> List[str]:
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

def load_questions() -> List[dict]:
    questions = []
    for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "." in line:
            left, right = line.split(".", 1)
            left = left.strip()
            if left.isdigit():
                questions.append({"qid": int(left), "question": right.strip()})
                continue
        questions.append({"qid": len(questions) + 1, "question": line})
    return questions

def build_bm25(chunks: List[Chunk]) -> BM25Okapi:
    tokenized = [tokenize(c.text) for c in chunks]
    return BM25Okapi(tokenized)

def open_chroma_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection("guidelines")

def rrf_fusion(sem_ids: List[str], bm25_ids: List[str], k: int = 60) -> Dict[str, float]:
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
    q_emb = embedder.encode([query]).tolist()
    sem = collection.query(query_embeddings=q_emb, n_results=k_sem)
    sem_ids = sem["ids"][0]

    q_tok = tokenize(query)
    bm25_scores = bm25.get_scores(q_tok)
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k_bm25]
    bm25_ids = [chunks[i].id for i in bm25_ranked]

    fused = rrf_fusion(sem_ids, bm25_ids)
    ranked_ids = sorted(fused.keys(), key=lambda doc_id: fused[doc_id], reverse=True)[:final_k]

    by_id = {c.id: c for c in chunks}
    return [by_id[i] for i in ranked_ids if i in by_id]

def build_prompt(question: str, contexts: List[Chunk]) -> str:
    source_blocks = []
    for i, c in enumerate(contexts, start=1):
        source_blocks.append(f"[S{i}] pdf={c.pdf_name} chunk_id={c.id}\n{c.text}")
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

def ask_openai(client: OpenAI, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=160,
    )
    return resp.choices[0].message.content

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("❌ OPENAI_API_KEY is not set.")
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_PATH}")
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Missing Chroma DB dir: {CHROMA_DIR}")
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing questions file: {QUESTIONS_PATH}")

    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    print("Loading questions...")
    questions = load_questions()
    print(f"Loaded {len(questions)} questions.")

    print("Loading Chroma collection...")
    collection = open_chroma_collection()

    print("Loading embedder...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Building BM25 (in-memory)...")
    bm25 = build_bm25(chunks)

    client = OpenAI()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"openai_rag_e2e_{MODEL_NAME}.jsonl"

    with out_path.open("w", encoding="utf-8") as f_out:
        for row in questions:
            qid = row["qid"]
            q = row["question"]

            contexts = hybrid_retrieve(q, chunks, bm25, collection, embedder)
            prompt = build_prompt(q, contexts)
            answer = ask_openai(client, prompt)

            f_out.write(json.dumps({
                "qid": qid,
                "model": MODEL_NAME,
                "question": q,
                "answer": answer,
                "sources_used": [
                    {"label": f"S{i}", "chunk_id": c.id, "pdf_name": c.pdf_name}
                    for i, c in enumerate(contexts, start=1)
                ],
            }, ensure_ascii=False) + "\n")
            f_out.flush()
            print(f"✅ qid={qid} done")

    print(f"\n✅ Wrote OpenAI RAG E2E outputs to: {out_path}")

if __name__ == "__main__":
    main()
