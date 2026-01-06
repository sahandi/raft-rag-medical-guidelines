from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import streamlit as st
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
MODEL_NAME = "qwen2.5-0.5b-instruct"  # <-- change this

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

def rrf_fusion(sem_ids: List[str], bm25_ids: List[str], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for rank, doc_id in enumerate(sem_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(bm25_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores

@st.cache_resource
def load_resources():
    chunks = load_chunks()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("guidelines")

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    lm_client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")

    by_id = {c.id: c for c in chunks}
    return chunks, by_id, collection, embedder, bm25, lm_client

def hybrid_retrieve(query: str, chunks: List[Chunk], by_id: Dict[str, Chunk], collection, embedder, bm25) -> List[Chunk]:
    q_emb = embedder.encode([query]).tolist()
    sem = collection.query(query_embeddings=q_emb, n_results=TOP_K_SEM)
    sem_ids = sem["ids"][0]

    scores = bm25.get_scores(tokenize(query))
    bm25_ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K_BM25]
    bm25_ids = [chunks[i].id for i in bm25_ranked]

    fused = rrf_fusion(sem_ids, bm25_ids)
    ranked_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:FINAL_CONTEXT_K]
    return [by_id[i] for i in ranked_ids if i in by_id]

def build_prompt(question: str, contexts: List[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        blocks.append(f"[S{i}] pdf={c.pdf_name} chunk_id={c.id}\n{c.text}")
    sources = "\n\n---\n\n".join(blocks)

    return f"""You are a medical document QA assistant.

RULES:
- Use ONLY the SOURCES below.
- If the answer is not clearly supported by the sources, say: "I don't know based on the provided documents."
- Cite sources like [S1], [S2] next to supported claims.
- Keep the answer concise and factual.

QUESTION:
{question}

SOURCES:
{sources}
"""

def main():
    st.set_page_config(page_title="RAFT Medical RAG", layout="wide")
    st.title("🧬 RAFT Medical RAG (Hybrid Retrieval + LM Studio)")

    chunks, by_id, collection, embedder, bm25, lm_client = load_resources()

    st.caption(f"Chunks loaded: {len(chunks)} | Chroma: {CHROMA_DIR}")

    question = st.text_input("Ask a question about your medical guidelines:")

    col1, col2 = st.columns([1, 1])
    with col1:
        ask = st.button("Ask")

    if ask and question.strip():
        ctxs = hybrid_retrieve(question, chunks, by_id, collection, embedder, bm25)
        prompt = build_prompt(question, ctxs)

        resp = lm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Sources used"):
            for i, c in enumerate(ctxs, start=1):
                st.markdown(f"**[S{i}]** pdf=`{c.pdf_name}` chunk_id=`{c.id}`")
                st.write(c.text)

if __name__ == "__main__":
    main()
