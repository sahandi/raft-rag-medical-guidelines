# RAFT + RAG on Medical Guidelines (Local, Mac)

Build a Retrieval-Augmented Generation (RAG) system over real medical guideline PDFs, then create a RAFT-style dataset (oracle + distractors) to fine-tune a small model to ignore noisy context and follow the correct evidence.

This repo contains **code only** (PDFs, extracted text, indexes, and model artifacts are excluded via `.gitignore`).

---

## What this project demonstrates
- PDF ingestion (including OCR when needed) → Markdown (Docling)
- Markdown → chunked JSONL dataset for retrieval
- RAFT dataset generation (oracle passage + distractor passages)
- Cleaning RAFT outputs when the generator returns JSON inside code fences
- Hybrid retrieval indexing:
  - **Chroma** (embeddings / semantic search)
  - **BM25** (keyword search)
- Local RAG demo:
  - CLI RAG (Terminal)
  - Streamlit UI (browser)
- (Day 2) LoRA fine-tuning artifact published to Hugging Face
- (Optional/Day 4) Base vs RAFT comparison in LM Studio using a merged GGUF

---

## Project layout
- `scripts/` : pipeline scripts (committed)
- `app.py` : Streamlit UI (committed)
- `data/raw/` : source PDFs (**not committed**)
- `data/markdown/` : extracted Markdown (**not committed**)
- `data/chunks/` : chunk outputs (**not committed**)
- `data/raft/` : RAFT datasets (**not committed**)
- `chroma_db/` : local vector DB (**not committed**)
- `hf_cache/` : Hugging Face cache (**not committed**)

> Large/derived files are excluded via `.gitignore`.

---

## Requirements
- macOS (tested on Apple Silicon)
- Homebrew
- Python 3.11
- `uv` (Python environment manager)
- OCR tools for Docling: `tesseract` + `leptonica`
- LM Studio (for local OpenAI-compatible inference)

---

## One-time setup (do once)

### 1) Create folders
```bash
mkdir -p /Volumes/AD/Rima/RAFT_Project
cd /Volumes/AD/Rima/RAFT_Project
mkdir -p data/raw data/markdown data/chunks data/raft chroma_db hf_cache scripts




