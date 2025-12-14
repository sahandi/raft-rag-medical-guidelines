# RAFT + RAG on Medical Guidelines (Local, Mac)

Build a small Retrieval-Augmented Generation (RAG) system over medical guideline PDFs, then create a RAFT-style dataset (oracle + distractors) to fine-tune a small model to ignore noisy context.

## What this project demonstrates
- PDF ingestion (including OCR when needed) → Markdown
- Markdown → chunked JSONL dataset for retrieval
- RAFT dataset generation (oracle passage + distractor passages)
- Cleaning RAFT outputs when the generator returns JSON inside code fences

---

## Project layout
- `scripts/` : all pipeline scripts
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

---

## One-time setup (do once)

### 1) Create folders
```bash
mkdir -p /Volumes/AD/Rima/RAFT_Project
cd /Volumes/AD/Rima/RAFT_Project
mkdir -p data/raw data/markdown data/chunks data/raft chroma_db hf_cache scripts




