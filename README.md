# RAFT + RAG on Medical Guidelines (Local)

## Goal
Build a small RAG system over medical guideline PDFs, then create a RAFT-style dataset to fine-tune a small model to ignore distractors and use oracle context.

## Day 1 (done)
- PDF → Markdown extraction
- Markdown → chunking into `data/chunks/chunks.jsonl`

## Project structure
- `scripts/` : ingestion + chunking scripts
- `data/raw/` : PDFs (not committed)
- `data/markdown/` : extracted text (not committed)
- `data/chunks/` : chunks output (not committed)
- `hf_cache/` : HF cache (not committed)

## Setup
```bash
cd /Volumes/AD/Rima/RAFT_Project
uv venv --python 3.11
uv pip install "docling[ocr]"

