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

## Day 2: Fine-tuning + Hybrid Retrieval Index

### 1) Fine-tuning (Google Colab)
Goal: run a small RAFT-style fine-tune so the model learns to rely on the ORACLE context and ignore DISTRACTORS.

What I did:
- Used the cleaned RAFT dataset: `data/raft/raft_clean.jsonl` (generated Day 1; ignored by git).
- Fine-tuned **Qwen2.5-0.5B-Instruct** using **Unsloth + LoRA** in Colab.
- Trained with TRL `SFTTrainer` for a small run (example config used: `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`, `max_steps=300`).
- Saved LoRA adapter artifacts as a folder: `raft_lora_adapter/`
  - Key files: `adapter_model.safetensors`, `adapter_config.json`, tokenizer files.
- Packaged the adapter to download from Colab: `raft_lora_adapter.zip`
- Published LoRA adapter to Hugging Face:
  - `vectormind/qwen2.5-0.5b-raft-lora`

Notes:
- LoRA adapters are small “diff” weights. To use them, you load the base model + apply the adapter.
- I did **not** publish the “merged 16-bit model” (optional artifact). A merged model means the LoRA changes are merged into full model weights (bigger, but standalone).

### 2) Retrieval indexing (Local Mac)
Goal: build retrieval indexes so RAG can fetch relevant chunks fast.

I built a hybrid index:
- **ChromaDB** (semantic / embedding search) persisted at: `chroma_db/` (ignored by git)
- **BM25** (keyword search) base texts saved at: `data/bm25_texts.json` (ignored by git)

Build command:
```bash
cd /Volumes/AD/Rima/RAFT_Project
uv run python scripts/4_build_index.py


