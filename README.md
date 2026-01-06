````md
# RAFT + RAG on Medical Guidelines (Local, macOS)

This repo demonstrates an end-to-end **document RAG system** over real medical guideline PDFs, plus an **evaluation harness** for comparing a local base model vs. SOTA (OpenAI) under the **same frozen evidence**.

It also includes an **experimental RAFT (Retrieval-Augmented Fine-Tuning)** LoRA notebook. In the initial small run, RAFT did **not** outperform the Base+RAG pipeline, so RAFT is documented as an experiment + future work.

> ✅ This repo contains **code + small reproducible inputs only**.  
> PDFs, extracted text, indexes, and model artifacts are excluded via `.gitignore`.

---

## What this project demonstrates

### RAG pipeline
- PDF ingestion (OCR when needed) → Markdown (`docling`)
- Markdown → chunked JSONL dataset for retrieval
- Hybrid retrieval:
  - **ChromaDB** (semantic / embeddings)
  - **BM25** (keyword)
- Local RAG demo:
  - CLI (terminal)
  - Streamlit UI (browser)

### Evaluation harness (mentor-friendly)
- “Frozen contexts” evaluation:
  - Freeze which chunks are used per question
  - Run multiple models against the **same evidence**
  - Score each answer: **Correct (0/1)** and **Grounded (0/1)**

### RAFT fine-tuning (experiment)
- Create RAFT training examples (oracle + distractors)
- Fine-tune `Qwen2.5-0.5B-Instruct` with LoRA (Unsloth + TRL)
- Documented in `notebooks/` with reproducible steps

---

## Repo layout (clean + professional)

- `scripts/` : runnable pipeline scripts
  - `scripts/1_ingest_pdfs.py` — PDF → Markdown
  - `scripts/2_make_chunks.py` — Markdown → `chunks.jsonl`
  - `scripts/4_build_index.py` — build Chroma + BM25 texts
  - `scripts/5_rag_cli.py` — CLI RAG demo (LM Studio)
  - `scripts/9_build_frozen_contexts.py` — build frozen contexts file
  - `scripts/10_run_openai_sota.py` — run OpenAI model on frozen contexts
- `app.py` : Streamlit UI
- `data/` : **documentation + small reproducible eval inputs**
  - `data/README.md` : dataset details (what PDFs, where to put them, chunk size)
  - `data/raw/manifest.csv` : list of PDFs + notes (committed)
  - `data/eval/raw/questions.txt` : evaluation questions (committed)
  - `data/eval/frozen_sources.jsonl` : frozen chunk IDs per question (committed)
  - `data/eval/frozen_contexts.jsonl` : frozen text contexts per question (committed)
  - `data/eval/out/` : model outputs (ignored)
- `notebooks/` : fine-tuning notebook + documentation
  - `notebooks/raft_finetune_unsloth.ipynb`
  - `notebooks/README.md`
- `requirements.txt` : top-level dependencies (portable)
- `requirements.lock.txt` : frozen environment snapshot (optional but useful)
- `uv.lock` : uv resolver lock file
- `scripts/initproject.sh` : one-command setup script (creates venv + installs deps)

---

## Requirements

- macOS (tested on Apple Silicon)
- Homebrew
- Python 3.11
- `uv` (Python environment manager)
- OCR tools for Docling: `tesseract` + `leptonica`
- LM Studio (for local OpenAI-compatible inference)

---

## Quickstart (recommended)

### 1) Clone
```bash
git clone <YOUR_REPO_URL>
cd raft-rag-medical-guidelines
````

### 2) Setup (creates venv + installs deps + creates folders)

```bash
./scripts/initproject.sh
```

### 3) Add PDFs (dataset)

Put the guideline PDFs into:

```txt
data/raw/
```

See:

* `data/README.md` (how dataset works + pipeline outputs)
* `data/raw/manifest.csv` (exact PDF list + notes)

### 4) Build the pipeline

```bash
uv run python scripts/1_ingest_pdfs.py
uv run python scripts/2_make_chunks.py
uv run python scripts/4_build_index.py
```

---

## Run the RAG demo (LM Studio)

### 1) Start LM Studio local server

In LM Studio:

1. Load a model (example: `Qwen2.5-0.5B-Instruct`)
2. Enable **OpenAI-compatible server**
3. Typical base URL:

   * `http://127.0.0.1:1234/v1`

### 2) Run CLI demo

```bash
export LMSTUDIO_MODEL="qwen2.5-0.5b-instruct"   # change to match your LM Studio model id
uv run python scripts/5_rag_cli.py
```

### 3) Run Streamlit UI

```bash
uv run streamlit run app.py
```

---

## Evaluation (frozen contexts + SOTA comparison)

This repo supports a controlled evaluation:

* Freeze which chunks are used per question (same evidence for all models)
* Run OpenAI (SOTA baseline) against the same frozen evidence
* Score each answer:

  * **Correct (0/1)**: answers the question correctly
  * **Grounded (0/1)**: answer is supported by the provided evidence (citations match)

### Inputs (committed)

* Questions:

  * `data/eval/raw/questions.txt`
* Frozen chunk IDs:

  * `data/eval/frozen_sources.jsonl`

### Build frozen contexts file (committed output)

```bash
uv run python scripts/9_build_frozen_contexts.py
```

Creates:

* `data/eval/frozen_contexts.jsonl`

### Run OpenAI on frozen contexts (SOTA baseline)

Set your key:

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="gpt-4o-mini"   # or another model you have access to
```

Run:

```bash
uv run python scripts/10_run_openai_sota.py
```

Outputs (ignored by git):

* `data/eval/out/openai_<MODEL>.jsonl`

---

## RAFT fine-tuning (experiment)

RAFT was attempted and documented (notebooks + scripts), but **the initial run used a small RAFT dataset** and **did not outperform Base+RAG** on the initial 10-question evaluation.

Where to look:

* Notebook + details: `notebooks/`

  * `notebooks/raft_finetune_unsloth.ipynb`
  * `notebooks/README.md`

Why it underperformed (likely):

* small training set (few examples)
* limited training steps
* formatting / cleanup consistency issues in generated RAFT data

Future work:

* increase RAFT dataset size
* hold-out eval set
* longer training runs + better data formatting checks

**Final recommended demo pipeline for this repo:**

> Base model + Hybrid Retrieval + Frozen-context Evaluation + SOTA comparison

---

## Notes on reproducibility / clean repo

* The repo uses **relative paths** (portable across machines).
* Large / generated artifacts are ignored:

  * PDFs, extracted markdown, chunks, vector DB
  * model weights and GGUFs
  * evaluation outputs (`data/eval/out/`)
* Dataset documentation is committed in:

  * `data/README.md`
  * `data/raw/manifest.csv`

---

## Common commands

Rebuild everything from scratch (after adding PDFs):

```bash
uv run python scripts/1_ingest_pdfs.py
uv run python scripts/2_make_chunks.py
uv run python scripts/4_build_index.py
```

Run CLI:

```bash
uv run python scripts/5_rag_cli.py
```

Run UI:

```bash
uv run streamlit run app.py
```

Run evaluation:

```bash
uv run python scripts/9_build_frozen_contexts.py
uv run python scripts/10_run_openai_sota.py
```

