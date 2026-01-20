# RAFT + RAG on Medical Guidelines (Local, macOS)

End-to-end **document RAG** over medical guideline PDFs (OCR → chunks → hybrid retrieval → cited answers), plus a **controlled evaluation harness** that compares systems using the **same frozen evidence** (frozen chunk IDs / contexts). RAFT fine-tuning is included as an experiment.

## Demo (Streamlit)

![Demo](docs/demo.gif?raw=1)

Run:
```bash
uv run streamlit run app.py
```


What you’ll see:

* Ask a medical guideline question
* The system retrieves evidence from indexed PDFs
* The answer is generated from retrieved chunks
* Sources appear as `pdf=` + `chunk_id=` citations

## Results (10-question eval)

| System                               | Correct | Grounded | Notes                                 |
| ------------------------------------ | ------: | -------: | ------------------------------------- |
| Base + RAG (local, hybrid retrieval) |    9/10 |     9/10 | Strong baseline                       |
| RAFT + RAG (fine-tuned experiment)   |    2/10 |     3/10 | Underperformed in initial small run   |
| SOTA (OpenAI) on frozen contexts     |   10/10 |    10/10 | Controlled comparison (same evidence) |

Rubric:

* **Correct (0/1):** answers the question correctly
* **Grounded (0/1):** answer is supported by the provided evidence and citations match

## Docs

* Dataset notes: `data/README.md`
* RAFT fine-tuning notes: `notebooks/README.md`

## Key features

* PDF ingestion with OCR when needed (`docling[ocr]`)
* Chunked corpus (`data/chunks/chunks.jsonl`)
* Hybrid retrieval:

  * ChromaDB (semantic)
  * BM25 (keyword)
  * RRF fusion for final top-k
* Local inference via **LM Studio** (OpenAI-compatible server)
* Evaluation harness:

  * Freeze chunk IDs per question
  * Build “frozen contexts”
  * Run SOTA against the exact same evidence

<details>
  <summary><b>Quickstart</b></summary>

### 1) Clone

```bash
git clone https://github.com/sahandi/raft-rag-medical-guidelines.git
cd raft-rag-medical-guidelines
```

### 2) Setup

```bash
./scripts/initproject.sh
```

### 3) Add PDFs

Put your guideline PDFs into:

```txt
data/raw/
```

See:

* `data/README.md`
* `data/raw/manifest.csv`

### 4) Build the pipeline

```bash
uv run python scripts/1_ingest_pdfs.py
uv run python scripts/2_make_chunks.py
uv run python scripts/4_build_index.py
```

### 5) Run the demo

```bash
uv run streamlit run app.py
```

</details>

<details>
  <summary><b>Pipeline scripts</b></summary>

* `scripts/1_ingest_pdfs.py` — PDF → Markdown (Docling OCR when needed)
* `scripts/2_make_chunks.py` — Markdown → `data/chunks/chunks.jsonl`
* `scripts/4_build_index.py` — build ChromaDB index + export BM25 base texts
* `scripts/5_rag_cli.py` — CLI demo (LM Studio)
* `scripts/9_build_frozen_contexts.py` — build frozen contexts
* `scripts/10_run_openai_sota.py` — run OpenAI model on frozen contexts

</details>

<details>
  <summary><b>Run with LM Studio (local model)</b></summary>

### 1) Start LM Studio server

In LM Studio:

1. Load a model (example: `Qwen2.5-0.5B-Instruct`)
2. Enable **OpenAI-compatible server**
3. Typical base URL:

   * `http://127.0.0.1:1234/v1`

### 2) Run CLI demo

```bash
export LMSTUDIO_MODEL="qwen2.5-0.5b-instruct"
uv run python scripts/5_rag_cli.py
```

### 3) Run Streamlit UI

```bash
uv run streamlit run app.py
```

</details>

<details>
  <summary><b>Evaluation (frozen contexts + SOTA)</b></summary>

This evaluation is designed to be defensible:

* Freeze which chunks are used per question (same evidence for all models)
* Run multiple models against the exact same evidence
* Score each answer: **Correct (0/1)** and **Grounded (0/1)**

### Inputs (committed)

* Questions: `data/eval/raw/questions.txt`
* Frozen chunk IDs: `data/eval/frozen_sources.jsonl`

### Build frozen contexts (committed output)

```bash
uv run python scripts/9_build_frozen_contexts.py
```

Creates:

* `data/eval/frozen_contexts.jsonl`

### Run OpenAI on frozen contexts (SOTA baseline)

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="gpt-4o-mini"
uv run python scripts/10_run_openai_sota.py
```

Output (ignored by git):

* `data/eval/out/openai_<MODEL>.jsonl`

</details>

<details>
  <summary><b>Troubleshooting</b></summary>

* If answers vary across runs, reduce randomness by lowering generation temperature in `app.py`.
* To wipe and rebuild the vector index, delete `chroma_db/` then re-run:

  ```bash
  uv run python scripts/4_build_index.py
  ```
* If PDFs aren’t found, confirm they are in `data/raw/` (they are ignored by git on purpose).
* If LM Studio connection fails, confirm the server is running at `http://127.0.0.1:1234/v1`.

</details>

