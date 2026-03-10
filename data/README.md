# Dataset (Medical Guideline PDFs)

This project uses a small set of public medical guideline PDFs as the document source for retrieval, evaluation, and RAFT data generation.

The raw PDFs are placed in `data/raw/`.

These PDFs are **not committed to GitHub** (see `.gitignore`).

---

## Data folders

- **Raw PDFs:** `data/raw/`
- **PDF manifest:** `data/raw/manifest.csv`
- **Extracted Markdown:** `data/markdown/` *(generated, not committed)*
- **Chunk dataset:** `data/chunks/chunks.jsonl` *(generated, not committed)*
- **RAFT training data:** `data/raft/` *(generated, not committed)*
- **Evaluation inputs and outputs:** `data/eval/`

---

## Pipeline: how PDFs become searchable data

### 1) PDF → Markdown
- **Script:** `scripts/ingest_pdfs.py`
- **What it does:** converts each PDF into Markdown using Docling, with OCR when needed
- **Output:** `data/markdown/<pdf_name>.md`

### 2) Markdown → chunks
- **Script:** `scripts/make_chunks.py`
- **What it does:** splits each Markdown file into smaller text chunks for retrieval
- **Output:** `data/chunks/chunks.jsonl`

Each line in `chunks.jsonl` is one JSON object with fields like:

- `id`
- `pdf_name`
- `text`

### 3) Build retrieval index
- **Script:** `scripts/build_index.py`
- **What it does:**
  - builds a semantic index in ChromaDB
  - prepares BM25 text data for keyword retrieval
- **Outputs:**
  - `chroma_db/`
  - `data/bm25_texts.json`

---

## Chunking settings

Chunking is paragraph-based.

Current setting in `scripts/make_chunks.py`:

- **Chunk size:** `max_chars = 800`

This means paragraphs are grouped together until the chunk reaches about 800 characters, then a new chunk starts.

---

## RAFT training data

RAFT-style training data is generated from the chunk dataset.

- **Script:** `scripts/build_raft_dataset.py`
- **Input:** `data/chunks/chunks.jsonl`
- **Output:** `data/raft/raft.jsonl`

The generated RAFT dataset is used later in the Colab fine-tuning notebook.

---

## Evaluation data

The project also stores evaluation inputs and outputs under:

- `data/eval/raw/`
- `data/eval/out/`

This includes:

- question files
- frozen source definitions
- frozen contexts
- standardized JSONL evaluation outputs
- summary tables such as `eval_summary.csv`

---

## PDF list

See:

- `data/raw/manifest.csv`

That file records the PDFs used in this project and their purpose.

---

## Notes

- Raw PDFs are intentionally kept out of GitHub
- Generated Markdown, chunks, RAFT data, and large retrieval artifacts are also ignored by Git where appropriate
- To rebuild the dataset pipeline, start from the raw PDFs in `data/raw/`