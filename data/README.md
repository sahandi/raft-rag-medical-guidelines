# Dataset (Medical Guideline PDFs)

This project’s dataset is a small set of public medical guideline PDFs.  
Place the PDFs in `data/raw/`. The PDFs are NOT committed to GitHub (see `.gitignore`).

## Folder locations
- Input PDFs: `data/raw/`
- Extracted text (generated): `data/markdown/` (not committed)
- Chunk dataset (generated): `data/chunks/chunks.jsonl` (not committed)

## Pipeline: how PDFs become searchable chunks
1) **PDF → Markdown**
- Script: `scripts/1_ingest_pdfs.py`
- What it does: converts each PDF to a Markdown `.md` file using Docling (OCR if needed)
- Output: `data/markdown/<pdf_name>.md`

2) **Markdown → Chunks (JSONL)**
- Script: `scripts/2_make_chunks.py`
- What it does: splits each Markdown file into smaller text chunks for retrieval
- Output: `data/chunks/chunks.jsonl` (one JSON object per line)

3) **Indexing for retrieval**
- Script: `scripts/4_build_index.py`
- What it does:
  - builds a semantic index in ChromaDB (embeddings)
  - saves BM25 base texts for keyword search
- Outputs: `chroma_db/` and `data/bm25_texts.json` (not committed)

## Chunking settings
- Chunk size: `max_chars = 800` (see `scripts/2_make_chunks.py`)
- Chunking method: paragraph-based splitting until max size is reached

## PDF list
See `data/raw/manifest.csv` for the full list of PDFs used and their purpose.
