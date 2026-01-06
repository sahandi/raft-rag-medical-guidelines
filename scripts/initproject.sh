#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Repo root: $REPO_ROOT"

# 1) Create folders
mkdir -p \
  data/raw data/markdown data/chunks data/raft \
  data/eval/raw data/eval/out \
  chroma_db hf_cache notebooks

# 2) Check Homebrew (macOS)
if ! command -v brew >/dev/null 2>&1; then
  echo "ERROR: Homebrew not found."
  echo "Install from https://brew.sh then re-run."
  exit 1
fi

# 3) Check uv
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found."
  echo "Install: brew install uv"
  exit 1
fi

# 4) OCR deps (Docling OCR)
if ! command -v tesseract >/dev/null 2>&1; then
  echo "==> Installing OCR deps (tesseract, leptonica)"
  brew install tesseract leptonica
fi

# 5) Create venv (Python 3.11)
uv venv --python 3.11

# 6) Install Python deps
if [ -f requirements.txt ]; then
  uv pip install -r requirements.txt
else
  echo "ERROR: requirements.txt missing."
  echo "Create it (clean): docling[ocr], chromadb, sentence-transformers, rank_bm25, openai, streamlit, pandas"
  exit 1
fi

echo ""
echo "✅ Setup complete."
echo "Next steps:"
echo "  1) Put PDFs into: data/raw/"
echo "  2) Run: uv run python scripts/1_ingest_pdfs.py"
echo "  3) Run: uv run python scripts/2_make_chunks.py"
echo "  4) Run: uv run python scripts/4_build_index.py"
echo "  5) Run: uv run python scripts/5_rag_cli.py"
