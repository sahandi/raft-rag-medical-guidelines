from pathlib import Path
import json
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MD_DIR = PROJECT_ROOT / "data" / "markdown"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"

MAX_CHARS = 800
OVERLAP = 100


def make_stable_chunk_id(pdf_name: str, chunk_text: str, chunk_num: int) -> str:
    raw = f"{pdf_name}\n{chunk_text}".encode("utf-8")
    short_hash = hashlib.sha1(raw).hexdigest()[:20]
    return f"{pdf_name}_chunk_{chunk_num:04d}_{short_hash}"


def split_long_paragraph(paragraph: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP):
    if len(paragraph) <= max_chars:
        yield paragraph
        return

    step = max_chars - overlap
    start = 0

    while start < len(paragraph):
        piece = paragraph[start:start + max_chars].strip()
        if piece:
            yield piece
        if start + max_chars >= len(paragraph):
            break
        start += step


def split_markdown_to_chunks(text: str, max_chars: int = MAX_CHARS):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current = []
    current_len = 0

    for p in paragraphs:
        pieces = list(split_long_paragraph(p, max_chars=max_chars, overlap=OVERLAP))

        for piece in pieces:
            extra_len = len(piece) + (2 if current else 0)

            if current_len + extra_len <= max_chars:
                current.append(piece)
                current_len += extra_len
            else:
                if current:
                    yield "\n\n".join(current)
                current = [piece]
                current_len = len(piece)

    if current:
        yield "\n\n".join(current)


def main():
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CHUNKS_PATH.open("w", encoding="utf-8") as f_out:
        for md_path in sorted(MD_DIR.glob("*.md")):
            pdf_name = md_path.stem.strip()
            text = md_path.read_text(encoding="utf-8")

            for chunk_num, chunk in enumerate(split_markdown_to_chunks(text, max_chars=MAX_CHARS)):
                record = {
                    "id": make_stable_chunk_id(pdf_name, chunk, chunk_num),
                    "pdf_name": pdf_name,
                    "text": chunk,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Wrote chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
