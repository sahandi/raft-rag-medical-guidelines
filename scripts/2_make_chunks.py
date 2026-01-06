from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MD_DIR = PROJECT_ROOT / "data" / "markdown"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"

def split_markdown_to_chunks(text: str, max_chars: int = 800):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current = []
    current_len = 0

    for p in paragraphs:
        if current_len + len(p) + 2 <= max_chars:
            current.append(p)
            current_len += len(p) + 2
        else:
            if current:
                yield "\n\n".join(current)
            current = [p]
            current_len = len(p)

    if current:
        yield "\n\n".join(current)

def main():
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f_out:
        idx = 0
        for md_path in sorted(MD_DIR.glob("*.md")):
            text = md_path.read_text(encoding="utf-8")
            for chunk in split_markdown_to_chunks(text, max_chars=800):
                record = {
                    "id": f"{md_path.stem}_chunk_{idx}",
                    "pdf_name": md_path.stem,
                    "text": chunk,
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                idx += 1
    print(f"✅ Wrote chunks to {CHUNKS_PATH}")

if __name__ == "__main__":
    main()
