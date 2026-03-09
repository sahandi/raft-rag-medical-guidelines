from __future__ import annotations

import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "raw" / "questions.txt"
OUT_DIR = PROJECT_ROOT / "data" / "eval" / "out"

SOURCE_RE = re.compile(
    r"^\[(S\d+)\]\s+pdf=(.*?)\s+chunk_id=(.*?)\s+::\s+(.*)$"
)

def load_questions() -> list[dict]:
    questions = []
    for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "." in line:
            left, right = line.split(".", 1)
            left = left.strip()
            if left.isdigit():
                questions.append({"qid": int(left), "question": right.strip()})
                continue
        questions.append({"qid": len(questions) + 1, "question": line})
    return questions

def parse_run_file(txt_path: Path, jsonl_path: Path) -> None:
    text = txt_path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("=" * 80) if "=== ANSWER ===" in b]

    questions = load_questions()
    if len(blocks) != len(questions):
        raise ValueError(
            f"Block count ({len(blocks)}) does not match questions count ({len(questions)}) in {txt_path}"
        )

    with jsonl_path.open("w", encoding="utf-8") as f_out:
        for idx, block in enumerate(blocks):
            q = questions[idx]

            answer = ""
            sources_used = []

            if "=== ANSWER ===" in block and "=== SOURCES USED ===" in block:
                answer_part = block.split("=== ANSWER ===", 1)[1].split("=== SOURCES USED ===", 1)[0].strip()
                sources_part = block.split("=== SOURCES USED ===", 1)[1].strip()
                answer = answer_part

                for line in sources_part.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    m = SOURCE_RE.match(line)
                    if m:
                        label, pdf_name, chunk_id, snippet = m.groups()
                        sources_used.append(
                            {
                                "label": label,
                                "chunk_id": chunk_id,
                                "pdf_name": pdf_name,
                                "snippet": snippet,
                            }
                        )

            row = {
                "qid": q["qid"],
                "question": q["question"],
                "answer": answer,
                "sources_used": sources_used,
            }
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {jsonl_path}")

def main():
    base_txt = OUT_DIR / "base_rag_e2e_run.txt"
    raft_txt = OUT_DIR / "raft_rag_e2e_run.txt"

    base_jsonl = OUT_DIR / "base_rag_e2e_run.jsonl"
    raft_jsonl = OUT_DIR / "raft_rag_e2e_run.jsonl"

    if not base_txt.exists():
        raise FileNotFoundError(f"Missing: {base_txt}")
    if not raft_txt.exists():
        raise FileNotFoundError(f"Missing: {raft_txt}")

    parse_run_file(base_txt, base_jsonl)
    parse_run_file(raft_txt, raft_jsonl)

if __name__ == "__main__":
    main()
