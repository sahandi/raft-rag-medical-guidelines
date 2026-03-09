from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "eval" / "raw" / "questions.txt"
FROZEN_SOURCES_PATH = PROJECT_ROOT / "data" / "eval" / "frozen_sources.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "eval" / "frozen_contexts.jsonl"


def load_questions() -> dict[int, str]:
    qmap: dict[int, str] = {}
    for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        # Expect "1. blah blah"
        if "." in line:
            left, right = line.split(".", 1)
            left = left.strip()
            if left.isdigit():
                qid = int(left)
                qmap[qid] = right.strip()
                continue
        # Fallback: ignore bad lines
    if not qmap:
        raise ValueError(f"No questions parsed from {QUESTIONS_PATH}")
    return qmap


def load_frozen_sources() -> dict[int, list[str]]:
    fmap: dict[int, list[str]] = {}
    with FROZEN_SOURCES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fmap[int(obj["qid"])] = list(obj["chunk_ids"])
    return fmap


def load_chunks_by_id() -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            by_id[obj["id"]] = obj
    return by_id


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing: {CHUNKS_PATH}")
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing: {QUESTIONS_PATH}")
    if not FROZEN_SOURCES_PATH.exists():
        raise FileNotFoundError(f"Missing: {FROZEN_SOURCES_PATH}")

    questions = load_questions()
    frozen = load_frozen_sources()
    chunks_by_id = load_chunks_by_id()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    missing = []
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for qid in sorted(frozen.keys()):
            q = questions.get(qid)
            if not q:
                raise ValueError(f"Question qid={qid} missing from {QUESTIONS_PATH}")

            sources = []
            for cid in frozen[qid]:
                c = chunks_by_id.get(cid)
                if not c:
                    missing.append((qid, cid))
                    continue
                sources.append(
                    {"chunk_id": c["id"], "pdf_name": c["pdf_name"], "text": c["text"]}
                )

            obj = {"qid": qid, "question": q, "sources": sources}
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if missing:
        print("❌ Missing chunk_ids (check spaces / exact spelling):")
        for qid, cid in missing:
            print(f"  qid={qid}: {repr(cid)}")
        raise SystemExit(1)

    print(f"✅ Wrote frozen contexts to: {OUT_PATH}")


if __name__ == "__main__":
    main()
