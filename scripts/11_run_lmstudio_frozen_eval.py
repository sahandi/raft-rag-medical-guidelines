from __future__ import annotations

import json
import os
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "eval" / "frozen_contexts.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "eval" / "out"

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.getenv("LMSTUDIO_MODEL", "").strip()
TEMP = float(os.getenv("LMSTUDIO_TEMPERATURE", "0.0"))

SYSTEM_RULES = """You are a medical document QA assistant.

RULES:
- Use ONLY the SOURCES provided.
- If the answer is not clearly supported by the sources, say: "I don't know based on the provided documents."
- Cite sources like [S1], [S2] next to the claims they support.
- Keep the answer concise and factual.
"""

def build_prompt(question: str, sources: list[dict]) -> str:
    blocks = []
    for i, s in enumerate(sources, start=1):
        blocks.append(
            f"[S{i}] pdf={s['pdf_name']} chunk_id={s['chunk_id']}\n{s['text']}"
        )
    sources_text = "\n\n---\n\n".join(blocks)
    return f"{SYSTEM_RULES}\nQUESTION:\n{question}\n\nSOURCES:\n{sources_text}\n"

def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing: {IN_PATH}")

    if not MODEL:
        raise SystemExit(
            "❌ LMSTUDIO_MODEL is not set.\n"
            "Example:\n"
            "  export LMSTUDIO_MODEL='qwen2.5-0.5b-instruct'\n"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"lmstudio_{MODEL}.jsonl"

    client = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key="lm-studio")

    with IN_PATH.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            row = json.loads(line)
            qid = row["qid"]
            question = row["question"]
            sources = row["sources"]

            prompt = build_prompt(question, sources)
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMP,
            )
            answer = resp.choices[0].message.content

            f_out.write(
                json.dumps(
                    {
                        "qid": qid,
                        "model": MODEL,
                        "answer": answer,
                        "sources_used": [
                            {"chunk_id": s["chunk_id"], "pdf_name": s["pdf_name"]}
                            for s in sources
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"✅ qid={qid} done")

    print(f"\n✅ Wrote LM Studio outputs to: {out_path}")

if __name__ == "__main__":
    main()
