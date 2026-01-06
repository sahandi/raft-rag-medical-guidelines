import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAFT_IN = PROJECT_ROOT / "data" / "raft" / "raft.jsonl"
RAFT_OUT = PROJECT_ROOT / "data" / "raft" / "raft_clean.jsonl"

CODEBLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def try_parse_obj(text: str):
    if not text:
        return None
    text = text.strip()

    # Case 1: already pure JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass

    # Case 2: JSON inside code block
    m = CODEBLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None

    return None

def main():
    kept = fixed = dropped = 0

    with RAFT_IN.open("r", encoding="utf-8") as fin, RAFT_OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)

            # If instruction is already good, keep it
            if rec.get("instruction") and rec["instruction"] != "PARSING_ERROR":
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                continue

            # Recover JSON from the model output field
            obj = try_parse_obj(rec.get("output", ""))
            if obj and obj.get("question"):
                rec["instruction"] = obj["question"].strip()
                cot = (obj.get("cot_answer") or "").strip()
                final = (obj.get("final_answer") or "").strip()
                rec["output"] = f"{cot}\n\nFinal answer: {final}".strip()

                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fixed += 1
            else:
                dropped += 1

    print(f"✅ Clean file written: {RAFT_OUT}")
    print(f"Kept: {kept} | Fixed: {fixed} | Dropped: {dropped}")

if __name__ == "__main__":
    main()
