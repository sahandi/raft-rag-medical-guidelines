import json
import random
from pathlib import Path
from typing import List
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
RAFT_PATH = PROJECT_ROOT / "data" / "raft" / "raft.jsonl"

client = OpenAI()  # uses OPENAI_API_KEY

def load_chunks() -> List[dict]:
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def sample_examples(chunks: List[dict], num_examples: int = 50):
    examples = []
    for _ in range(num_examples):
        oracle = random.choice(chunks)
        others = [c for c in chunks if c["pdf_name"] != oracle["pdf_name"]]
        if len(others) < 3:
            others = [c for c in chunks if c["id"] != oracle["id"]]
        distractors = random.sample(others, k=3)
        examples.append((oracle, distractors))
    return examples

def make_prompt(oracle_text: str, distractors: list[str]) -> str:
    return f"""
You are helping create a medical RAFT dataset.

You will see one ORACLE passage (contains the true answer) and 3 DISTRACTOR passages (related but irrelevant).

Task:
1. Write ONE specific medical question that can ONLY be answered from the ORACLE passage.
2. Write a detailed chain-of-thought explaining how you use the ORACLE to answer.
3. Write a final short answer.

Return JSON with keys: "question", "cot_answer", "final_answer".

ORACLE:
\"\"\"{oracle_text}\"\"\"

DISTRACTORS:
1)
\"\"\"{distractors[0]}\"\"\"

2)
\"\"\"{distractors[1]}\"\"\"

3)
\"\"\"{distractors[2]}\"\"\"
"""

def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    NUM = 50  # reduce to 30 if needed for cost/time
    examples = sample_examples(chunks, NUM)

    RAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAFT_PATH.open("w", encoding="utf-8") as fout:
        for i, (oracle, distractors) in enumerate(examples):
            prompt = make_prompt(oracle["text"], [d["text"] for d in distractors])
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                content = resp.choices[0].message.content
                try:
                    obj = json.loads(content)
                except Exception:
                    obj = {
                        "question": "PARSING_ERROR",
                        "cot_answer": content,
                        "final_answer": "PARSING_ERROR",
                    }
                # Alpaca-style for Unsloth
                rec = {
                    "instruction": obj.get("question", ""),
                    "input": "Context (oracle):\n"
                             + oracle["text"]
                             + "\n\nDistractors:\n"
                             + "\n\n---\n\n".join(d["text"] for d in distractors),
                    "output": obj.get("cot_answer", "") + "\n\nFinal answer: " + obj.get("final_answer", ""),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"✅ Example {i+1}/{NUM}")
            except Exception as e:
                print(f"⚠️ Skipped example {i+1} error: {e}")

    print(f"✅ RAFT dataset saved to {RAFT_PATH}")

if __name__ == "__main__":
    main()
