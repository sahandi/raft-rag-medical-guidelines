import json
import random
from pathlib import Path
from typing import List
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
RAFT_PATH = PROJECT_ROOT / "data" / "raft" / "raft.jsonl"

client = OpenAI()  # uses OPENAI_API_KEY

random.seed(42)

def load_chunks() -> List[dict]:
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def is_good_oracle_chunk(chunk: dict) -> bool:
    text = chunk["text"].lower().strip()

    bad_patterns = [
        "references",
        "doi:",
        " et al.",
        "accessed ",
        "isbn",
        "all rights reserved",
        "creative commons",
        "guideline group",
        "peer review",
        "committee",
        "chaired",
        "level of evidence",
        "recommended as of",
    ]

    if len(text) < 120:
        return False

    if any(p in text for p in bad_patterns):
        return False

    # Too table-heavy / formatting-heavy is usually poor for RAFT question generation
    if text.count("|") > 12:
        return False

    # Too many URLs usually means reference/frontmatter content
    if text.count("http") >= 2:
        return False

    return True


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
2. Write one short grounded answer using only the ORACLE passage.

Return JSON with keys: "question", "final_answer".

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

def clean_json_text(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[len("```json"):].strip()
    elif content.startswith("```"):
        content = content[len("```"):].strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    return content


def is_good_generated_example(question: str, final_answer: str) -> bool:
    q = question.lower().strip()
    a = final_answer.lower().strip()

    bad_question_patterns = [
        "what is the focus of the article",
        "who chaired",
        "who published",
        "what organization published",
        "what was the comparison made",
        "what are some complications",
    ]

    bad_answer_patterns = [
        "as detailed in recommendation",
        "see recommendation",
        "the article focuses on",
    ]

    if len(q) < 20 or len(a) < 12:
        return False

    if any(p in q for p in bad_question_patterns):
        return False

    if any(p in a for p in bad_answer_patterns):
        return False

    return True


def main():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    filtered_chunks = [c for c in chunks if is_good_oracle_chunk(c)]
    print(f"Usable oracle chunks after filtering: {len(filtered_chunks)}")

    NUM = 50  # reduce to 30 if needed for cost/time
    examples = sample_examples(filtered_chunks, NUM)

    RAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAFT_PATH.open("w", encoding="utf-8") as fout:
        for i, (oracle, distractors) in enumerate(examples):
            prompt = make_prompt(oracle["text"], [d["text"] for d in distractors])
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                content = resp.choices[0].message.content
                try:
                    obj = json.loads(clean_json_text(content))
                except Exception:
                    print(f"⚠️ Skipped example {i+1} due to JSON parsing error")
                    continue

                question = obj.get("question", "").strip()
                final_answer = obj.get("final_answer", "").strip()

                if not question or not final_answer:
                    print(f"⚠️ Skipped example {i+1} due to empty question/answer")
                    continue

                if not is_good_generated_example(question, final_answer):
                    print(f"⚠️ Skipped example {i+1} due to weak generated QA")
                    continue

                # Alpaca-style for Unsloth
                rec = {
                    "instruction": question,
                    "input": "Context (oracle):\n"
                             + oracle["text"]
                             + "\n\nDistractors:\n"
                             + "\n\n---\n\n".join(d["text"] for d in distractors),
                    "output": final_answer,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"✅ Example {i+1}/{NUM}")
            except Exception as e:
                print(f"⚠️ Skipped example {i+1} error: {e}")

    print(f"✅ RAFT dataset saved to {RAFT_PATH}")

if __name__ == "__main__":
    main()
