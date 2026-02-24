# Fine-tuning (RAFT LoRA) — Notebook Summary

This folder contains the notebook used to fine-tune a small model using **RAFT-style training data** (oracle + distractors).

The goal is to teach the model to:
- focus on the correct retrieved evidence (**oracle** chunk)
- ignore misleading or irrelevant retrieved text (**distractor** chunks)
- produce grounded answers with citations

---

## Notebook

- **File:** `notebooks/raft_finetune_unsloth.ipynb`
- **Environment:** Google Colab (GPU runtime)

---

## Base model

- **Base model (HF):** `unsloth/Qwen2.5-0.5B-Instruct`
- **Fine-tuning method:** LoRA (via Unsloth + TRL `SFTTrainer`)
- **Intended inference target:** LM Studio (local OpenAI-compatible server) via GGUF export

---

## Training dataset

- **Dataset file (local):** `data/raft/raft_clean.jsonl`
- Uploaded to Colab using `files.upload()`

### JSONL schema
Each row contains:
- `instruction` → question + answering rules
- `input` → context with **oracle** chunk + **distractor** chunks
- `output` → final grounded answer (with citations)

---

## How to run (repro steps)

1. Generate the RAFT dataset locally:
   - `scripts/3_build_raft_dataset.py`
   - `scripts/3b_clean_raft_jsonl.py`
2. Open `notebooks/raft_finetune_unsloth.ipynb` in Google Colab.
3. Run install/setup cells (Unsloth / TRL / Datasets / Accelerate).
4. Upload `raft_clean.jsonl` to Colab when prompted.
5. Run training cells (LoRA + TRL `SFTTrainer`).
6. Save/export artifacts:
   - LoRA adapter (zip)
   - optional merged 16-bit model
   - optional GGUF exports (F16 + Q8_0)
7. Download artifacts and store them locally (recommended: `artifacts/`).

---

## LoRA configuration (from the notebook)

- **LoRA rank (`r`)**: `16`
- **LoRA alpha**: `16`
- **LoRA dropout**: `0.0`

### Target modules
- `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `gate_proj`, `up_proj`, `down_proj`

---

## Training hyperparameters (from the notebook)

- **Max sequence length:** `2048`
- **Per-device batch size:** `2`
- **Gradient accumulation steps:** `8`
- **Effective batch size:** `16` (`2 × 8`)
- **Learning rate:** `2e-4`
- **Warmup steps:** `10`
- **Training steps:** `300`
- **Trainer:** TRL `SFTTrainer`

---

## Outputs produced (Colab)

### Main output
- LoRA adapter folder: `raft_lora_adapter/`
- Downloaded artifact: `raft_lora_adapter.zip`

### Optional export outputs (if export cells are enabled)
- Merged model (16-bit): `merged_model_16bit/`
- GGUF (float16): `qwen2.5-0.5b-raft-f16.gguf`
- GGUF (quantized): `qwen2.5-0.5b-raft-q8_0.gguf`

---

## GGUF export notes (llama.cpp pin for reproducibility)

The notebook pins `llama.cpp` to a known working commit before converting to GGUF:

- **`llama.cpp` commit:** `244641955f6146f7e8474afff7772d427593a534`

This helps make GGUF conversion more reproducible across Colab runs (since `llama.cpp` changes frequently).

---

## Export verification metadata (Cell 8)

After GGUF export/quantization, the notebook prints a small JSON summary to verify outputs and record reproducibility metadata.

It includes:
- base model family
- pinned `llama.cpp` commit
- F16 GGUF path
- Q8_0 GGUF path
- existence checks (`true/false`)
- file sizes (MB)

Example fields:
- `f16_exists`
- `q8_0_exists`
- `f16_size_mb`
- `q8_0_size_mb`

This is used as a lightweight reproducibility record (without hashing).

---

## Where to store artifacts locally (recommended)

These are intentionally **NOT committed to Git** (see `.gitignore`):

- `artifacts/raft_lora_adapter.zip`
- `artifacts/gguf/qwen2.5-0.5b-raft-f16.gguf` (if exported)
- `artifacts/gguf/qwen2.5-0.5b-raft-q8_0.gguf` (if exported)
- `artifacts/merged_model_16bit/` (if exported)

---

## Versions (recommended to record for reproducibility)

Colab package versions can change over time. For stronger reproducibility, record versions after installs:

- `python -V`
- `pip show torch transformers trl unsloth datasets accelerate`

---

## Result summary (honest outcome)

Initial small-scale evaluation (10 questions) showed:

- Base model + RAG performed better than the RAFT fine-tuned model.
- This RAFT run likely underperformed due to:
  - small training set size (~50 examples)
  - short training duration (`300` steps)

This notebook run should be treated as a **proof-of-pipeline** / first fine-tuning pass, not a final model.

---

## Planned improvements (future work)

- Increase RAFT dataset size (more examples + more diverse questions)
- Improve dataset generation quality and JSON cleanup robustness
- Add a held-out evaluation set
- Train longer and compare checkpoints
- Tune prompt formatting / output constraints for better grounding