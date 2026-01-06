# Fine-tuning (RAFT LoRA) — Notebook Summary

This folder contains the notebook used to fine-tune a small model using RAFT-style data (oracle + distractors).
The goal is to teach the model to ignore distracting context and answer using the correct retrieved evidence.

## Notebook
- File: `notebooks/raft_finetune_unsloth.ipynb`
- Environment: Google Colab (GPU runtime)

## Base model
- Base model: `unsloth/Qwen2.5-0.5B-Instruct`
- Intended inference target: LM Studio (local OpenAI-compatible server)

## Training dataset
- Dataset file (local): `data/raft/raft_clean.jsonl`
- Uploaded to Colab via `files.upload()`
- JSONL fields: `instruction`, `input`, `output`
  - `instruction`: question + answering rules
  - `input`: context containing **oracle** chunk + **distractor** chunks
  - `output`: final answer with citations

## How to run (repro steps)
1. Generate `data/raft/raft_clean.jsonl` locally (see `scripts/3_build_raft_dataset.py` + `scripts/3b_clean_raft_jsonl.py`).
2. Open `notebooks/raft_finetune_unsloth.ipynb` in Colab.
3. Run install cells (Unsloth / TRL / Datasets / Accelerate).
4. Upload `raft_clean.jsonl` to Colab when prompted.
5. Run training cells (LoRA + TRL `SFTTrainer`).
6. Save/export outputs (adapter zip; optional merged model / GGUF exports).
7. Download artifacts and store them locally (recommended: `artifacts/`).

## LoRA configuration (from the notebook)
- LoRA rank (r): `16`
- LoRA alpha: `16`
- LoRA dropout: `0.0`
- Target modules:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - `gate_proj`, `up_proj`, `down_proj`

## Training hyperparameters (from the notebook)
- Max sequence length: `2048`
- Per-device batch size: `2`
- Gradient accumulation: `8`
- Effective batch size: `16` (2 × 8)
- Learning rate: `2e-4`
- Warmup steps: `10`
- Training steps: `300`
- Trainer: TRL `SFTTrainer`

## Outputs produced (Colab)
- LoRA adapter folder: `raft_lora_adapter/`
- Downloaded artifact: `raft_lora_adapter.zip`

Optional exports (if enabled in the notebook):
- merged model (16-bit): `merged_model_16bit/`
- GGUF float16: `qwen2.5-0.5b-raft-f16.gguf`
- GGUF quantized: `qwen2.5-0.5b-raft-q8_0.gguf`

## Where to store artifacts locally (recommended)
These are intentionally NOT committed to Git (see `.gitignore`):
- `artifacts/raft_lora_adapter.zip`
- `artifacts/gguf/qwen2.5-0.5b-raft-q8_0.gguf` (if exported)
- `artifacts/merged_model_16bit/` (if exported)

## Versions (record for reproducibility)
Colab versions change. If you want full reproducibility, record versions after installs:
- `python -V`
- `pip show torch transformers trl unsloth datasets accelerate`

## Result summary (honest outcome)
Initial small-scale evaluation (10 questions) showed:
- Base model + RAG performed better than the RAFT-fine-tuned model.
- The RAFT run likely underperformed due to small training set size (50 examples) and short training (300 steps).

## Planned improvements (future work)
- Increase RAFT dataset size (more examples, more diverse questions)
- Add stricter output formatting + more robust JSON cleanup during dataset generation
- Add a held-out evaluation set and train longer
