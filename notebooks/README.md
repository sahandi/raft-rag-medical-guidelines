# Fine-tuning (RAFT LoRA) — Notebook Summary

This folder contains the notebook used to fine-tune a small model using RAFT-style data (oracle + distractors).
The goal is to teach the model to ignore distracting context and answer using the correct retrieved evidence.

## Notebook
- File: `notebooks/raft_finetune_unsloth.ipynb`
- Environment: Google Colab (GPU runtime)

## Base model
- Base: `unsloth/Qwen2.5-0.5B-Instruct`
- Intended inference target: LM Studio (local OpenAI-compatible server)

## Training dataset
- Local file: `data/raft/raft_clean.jsonl`
- Uploaded to Colab via `files.upload()`
- JSONL fields: `instruction`, `input`, `output`
  - `instruction`: question + answering rules
  - `input`: context containing **oracle** chunk + **distractor** chunks
  - `output`: final answer with citations

## How to run (repro steps)
1. Generate `data/raft/raft_clean.jsonl` locally (see `scripts/` RAFT generation pipeline).
2. Open `notebooks/raft_finetune_unsloth.ipynb` in Colab.
3. Run install cells (Unsloth/TRL/Datasets/Accelerate).
4. Upload `data/raft/raft_clean.jsonl` to Colab when prompted.
5. Run training cells (LoRA + TRL `SFTTrainer`).
6. Save/export outputs (adapter zip, optional merged/GGUF exports).
7. Download artifacts and store them locally under `artifacts/` (recommended).

## LoRA configuration (from the notebook)
- LoRA rank (r): [e.g. 16]
- LoRA alpha: [e.g. 16]
- LoRA dropout: [e.g. 0.0]
- Target modules:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - `gate_proj`, `up_proj`, `down_proj`

## Training hyperparameters (from the notebook)
- Max sequence length: [e.g. 2048]
- Per-device batch size: [e.g. 2]
- Gradient accumulation: [e.g. 8]
- Effective batch size: (batch_size * grad_accum) = [e.g. 16]
- Learning rate: [e.g. 2e-4]
- Warmup steps: [e.g. 10]
- Training steps: [e.g. 300]
- Trainer: TRL `SFTTrainer`

## Outputs produced (Colab)
- LoRA adapter folder:
  - `raft_lora_adapter/`
- Downloaded artifact:
  - `raft_lora_adapter.zip`

Optional exports (if enabled in the notebook):
- merged model (16-bit): `merged_model_16bit/`
- GGUF float16: `qwen2.5-0.5b-raft-f16.gguf`
- GGUF quantized: `qwen2.5-0.5b-raft-q8_0.gguf`

## Where to store artifacts locally (recommended)
- `artifacts/raft_lora_adapter.zip`
- `artifacts/gguf/qwen2.5-0.5b-raft-q8_0.gguf` (if exported)
- `artifacts/merged_model_16bit/` (if exported)

## Versions (record for reproducibility)
- torch: [e.g. 2.x]
- transformers: [version]
- trl: [version]
- unsloth: [version]
- datasets: [version]

## Result summary (honest outcome)
Initial small-scale evaluation (10 questions) showed:
- Base model + RAG performed better than the RAFT-fine-tuned model.
- RAFT run likely underperformed due to small training set size and limited training steps.

## Planned improvements (future work)
- Increase RAFT dataset size (more examples)
- Improve formatting consistency + stricter generation cleanup
- Add a held-out evaluation set and run longer training
