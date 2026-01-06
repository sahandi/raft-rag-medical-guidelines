# Fine-tuning (RAFT LoRA) — Notebook Summary

This folder contains the notebook used to fine-tune a small local model using RAFT-style data (oracle + distractors).
The goal was to teach the model to ignore distracting context and answer using the correct retrieved evidence.

## Notebook
- File: `notebooks/raft_finetune_unsloth.ipynb`
- Environment: Google Colab (GPU)

## Base model
- Model: `unsloth/Qwen2.5-0.5B-Instruct`
- Inference target: LM Studio (local OpenAI-compatible server)

## Training dataset
- File used: `data/raft/raft_clean.jsonl` (generated locally, then uploaded to Colab)
- Format: JSONL with fields: `instruction`, `input`, `output`
- Typical example structure:
  - `instruction`: question + answering rules
  - `input`: context containing **oracle** chunk + **distractor** chunks
  - `output`: answer with citations

## LoRA configuration
- LoRA rank (r): [PUT YOUR r, e.g. 16]
- LoRA alpha: [PUT YOUR alpha, e.g. 16]
- LoRA dropout: [PUT YOUR dropout, e.g. 0.0]
- Target modules:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - `gate_proj`, `up_proj`, `down_proj`

## Training hyperparameters
- Max sequence length: [e.g. 2048]
- Per-device batch size: [e.g. 2]
- Gradient accumulation: [e.g. 8]
- Effective batch size: (batch_size * grad_accum) = [e.g. 16]
- Learning rate: [e.g. 2e-4]
- Warmup steps: [e.g. 10]
- Training steps: [e.g. 300]
- Trainer: TRL `SFTTrainer`

## Outputs produced
- LoRA adapter folder (saved in Colab):
  - `raft_lora_adapter/`
- Downloaded artifact:
  - `raft_lora_adapter.zip`
- Optional exports (if enabled in the notebook):
  - merged model (16-bit): `merged_model_16bit/`
  - GGUF float16: `qwen2.5-0.5b-raft-f16.gguf`
  - GGUF quantized: `qwen2.5-0.5b-raft-q8_0.gguf`

## Result summary (honest outcome)
Initial small-scale evaluation (10 questions) showed:
- Base model + RAG performed better than the RAFT-fine-tuned model.
- RAFT run likely underperformed due to small training set size and limited training steps.

Planned next improvements (future work):
- Increase RAFT dataset size (more examples)
- Improve formatting consistency + stricter JSON generation
- Add held-out evaluation set and run longer training

