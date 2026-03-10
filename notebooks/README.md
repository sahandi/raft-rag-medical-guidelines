# Fine-tuning (RAFT LoRA) — Notebook Summary

This folder contains the notebook used to fine-tune a small local model using **RAFT-style training data** built from:

- an **oracle** evidence chunk
- several **distractor** chunks
- a grounded answer format with source citations

The goal is to teach the model to:

- focus on the correct retrieved evidence
- ignore misleading or irrelevant retrieved text
- produce short grounded answers with citations

---

## Notebook

- **File:** `notebooks/raft_finetune_unsloth.ipynb`
- **Environment:** Google Colab (GPU runtime)

---

## Base model and training method

- **Base model:** `unsloth/Qwen2.5-0.5B-Instruct`
- **Fine-tuning method:** LoRA
- **Training library stack:** Unsloth + TRL `SFTTrainer`
- **Intended inference target:** local GGUF model for LM Studio

---

## Training dataset

- **Local dataset file:** `data/raft/raft.jsonl`
- **Uploaded file in Colab:** `raft.jsonl`
- **Final cleaned dataset size used for retraining:** **135 rows**

### Dataset row structure

Each JSONL row contains:

- `instruction` → the question
- `input` → oracle context plus distractor contexts
- `output` → grounded answer with citations

---

## Training prompt format used in the notebook

The notebook reformats each row into a project-style training prompt with this structure:

- system-like QA rules
- `QUESTION:`
- `SOURCES:`
- labeled source blocks like `[S1]`, `[S2]`, `[S3]`
- `ANSWER:`
- end marker: `<END_ANSWER>`

This format was added to make the fine-tuned model behave more like the project’s real RAG prompt and to make answer stopping cleaner during inference.

---

## Reproducibility setup in the notebook

The notebook includes reproducibility-focused setup steps:

- sets `SEED = 42`
- sets Python / NumPy / Torch seeds
- sets `TOKENIZERS_PARALLELISM=false`
- records package versions into:
  - `colab_versions.txt`
  - `colab_pip_freeze.txt`

It also saves training metadata into:

- `raft_lora_adapter/train_run_meta.json`

That metadata includes:

- base model name
- seed
- max sequence length
- LoRA settings
- training hyperparameters
- dataset row count
- Python / Torch environment info

---

## How to run

1. Build the RAFT dataset locally:

   ```bash
   uv run python scripts/build_raft_dataset.py
   ```

2. Confirm the dataset exists locally:

   ```bash
   ls -lh data/raft/raft.jsonl
   ```

3. Open `notebooks/raft_finetune_unsloth.ipynb` in Google Colab.

4. Run the install/setup cell.

5. Upload `raft.jsonl` when prompted.

6. Run the dataset formatting, model loading, LoRA setup, and training cells.

7. Run the quick sanity-check inference cell.

8. Save the LoRA adapter.

9. Optionally merge and export GGUF files for LM Studio.

---

## LoRA configuration

* **LoRA rank (`r`)**: `16`
* **LoRA alpha**: `16`
* **LoRA dropout**: `0.0`

### Target modules

* `q_proj`
* `k_proj`
* `v_proj`
* `o_proj`
* `gate_proj`
* `up_proj`
* `down_proj`

---

## Training hyperparameters

* **Max sequence length:** `2048`
* **Load in 4-bit:** `True`
* **Per-device batch size:** `2`
* **Gradient accumulation steps:** `8`
* **Effective batch size:** `16`
* **Warmup steps:** `10`
* **Learning rate:** `2e-4`
* **Training steps:** `20`
* **FP16:** `True`
* **Logging steps:** `10`
* **Trainer:** TRL `SFTTrainer`

Additional run controls used in the notebook:

* `seed=42`
* `data_seed=42`
* `dataloader_num_workers=0`
* `report_to="none"`

---

## Quick sanity-check inference

The notebook includes a short real-task sanity-check prompt before export.

That check uses:

* project-style rules
* a real diabetes guideline question
* explicit source formatting
* stopping cleanup using `<END_ANSWER>` and source marker trimming

This helps confirm the fine-tuned model is producing project-style grounded answers before export.

---

## Saved outputs

### Main saved output

* LoRA adapter folder: `raft_lora_adapter/`

### Saved metadata

* `raft_lora_adapter/train_run_meta.json`

### Optional export outputs

If the export cells are run, the notebook can also produce:

* merged model folder: `merged_model_16bit/`
* GGUF (F16): `qwen2.5-0.5b-raft-f16.gguf`
* GGUF (Q8_0): `qwen2.5-0.5b-raft-q8_0.gguf`

---

## GGUF export notes

The notebook pins `llama.cpp` to a known working commit before GGUF conversion:

* **Pinned `llama.cpp` commit:** `244641955f6146f7e8474afff7772d427593a534`

This helps make GGUF export more reproducible across Colab runs.

---

## Local usage after export

After downloading the quantized GGUF, the model can be moved into the local project model folder and loaded in LM Studio for evaluation and demo use.

In this project, the exported RAFT model was later loaded in LM Studio as:

* `qwen2.5-0.5b-raft.gguf`

---

## Honest result summary

This notebook corresponds to the improved RAFT retraining pass used later in project evaluation.

After rebuilding the dataset and retraining:

* **Frozen-context evaluation**

  * Base local model: **2/10**
  * RAFT local model: **5/10**
  * GPT-4o-mini: **9/10**

* **End-to-end RAG evaluation**

  * Base + RAG: **5/10**
  * RAFT + RAG: **6/10**
  * GPT + same retriever + RAG: **10/10**

### Main takeaway

* RAFT improved over the base local model
* the improvement was more visible in frozen-context evaluation
* the local generator still remained the main bottleneck compared with GPT

---

## Notes

* This notebook is part of a portfolio / engineering project, not a production medical system
* The training run is intentionally small and practical
* GGUF export is optional, but needed for LM Studio local inference
* Colab package versions can change over time, so saved version logs are important for reproducibility