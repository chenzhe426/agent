# Knowledge Base – Eval & SFT Makefile
#
# Prerequisites:
#   pip install llmafactory          # for LlamaFactory backend
#   pip install transformers peft     # for HuggingFace backend
#   pip install verl                 # for veRL backend
#
# Quick start – full pipeline:
#   make eval                    # Run evaluation on eval dataset
#   make export-sft              # Export eval results → SFT format
#   make train-sft               # Train with LlamaFactory
#
# Or run everything at once:
#   make full-pipeline

# ── Paths ──────────────────────────────────────────────────────────────────
DATASET       ?= evals/datasets/kb_eval_seed.jsonl
SFT_DATA      := evals/datasets/kb_eval_sft.jsonl
EVAL_OUTPUT   := evals/reports/eval_$(shell date +%Y%m%d_%H%M%S)
SFT_OUTPUT    := evals/training/llamafactory_output
BACKEND       ?= llamafactory
MODEL_ID      ?= meta-llama/Llama-3.1-8B-Instruct
NUM_GPUS      ?= 1
CONFIG_FILE   := evals/configs/baseline.yaml

# ── Evaluation ────────────────────────────────────────────────────────────────
.PHONY: eval
eval:
	python -m evals.scripts.run_eval \
		--dataset $(DATASET) \
		--config $(CONFIG_FILE) \
		--output $(EVAL_OUTPUT).json \
		--parallel 4

# ── Export to SFT ────────────────────────────────────────────────────────────
.PHONY: export-sft
export-sft:
	python -m evals.scripts.export_sft \
		--dataset $(DATASET) \
		--output $(SFT_DATA)

# ── SFT Training ─────────────────────────────────────────────────────────────
.PHONY: train-sft
train-sft: export-sft
	python -m evals.scripts.train_sft \
		--backend $(BACKEND) \
		--sft_data $(SFT_DATA) \
		--model_id $(MODEL_ID) \
		--output_dir $(SFT_OUTPUT) \
		--num_gpus $(NUM_GPUS) \
		--epochs 3 \
		--max_steps 1000

# ── LlamaFactory shortcut ─────────────────────────────────────────────────────
.PHONY: train-llamafactory
train-llamafactory: export-sft
	python -m evals.scripts.train_sft \
		--backend llamafactory \
		--sft_data $(SFT_DATA) \
		--model_id $(MODEL_ID) \
		--output_dir $(SFT_OUTPUT) \
		--config evals/configs/sft_llamafactory.yaml \
		--num_gpus $(NUM_GPUS)

# ── HuggingFace shortcut ─────────────────────────────────────────────────────
.PHONY: train-hf
train-hf: export-sft
	python -m evals.scripts.train_sft \
		--backend huggingface \
		--sft_data $(SFT_DATA) \
		--model_id $(MODEL_ID) \
		--output_dir evals/training/hf_output \
		--num_gpus $(NUM_GPUS)

# ── Full pipeline ────────────────────────────────────────────────────────────
.PHONY: full-pipeline
full-pipeline: eval export-sft train-sft
	@echo ""
	@echo "=== Full pipeline complete ==="
	@echo "  Eval report:    $(EVAL_OUTPUT).json"
	@echo "  SFT data:       $(SFT_DATA)"
	@echo "  Model output:   $(SFT_OUTPUT)"

# ── Clean ────────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -rf evals/training/*
	rm -rf evals/reports/eval_*
	rm -f evals/datasets/*_sft.jsonl

# ── Help ─────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo "Knowledge Base – Eval & SFT Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  eval              Run evaluation on dataset"
	@echo "  export-sft        Export eval dataset to SFT format"
	@echo "  train-sft         Run SFT training (backend from BACKEND=$(BACKEND))"
	@echo "  train-llamafactory  Train with LlamaFactory"
	@echo "  train-hf          Train with HuggingFace Trainer"
	@echo "  full-pipeline     eval → export-sft → train-sft"
	@echo "  clean             Remove training outputs"
	@echo ""
	@echo "Variables:"
	@echo "  DATASET=$(DATASET)"
	@echo "  SFT_DATA=$(SFT_DATA)"
	@echo "  BACKEND=$(BACKEND)"
	@echo "  MODEL_ID=$(MODEL_ID)"
	@echo "  NUM_GPUS=$(NUM_GPUS)"
