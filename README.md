# SQL FP8 KV-Cache Study

## Objective

This project investigates whether FP8 KV-cache compression can reduce
KV-cache memory usage while preserving downstream SQL generation quality.

## Base model

Qwen/Qwen2.5-3B-Instruct

## Fine-tuning

QLoRA:
- 4-bit NF4 quantization
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Learning rate: 2e-4
- Epochs: 3
- Maximum sequence length: 512

## Dataset

b-mc2/sql-create-context

Training examples: 4000
Held-out test examples: 500

## Evaluation

The same trained LoRA adapter is evaluated under:

1. Automatic/native KV-cache precision
2. FP8 KV-cache precision

Evaluation measures include:
- SQL exact-match accuracy
- WikiText-2 perplexity
- KV-cache memory usage

## Hardware

LoRA training:
- NVIDIA RTX 4090

KV-cache evaluation:
- NVIDIA RTX 4090

## Reproducibility

See the Python scripts and configuration files in this repository.


sql-fp8-kv-cache-study/
│
├── README.md
├── requirements.txt
├── requirements_rtx4090.txt
├── config.py
│
├── 01_finetune_lora.py
├── 02_generate_predictions.py
├── 03_evaluate.py
├── 04_compare_and_plot.py
│
├── lora_adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── vocab.json
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── added_tokens.json
│   └── chat_template.jinja
│
├── results/
│   ├── held_out_test.jsonl
│   ├── sql_preds_auto.jsonl
│   ├── perplexity_auto.json
│   ├── summary_auto.json
│   ├── mismatches_auto.jsonl
│   ├── sql_preds_fp8.jsonl
│   ├── perplexity_fp8.json
│   ├── summary_fp8.json
│   ├── mismatches_fp8.jsonl
│   ├── comparison_table.csv
│   └── comparison_plot.png
│
└── .gitignore
