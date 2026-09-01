
"""
QLoRA fine-tuning for the SQL task.

Training:
    T4 GPU
    4-bit NF4 base model
    LoRA adapter

Important:
    KV-cache dtype is NOT involved during training.

The saved adapter is later evaluated on an L4 under:
    1. auto KV cache
    2. FP8 E4M3 KV cache
"""

import math
import os
import random

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

import config as cfg


PROMPT_TEMPLATE = """You are a SQL expert. Given the database schema and a question, write the SQL query that answers it.

### Schema:
{context}

### Question:
{question}

### SQL:
"""


def build_example(row, tokenizer):
    """
    Build a causal-LM example where:
      - prompt tokens have label -100
      - SQL completion tokens are trained
    """

    prompt = PROMPT_TEMPLATE.format(
        context=row["context"],
        question=row["question"],
    )

    completion = row["answer"].strip()

    if tokenizer.eos_token:
        completion += tokenizer.eos_token

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    completion_ids = tokenizer(
        completion,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    # We want to preserve the completion whenever possible.
    if len(completion_ids) >= cfg.MAX_SEQ_LEN:
        completion_ids = completion_ids[: cfg.MAX_SEQ_LEN - 1]

    max_prompt_tokens = cfg.MAX_SEQ_LEN - len(completion_ids)

    if max_prompt_tokens <= 0:
        raise ValueError("MAX_SEQ_LEN is too small for the completion.")

    prompt_ids = prompt_ids[:max_prompt_tokens]

    input_ids = prompt_ids + completion_ids

    labels = (
        [-100] * len(prompt_ids)
        + completion_ids
    )

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class CausalLMDataCollator:
    """
    Pads input_ids, attention_mask and labels to the longest
    sequence in the batch.

    This preserves -100 masking on the prompt.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        attention_mask = []
        labels = []

        pad_id = self.tokenizer.pad_token_id

        for x in features:
            length = len(x["input_ids"])
            padding = max_len - length

            input_ids.append(
                x["input_ids"] + [pad_id] * padding
            )

            attention_mask.append(
                x["attention_mask"] + [0] * padding
            )

            labels.append(
                x["labels"] + [-100] * padding
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


def main():

    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for QLoRA training.")

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    os.makedirs(cfg.ADAPTER_DIR, exist_ok=True)

    print(f"Loading dataset: {cfg.SQL_DATASET}")

    ds = load_dataset(
        cfg.SQL_DATASET,
        split="train",
    ).shuffle(seed=cfg.SEED)

    needed = cfg.TRAIN_SIZE + cfg.TEST_SIZE

    if needed > len(ds):
        raise ValueError(
            f"Requested {needed} examples, "
            f"but dataset only has {len(ds)}."
        )

    train_ds = ds.select(range(cfg.TRAIN_SIZE))

    test_ds = ds.select(
        range(
            cfg.TRAIN_SIZE,
            needed,
        )
    )

    test_path = os.path.join(
        cfg.RESULTS_DIR,
        "held_out_test.jsonl",
    )

    test_ds.to_json(
        test_path,
        force_ascii=False,
    )

    print(
        f"Train: {len(train_ds)} | "
        f"Held-out test: {len(test_ds)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.BASE_MODEL,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # --------------------------------------------------------
    # T4 = compute capability 7.5
    # Use FP16, NOT BF16.
    # --------------------------------------------------------

    major, minor = torch.cuda.get_device_capability()

    use_bf16 = major >= 8

    compute_dtype = (
        torch.bfloat16
        if use_bf16
        else torch.float16
    )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"Compute capability: {major}.{minor}"
    )

    print(
        "QLoRA compute dtype:",
        "BF16" if use_bf16 else "FP16",
    )

    # --------------------------------------------------------
    # 4-bit NF4 QLoRA
    # --------------------------------------------------------

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model
    )

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    lora_config = LoraConfig(
        r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=cfg.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    print("Tokenizing training set...")

    tokenized_train = train_ds.map(
        lambda row: build_example(
            row,
            tokenizer,
        ),
        remove_columns=train_ds.column_names,
        desc="Tokenizing",
    )

    collator = CausalLMDataCollator(
        tokenizer
    )

    # --------------------------------------------------------
    # Training schedule
    # --------------------------------------------------------

    batches_per_epoch = math.ceil(
        len(tokenized_train)
        / cfg.PER_DEVICE_BATCH_SIZE
        / cfg.GRAD_ACCUM_STEPS
    )

    total_steps = max(
        1,
        batches_per_epoch * cfg.NUM_EPOCHS,
    )

    warmup_steps = max(
        1,
        round(
            total_steps * cfg.WARMUP_RATIO
        ),
    )

    print(
        f"Training steps: {total_steps}"
    )

    print(
        f"Warmup steps: {warmup_steps}"
    )

    # --------------------------------------------------------
    # TrainingArguments
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir="training_checkpoints",

        per_device_train_batch_size=(
            cfg.PER_DEVICE_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            cfg.GRAD_ACCUM_STEPS
        ),

        learning_rate=cfg.LEARNING_RATE,

        num_train_epochs=cfg.NUM_EPOCHS,

        warmup_steps=warmup_steps,

        logging_steps=cfg.LOGGING_STEPS,

        fp16=not use_bf16,
        bf16=use_bf16,

        gradient_checkpointing=True,

        optim="paged_adamw_8bit",

        save_strategy="epoch",

        report_to="none",

        seed=cfg.SEED,

        remove_unused_columns=False,

        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=collator,
    )

    print("Starting training...")

    trainer.train()

    # --------------------------------------------------------
    # Save adapter
    # --------------------------------------------------------

    model.save_pretrained(
        cfg.ADAPTER_DIR
    )

    tokenizer.save_pretrained(
        cfg.ADAPTER_DIR
    )

    print(
        f"LoRA adapter saved to: "
        f"{cfg.ADAPTER_DIR}"
    )


if __name__ == "__main__":
    main()
