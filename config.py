
"""Configuration for the SQL QLoRA experiment."""

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

ADAPTER_DIR = "lora_adapter"
ADAPTER_NAME = "sql-lora"

SQL_DATASET = "b-mc2/sql-create-context"

# REAL EXPERIMENT
TRAIN_SIZE = 4000
TEST_SIZE = 500
SEED = 42

# Perplexity
PPL_DATASET = "wikitext"
PPL_CONFIG = "wikitext-2-raw-v1"
PPL_SPLIT = "test"
PPL_NUM_DOCS = 100

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LEN = 512
WARMUP_RATIO = 0.03
LOGGING_STEPS = 20

# Generation
MAX_NEW_TOKENS = 128
GEN_TEMPERATURE = 0.0
GEN_SEED = 0

RESULTS_DIR = "results"

print("Config written.")
print("TRAIN_SIZE =", TRAIN_SIZE)
print("TEST_SIZE =", TEST_SIZE)
