"""
KATHE 2026 — Central Configuration
English-to-Kashmiri Machine Translation
"""

import os
from pathlib import Path

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
BPCC_DIR = DATA_DIR / "bpcc"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

# Input file
TEST_FILE = PROJECT_ROOT / "englishdev.csv"

# Create directories
for d in [DATA_DIR, BPCC_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Kaggle Competition
# ============================================================
KAGGLE_COMPETITION = "kathe-2026"

# ============================================================
# Model Configuration
# ============================================================

# Primary model: IndicTrans2 (best for Indic languages)
# Fallback: NLLB-200 distilled (smaller, also supports Kashmiri)

# Choose one:
# MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
MODEL_NAME = "facebook/nllb-200-distilled-600M"  # safer default for mid-range GPUs

# Language codes
# For NLLB: kas_Arab (Kashmiri in Arabic script)
# For IndicTrans2: kas_Arab or kas_Deva
SRC_LANG = "eng_Latn"   # English
TGT_LANG = "kas_Arab"   # Kashmiri (Arabic script — most commonly used)

# ============================================================
# Training Hyperparameters
# ============================================================
TRAIN_CONFIG = {
    "epochs": 3,
    "batch_size": 8,                 # Adjust based on GPU memory
    "gradient_accumulation_steps": 4, # Effective batch size = 8 * 4 = 32
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_source_length": 128,
    "max_target_length": 128,
    "fp16": True,                    # Mixed precision
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "logging_steps": 50,
    "seed": 42,
}

# ============================================================
# LoRA Configuration (PEFT)
# ============================================================
LORA_CONFIG = {
    "r": 16,                     # LoRA rank
    "lora_alpha": 32,            # LoRA scaling
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "bias": "none",
    "task_type": "SEQ_2_SEQ_LM",
}

# ============================================================
# Inference Configuration
# ============================================================
INFERENCE_CONFIG = {
    "batch_size": 16,            # Inference batch size (can be larger)
    "max_length": 128,
    "num_beams": 5,              # Beam search
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
}

# ============================================================
# Submission
# ============================================================
SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"
