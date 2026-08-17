# KATHE 2026 — English to Kashmiri Machine Translation

> **Competition**: [KATHE 2026: AI Challenge for Kashmiri Language Translation](https://kaggle.com/competitions/kathe-2026)  
> **Organized by**: Gaash Lab, NIT Srinagar | Bureau of Indian Standards (BIS)  
> **Task**: English → Kashmiri machine translation  

## Overview

This project implements an English-to-Kashmiri machine translation pipeline for the KATHE 2026 Kaggle competition. It supports both zero-shot inference with pretrained multilingual models and fine-tuning on the BPCC (Bharat Parallel Corpus Collection) using LoRA for parameter-efficient training.

## Methodology

### 1. Data Preparation
- **Corpus**: BPCC (Bharat Parallel Corpus Collection) — English↔Kashmiri parallel sentences
- **Preprocessing**: 
  - Filter empty/duplicate pairs
  - Length filtering (max 128 tokens source/target)
  - Train/validation split (95/5)
  - Kashmiri in Arabic script (`kas_Arab`)

### 2. Base Models
- **NLLB-200 Distilled 600M** (default): Meta's multilingual model covering 200 languages, including Kashmiri (`kas_Arab`)
- **IndicTrans2 EN→Indic 1B** (alternative): AI4Bharat's model optimized for Indian languages

### 3. Fine-Tuning Strategy: LoRA (Low-Rank Adaptation)
- **Why LoRA**: Full fine-tuning requires >24GB VRAM; LoRA freezes base model and trains only low-rank adapter matrices (~0.5-2% parameters)
- **Configuration** (`config.py:71-78`):
  - Rank `r=32`, Alpha `64` (α=2r)
  - Dropout `0.05`
  - Target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `fc1`, `fc2` (attention + FFN)
  - Task type: `SEQ_2_SEQ_LM`

### 4. Training Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 5 | Balance convergence vs overfitting |
| Batch size | 4 (+ grad accum 4) | Effective batch 16, fits 8GB VRAM |
| Learning rate | 5e-5 | Lower LR for stable adapter training |
| Warmup | 10% | Stabilizes early training |
| Label smoothing | 0.1 | Prevents overconfidence |
| Precision | FP16 | Memory efficiency |
| Seed | 42 | Reproducibility |

### 5. Inference & Decoding
- **Beam search**: 8 beams, length penalty 1.0
- **Constraints**: No repeat 3-grams, early stopping
- **Batch size**: 16 (inference can use larger batches)

### 6. Ensemble (Optional)
- **MBR (Minimum Bayes Risk)**: Generate N-best candidates (n=5) from multiple checkpoints, rerank by expected chrF++ against each other
- **Checkpoints**: Best model + epoch checkpoints (e.g., epoch 3, 5)

### 7. Evaluation
- **Metrics**: BLEU (n-gram precision) + chrF++ (character n-gram F-score)
- **Competition score**: Geometric mean `√(BLEU × chrF++)`
- **Validation**: Held-out BPCC split (separate from test)

## Model

- **Primary**: [NLLB-200 Distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) — Meta's multilingual translation model supporting 200 languages including Kashmiri
- **Alternative**: [IndicTrans2](https://huggingface.co/ai4bharat/indictrans2-en-indic-1B) — AI4Bharat's model specifically designed for Indian languages
- **Fine-tuning**: LoRA (Low-Rank Adaptation) via PEFT for memory-efficient training on mid-range GPUs

## Evaluation

Submissions are scored using the **geometric mean of BLEU and chrF++**:

```
score = √(BLEU × chrF++)
```

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone <repo-url>
cd etk

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Zero-Shot Translation (Fastest)

```bash
# Translate test sentences using the pretrained model (no training needed)
python inference.py
```

This generates `outputs/submission.csv` ready for Kaggle submission.

### 3. Fine-Tune on BPCC (Better Scores)

```bash
# Step 1: Download BPCC training data from Kaggle
python download_data.py

# Step 2: Fine-tune the model
python train.py

# Step 3: Run inference with the fine-tuned model
python inference.py --model_path models/best
```

### 4. Evaluate & Submit

```bash
# Evaluate on validation set
python evaluate.py --predictions outputs/submission.csv --references data/bpcc/val.csv

# Validate submission format
python submission.py

# Upload to Kaggle
python submission.py --upload
```

## Project Structure

```
etk/
├── config.py           # Central configuration (model, hyperparams, paths)
├── download_data.py    # Download BPCC data from Kaggle
├── train.py            # LoRA fine-tuning on BPCC
├── inference.py        # Zero-shot or fine-tuned translation
├── evaluate.py         # BLEU + chrF++ evaluation
├── submission.py       # Format & upload submission to Kaggle
├── utils.py            # Helper utilities
├── requirements.txt    # Python dependencies
├── englishdev.csv      # Test data (1730 English sentences)
├── LICENSE             # MIT License
├── data/               # Training data
│   └── bpcc/           # Processed BPCC train/val splits
├── models/             # Saved model checkpoints
├── outputs/            # Submission CSVs
└── logs/               # Training logs
```

## Configuration

Edit `config.py` to customize:
- **Model**: Switch between NLLB-200 and IndicTrans2
- **Training**: Epochs, batch size, learning rate, LoRA rank
- **Inference**: Beam search parameters, batch size
- **Paths**: Data directories, output locations

## Hardware Requirements

- **GPU (recommended)**: 8-16GB VRAM (RTX 3060/3070/4060)
- **RAM**: 16GB+
- **Disk**: ~5GB for model weights + training data

## Citation

```
Gaash Lab NITSGR. KATHE 2026: AI Challenge for Kashmiri Language Translation.
https://kaggle.com/competitions/kathe-2026, 2026. Kaggle.
```

## License

MIT License — see [LICENSE](LICENSE)
