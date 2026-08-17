"""
KATHE 2026 — Full Pipeline Orchestrator (Both Models from Scratch)
Runs:
1. NLLB: 3-cycle iterative back-translation
2. IndicTrans2: Fresh high-capacity training
3. Ensemble inference across all checkpoints
"""

import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from utils import setup_logging

logger = setup_logging()


def run_cmd(cmd: str, cwd: str = None) -> bool:
    logger.info(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Full Pipeline: NLLB BT + IndicTrans2 + Ensemble")
    parser.add_argument("--nllb_cycles", type=int, default=3)
    parser.add_argument("--nllb_mono", type=int, default=50000)
    parser.add_argument("--it2_epochs", type=int, default=10)
    parser.add_argument("--it2_lora_r", type=int, default=64)
    parser.add_argument("--it2_lora_alpha", type=int, default=128)
    parser.add_argument("--it2_lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    args = parser.parse_args()

    root = Path(__file__).parent.resolve()

    # ---------- NLLB PIPELINE ----------
    logger.info("="*70)
    logger.info("PHASE 1: NLLB Iterative Back-Translation")
    logger.info("="*70)

    # Train reverse model once
    if not run_cmd(f"python train_reverse.py --epochs 3 --batch_size {args.batch_size} --gradient_accumulation_steps {args.grad_accum} --learning_rate 5e-5", cwd=root):
        sys.exit(1)

    for cycle in range(1, args.nllb_cycles + 1):
        logger.info(f"\n--- NLLB Cycle {cycle}/{args.nllb_cycles} ---")

        # Back-translation
        if not run_cmd(
            f"python backtranslate.py --model_path models/best --reverse_model_path models/best_reverse "
            f"--monolingual_source all --max_monolingual {args.nllb_mono} --chrf_threshold 0.65 "
            f"--max_synthetic 20000 --cycle {cycle}", cwd=root):
            sys.exit(1)

        # Retrain forward
        shutil.copy2(root/"data/bpcc/train.csv", root/"data/bpcc/train_original_backup.csv")
        shutil.copy2(root/f"data/bpcc/train_augmented_cycle{cycle}.csv", root/"data/bpcc/train.csv")
        if not run_cmd(
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train_optimized.py "
            f"--epochs 3 --batch_size {args.batch_size} --gradient_accumulation_steps {args.grad_accum} "
            f"--learning_rate 2e-4", cwd=root):
            sys.exit(1)
        shutil.copy2(root/"data/bpcc/train_original_backup.csv", root/"data/bpcc/train.csv")

    logger.info("[+] NLLB pipeline complete. Best model at models/best")

    # ---------- INDICTRANS2 PIPELINE ----------
    logger.info("="*70)
    logger.info("PHASE 2: IndicTrans2 High-Capacity Training")
    logger.info("="*70)

    if not run_cmd(
        f"python train_indictrans2_scratch.py "
        f"--epochs {args.it2_epochs} --batch_size 4 --gradient_accumulation_steps 4 "
        f"--learning_rate {args.it2_lr} --lora_r {args.it2_lora_r} --lora_alpha {args.it2_lora_alpha}",
        cwd=root):
        sys.exit(1)

    logger.info("[+] IndicTrans2 pipeline complete. Best model at models/indictrans2-best")

    # ---------- ENSEMBLE ----------
    logger.info("="*70)
    logger.info("PHASE 3: Multi-Model Ensemble Inference")
    logger.info("="*70)

    # Collect all checkpoints
    nllb_ckpts = ["models/best"]
    for i in range(1, 4):
        p = root / f"models/checkpoint-epoch-{i}"
        if p.exists(): nllb_ckpts.append(str(p))

    it2_ckpts = ["models/indictrans2-best"]
    for i in range(1, args.it2_epochs + 1):
        p = root / f"models/indictrans2-checkpoint-epoch-{i}"
        if p.exists(): it2_ckpts.append(str(p))

    all_models = nllb_ckpts + it2_ckpts
    model_types = ["nllb"] * len(nllb_ckpts) + ["indictrans2"] * len(it2_ckpts)

    logger.info(f"Ensembling {len(all_models)} models: {all_models}")

    if not run_cmd(
        f"python ensemble_inference.py "
        f"--models {' '.join(all_models)} "
        f"--model_types {' '.join(model_types)} "
        f"--test_file englishdev.csv "
        f"--output outputs/submission_ensemble_final.csv "
        f"--num_beams 8 --n_candidates 10 --batch_size 4",
        cwd=root):
        sys.exit(1)

    # Clean submission
    import pandas as pd, re, unicodedata
    df = pd.read_csv(root/"outputs/submission_ensemble_final.csv")
    def clean(t):
        t = str(t)
        t = re.sub(r"^(?:kas[@_/\s0-9]*Arab|<2kas_Arab>|__kas_Arab__|\s+)+", "", t, flags=re.IGNORECASE)
        t = unicodedata.normalize("NFKC", t).replace("ك","ک").replace("ي","ی").replace("ى","ی")
        return re.sub(r"\s+"," ",t).strip()
    df["kashmiri_text"] = df["kashmiri_text"].apply(clean)
    df.to_csv(root/"outputs/submission_final_clean.csv", index=False)
    logger.info(f"[+] Final submission: outputs/submission_final_clean.csv")
    logger.info("="*70)


if __name__ == "__main__":
    main()