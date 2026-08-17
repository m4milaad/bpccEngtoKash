"""
KATHE 2026 — Iterative Back-Translation Orchestrator
Runs complete cycles: reverse train → back-translate → forward train → repeat

Usage:
    python iterative_bt.py --cycles 3 --monolingual 50000
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from utils import setup_logging

logger = setup_logging()


def run_cmd(cmd: str, cwd: str = None) -> bool:
    """Run a shell command and return success."""
    logger.info(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 Iterative Back-Translation")
    parser.add_argument("--cycles", type=int, default=3, help="Number of BT cycles")
    parser.add_argument("--monolingual", type=int, default=50000, help="Monolingual sentences per cycle")
    parser.add_argument("--chrf_threshold", type=float, default=0.65, help="chrF++ filter threshold")
    parser.add_argument("--max_synthetic", type=int, default=20000, help="Max synthetic pairs per cycle")
    parser.add_argument("--forward_epochs", type=int, default=3, help="Epochs for forward training per cycle")
    parser.add_argument("--reverse_epochs", type=int, default=2, help="Epochs for reverse training per cycle")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--source", type=str, default="all", choices=["wikipedia", "oscar", "cc100", "all"])
    parser.add_argument("--skip_reverse", action="store_true", help="Skip reverse training (use base model)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.resolve()

    logger.info("=" * 70)
    logger.info("KATHE 2026 — ITERATIVE BACK-TRANSLATION ORCHESTRATOR")
    logger.info("=" * 70)
    logger.info(f"Cycles: {args.cycles}")
    logger.info(f"Monolingual per cycle: {args.monolingual}")
    logger.info(f"chrF++ threshold: {args.chrf_threshold}")
    logger.info(f"Max synthetic per cycle: {args.max_synthetic}")

    # Initial forward model path (your current best)
    forward_model = "models/best"

    for cycle in range(1, args.cycles + 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"CYCLE {cycle}/{args.cycles}")
        logger.info(f"{'='*70}")

        # Step 1: Train reverse model (Kashmiri → English) if not skipped
        reverse_model = f"models/best_reverse_cycle{cycle}"
        if not args.skip_reverse:
            logger.info(f"\n[CYCLE {cycle}] Step 1: Training reverse model...")
            cmd = (
                f"python train_reverse.py "
                f"--model_name {args.model_name} "
                f"--epochs {args.reverse_epochs} "
                f"--batch_size {args.batch_size} "
                f"--gradient_accumulation_steps {args.grad_accum} "
                f"--learning_rate 5e-5"
            )
            if not run_cmd(cmd, cwd=project_root):
                logger.error("[-] Reverse training failed!")
                sys.exit(1)
            logger.info(f"[+] Reverse model saved to: {reverse_model}")
        else:
            reverse_model = None
            logger.info("[*] Skipping reverse training, using base model")

        # Step 2: Run back-translation
        logger.info(f"\n[CYCLE {cycle}] Step 2: Running back-translation...")
        bt_cmd = (
            f"python backtranslate.py "
            f"--model_path {forward_model} "
            f"--reverse_model_path {reverse_model or ''} "
            f"--monolingual_source {args.source} "
            f"--max_monolingual {args.monolingual} "
            f"--output_synthetic data/synthetic_cycle{cycle}.csv "
            f"--output_merged data/bpcc/train_augmented_cycle{cycle}.csv "
            f"--batch_size 16 "
            f"--num_beams 5 "
            f"--chrf_threshold {args.chrf_threshold} "
            f"--max_synthetic {args.max_synthetic} "
            f"--cycle {cycle}"
        )
        if not run_cmd(bt_cmd, cwd=project_root):
            logger.error("[-] Back-translation failed!")
            sys.exit(1)

        # Step 3: Train forward model on augmented data
        logger.info(f"\n[CYCLE {cycle}] Step 3: Training forward model on augmented data...")
        # Temporarily replace train.csv with augmented version
        import shutil
        original_train = project_root / "data/bpcc/train.csv"
        augmented_train = project_root / f"data/bpcc/train_augmented_cycle{cycle}.csv"
        backup_train = project_root / f"data/bpcc/train_original_backup.csv"

        if not backup_train.exists():
            shutil.copy2(original_train, backup_train)

        shutil.copy2(augmented_train, original_train)

        try:
            fw_cmd = (
                f"python train_optimized.py "
                f"--model_name {args.model_name} "
                f"--epochs {args.forward_epochs} "
                f"--batch_size {args.batch_size} "
                f"--gradient_accumulation_steps {args.grad_accum} "
                f"--learning_rate 2e-4"
            )
            if not run_cmd(fw_cmd, cwd=project_root):
                logger.error("[-] Forward training failed!")
                sys.exit(1)

            # Update forward model path for next cycle
            forward_model = f"models/best"
            logger.info(f"[+] Forward model updated: {forward_model}")

        finally:
            # Restore original for next cycle's augmentation
            shutil.copy2(backup_train, original_train)

        logger.info(f"\n[CYCLE {cycle}] COMPLETE")

    logger.info("\n" + "=" * 70)
    logger.info("ALL CYCLES COMPLETE!")
    logger.info(f"Final forward model: {forward_model}")
    logger.info("Run evaluation:")
    logger.info(f"  python eval_validation.py --model_type nllb --model_path {forward_model} --num_beams 6")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()