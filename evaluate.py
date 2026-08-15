"""
KATHE 2026 — Evaluation Script
Compute BLEU and chrF++ scores (the competition metric is their geometric mean).

Usage:
    python evaluate.py --predictions outputs/submission.csv --references data/bpcc/val.csv
"""

import argparse
import math
import sys

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import sacrebleu

from config import SUBMISSION_FILE, BPCC_DIR
from utils import setup_logging, safe_print

logger = setup_logging()


def compute_scores(predictions: list[str], references: list[str]) -> dict:
    """
    Compute BLEU, chrF++, and their geometric mean.
    
    The competition uses: score = sqrt(BLEU * chrF++)
    """
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)

    bleu_score = bleu.score
    chrf_score = chrf.score

    if bleu_score > 0 and chrf_score > 0:
        geo_mean = math.sqrt(bleu_score * chrf_score)
    else:
        geo_mean = 0.0

    return {
        "bleu": bleu_score,
        "chrf++": chrf_score,
        "geometric_mean": geo_mean,
        "bleu_details": str(bleu),
        "chrf_details": str(chrf),
    }


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 -- Evaluation")
    parser.add_argument(
        "--predictions",
        type=str,
        default=str(SUBMISSION_FILE),
        help="Path to predictions CSV (must have 'kashmiri_text' column)",
    )
    parser.add_argument(
        "--references",
        type=str,
        default=str(BPCC_DIR / "val.csv"),
        help="Path to references CSV (must have 'kashmiri' column)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("KATHE 2026 -- Evaluation")
    logger.info("=" * 60)

    pred_df = pd.read_csv(args.predictions)
    if "kashmiri_text" in pred_df.columns:
        predictions = pred_df["kashmiri_text"].astype(str).tolist()
    elif "kashmiri" in pred_df.columns:
        predictions = pred_df["kashmiri"].astype(str).tolist()
    else:
        logger.error("[-] Predictions file must have 'kashmiri_text' or 'kashmiri' column")
        logger.error(f"    Found columns: {list(pred_df.columns)}")
        return

    ref_df = pd.read_csv(args.references)
    if "kashmiri" in ref_df.columns:
        references = ref_df["kashmiri"].astype(str).tolist()
    elif "kashmiri_text" in ref_df.columns:
        references = ref_df["kashmiri_text"].astype(str).tolist()
    else:
        logger.error("[-] References file must have 'kashmiri' or 'kashmiri_text' column")
        logger.error(f"    Found columns: {list(ref_df.columns)}")
        return

    min_len = min(len(predictions), len(references))
    if len(predictions) != len(references):
        logger.warning(
            f"[!] Length mismatch: {len(predictions)} predictions vs {len(references)} references. "
            f"Using first {min_len} pairs."
        )
        predictions = predictions[:min_len]
        references = references[:min_len]

    logger.info(f"[*] Evaluating {len(predictions)} translation pairs...\n")

    scores = compute_scores(predictions, references)

    safe_print("+" + "-" * 52 + "+")
    safe_print("|           KATHE 2026 -- Evaluation Results         |")
    safe_print("+" + "-" * 52 + "+")
    safe_print(f"|  BLEU Score:       {scores['bleu']:>8.2f}                        |")
    safe_print(f"|  chrF++ Score:     {scores['chrf++']:>8.2f}                        |")
    safe_print("+" + "-" * 52 + "+")
    safe_print(f"|  * Geometric Mean: {scores['geometric_mean']:>8.2f}  (competition metric)   |")
    safe_print("+" + "-" * 52 + "+")

    safe_print(f"\nBLEU details:   {scores['bleu_details']}")
    safe_print(f"chrF++ details: {scores['chrf_details']}")

    return scores


if __name__ == "__main__":
    main()

