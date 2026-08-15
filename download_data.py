"""
KATHE 2026 — Download & Prepare BPCC Training Data

Downloads the official BPCC (Bharat Parallel Corpus Collection) English-Kashmiri
data from Hugging Face (ai4bharat/BPCC) using direct file download for speed.

Usage:
    python download_data.py
"""

import os
import sys
from pathlib import Path

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
from huggingface_hub import hf_hub_download

from config import DATA_DIR, BPCC_DIR
from utils import setup_logging, safe_print

logger = setup_logging()

# Direct BPCC files containing Kashmiri
BPCC_KASHMIRI_FILES = [
    "bpcc-seed-latest/kas_Arab.tsv",
    "nllb-filtered/kas_Arab.tsv",
    "daily/kas_Arab.tsv",
    "nllb-seed/kas_Arab.tsv",
    "bpcc-seed-v2/kas_Arab.tsv",
]


def download_bpcc_files() -> pd.DataFrame:
    """Download individual Kashmiri TSV files directly from BPCC repo."""
    all_english = []
    all_kashmiri = []

    for rel_path in BPCC_KASHMIRI_FILES:
        logger.info(f"[*] Downloading {rel_path}...")
        try:
            local_file = hf_hub_download(
                repo_id="ai4bharat/BPCC",
                filename=rel_path,
                repo_type="dataset",
            )
            logger.info(f"    Downloaded to: {local_file}")

            # Read TSV
            df = pd.read_csv(local_file, sep="\t", on_bad_lines="skip")
            cols = list(df.columns)
            logger.info(f"    Loaded {len(df)} rows. Columns: {cols}")

            src_col = None
            tgt_col = None

            for col in cols:
                cl = str(col).lower()
                if any(k in cl for k in ["english", "eng", "en_", "source", "src"]):
                    if not src_col:
                        src_col = col
                if any(k in cl for k in ["kashmiri", "kas", "ks_", "target", "tgt"]):
                    if not tgt_col:
                        tgt_col = col

            if not src_col or not tgt_col:
                if len(cols) >= 2:
                    src_col, tgt_col = cols[0], cols[1]

            if src_col and tgt_col:
                logger.info(f"    Mapping: {src_col} -> {tgt_col}")
                for _, row in df.iterrows():
                    e = str(row.get(src_col, "")).strip()
                    k = str(row.get(tgt_col, "")).strip()
                    if e and k and e.lower() != "nan" and k.lower() != "nan":
                        all_english.append(e)
                        all_kashmiri.append(k)

        except Exception as e:
            logger.info(f"[-] Could not download {rel_path}: {e}")
            continue

    df = pd.DataFrame({"english": all_english, "kashmiri": all_kashmiri})
    logger.info(f"[+] Total raw BPCC Kashmiri pairs collected: {len(df)}")
    return df


def clean_and_split(df: pd.DataFrame) -> dict:
    """Clean data and split into train / val."""
    logger.info("[*] Cleaning parallel corpus...")
    init_len = len(df)

    df = df.dropna(subset=["english", "kashmiri"])
    df["english"] = df["english"].astype(str).str.strip()
    df["kashmiri"] = df["kashmiri"].astype(str).str.strip()
    df = df[df["english"] != ""]
    df = df[df["kashmiri"] != ""]
    df = df.drop_duplicates(subset=["english", "kashmiri"])

    # Filter out short sentences
    df = df[df["english"].str.len() >= 3]
    df = df[df["kashmiri"].str.len() >= 3]

    logger.info(f"    {init_len} -> {len(df)} pairs after cleaning")

    if len(df) == 0:
        logger.error("[-] No valid pairs after cleaning")
        return None

    # 95/5 train/val split
    val_count = min(1000, max(50, int(len(df) * 0.05)))
    val_df = df.sample(n=val_count, random_state=42)
    train_df = df.drop(val_df.index)

    train_path = BPCC_DIR / "train.csv"
    val_path = BPCC_DIR / "val.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")

    logger.info(f"[+] Saved train: {len(train_df)} rows -> {train_path}")
    logger.info(f"[+] Saved val:   {len(val_df)} rows -> {val_path}")

    logger.info("\n[*] Sample BPCC pairs:")
    for i in range(min(3, len(train_df))):
        logger.info(f"    EN: {train_df['english'].iloc[i]}")
        try:
            logger.info(f"    KS: {train_df['kashmiri'].iloc[i]}")
        except Exception:
            pass
        logger.info("")

    return {"train": train_path, "val": val_path}


def main():
    logger.info("=" * 60)
    logger.info("KATHE 2026 -- BPCC Kashmiri Parallel Data Download")
    logger.info("=" * 60)

    df = download_bpcc_files()
    if len(df) == 0:
        logger.error("[-] No Kashmiri sentence pairs found in BPCC files.")
        sys.exit(1)

    result = clean_and_split(df)
    if result:
        logger.info("\n[+] BPCC data pipeline complete and ready for training!")
        logger.info(f"    Train file: {result['train']}")
        logger.info(f"    Val file:   {result['val']}")


if __name__ == "__main__":
    main()



