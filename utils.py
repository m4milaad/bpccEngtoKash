"""
KATHE 2026 — Utility Functions
"""

import logging
import sys
from pathlib import Path

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import torch


def safe_print(msg: str):
    """Print message safely avoiding Windows charmap encoding errors."""
    try:
        print(msg)
    except UnicodeEncodeError:
        safe_msg = msg.encode("ascii", "replace").decode("ascii")
        print(safe_msg)


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """Configure logging for the project."""
    stream_handler = logging.StreamHandler(sys.stdout)
    handlers = [stream_handler]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("kathe2026")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        safe_print(f"[+] Using GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        safe_print("[!] No GPU found -- using CPU (this will be slow)")
    return device


def print_gpu_summary():
    """Print name + VRAM for every visible GPU. Handy on Kaggle to confirm
    both T4s are actually visible before kicking off a multi-GPU run."""
    n = torch.cuda.device_count()
    if n == 0:
        safe_print("[!] No GPU found -- using CPU (this will be slow)")
        return
    safe_print(f"[+] Detected {n} GPU(s):")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        safe_print(f"    GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB VRAM)")


def load_test_data(test_file: Path) -> pd.DataFrame:
    """Load the test CSV file (englishdev.csv)."""
    df = pd.read_csv(test_file)
    safe_print(f"[+] Loaded {len(df)} test sentences from {test_file.name}")
    safe_print(f"    Columns: {list(df.columns)}")
    safe_print(f"    Sample: '{df['sentence'].iloc[0]}'")
    return df


def save_submission(ids: list, translations: list, output_path: Path):
    """Save translations in the required submission format."""
    assert len(ids) == len(translations), (
        f"Mismatch: {len(ids)} IDs vs {len(translations)} translations"
    )

    df = pd.DataFrame({"ID": ids, "kashmiri_text": translations})
    df.to_csv(output_path, index=False, encoding="utf-8")
    safe_print(f"[+] Submission saved to {output_path} ({len(df)} rows)")
    return df


def validate_submission(submission_path: Path, expected_count: int = 1730):
    """Validate the submission file format."""
    df = pd.read_csv(submission_path)
    errors = []

    # Check columns
    required_cols = {"ID", "kashmiri_text"}
    if set(df.columns) != required_cols:
        errors.append(f"Columns should be {required_cols}, got {set(df.columns)}")

    # Check row count
    if len(df) != expected_count:
        errors.append(f"Expected {expected_count} rows, got {len(df)}")

    # Check for missing values
    null_count = df["kashmiri_text"].isna().sum()
    if null_count > 0:
        errors.append(f"{null_count} missing translations found")

    # Check for empty strings
    empty_count = (df["kashmiri_text"].astype(str).str.strip() == "").sum()
    if empty_count > 0:
        errors.append(f"{empty_count} empty translations found")

    # Check IDs are sequential 1..N
    expected_ids = set(range(1, expected_count + 1))
    actual_ids = set(df["ID"].tolist())
    if actual_ids != expected_ids:
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        if missing:
            errors.append(f"Missing IDs: {sorted(missing)[:10]}...")
        if extra:
            errors.append(f"Unexpected IDs: {sorted(extra)[:10]}...")

    if errors:
        safe_print("[-] Submission validation FAILED:")
        for e in errors:
            safe_print(f"    * {e}")
        return False
    else:
        safe_print("[+] Submission validation PASSED")
        safe_print(f"    * {len(df)} rows, all IDs present, no missing values")
        return True

