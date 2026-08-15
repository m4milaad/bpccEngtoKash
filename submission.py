"""
KATHE 2026 — Submission Formatter & Uploader

Usage:
    python submission.py                              # Validate existing submission
    python submission.py --upload                      # Upload to Kaggle
    python submission.py --input outputs/submission.csv --upload
"""

import argparse
import sys
from pathlib import Path

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd

from config import SUBMISSION_FILE, KAGGLE_COMPETITION, TEST_FILE
from utils import setup_logging, validate_submission

logger = setup_logging()


def format_submission(input_path: Path, output_path: Path):
    """Ensure submission is in the correct format."""
    df = pd.read_csv(input_path)

    # Ensure correct columns
    if "kashmiri_text" not in df.columns:
        for col in ["kashmiri", "translation", "target", "tgt", "output"]:
            if col in df.columns:
                df = df.rename(columns={col: "kashmiri_text"})
                logger.info(f"    Renamed column '{col}' -> 'kashmiri_text'")
                break

    # Ensure ID column
    if "ID" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "ID"})
        else:
            df["ID"] = range(1, len(df) + 1)

    # Select only required columns
    submission = df[["ID", "kashmiri_text"]].copy()

    # Clean translations
    submission["kashmiri_text"] = (
        submission["kashmiri_text"]
        .astype(str)
        .str.strip()
        .replace("nan", "")
    )

    # Sort by ID
    submission = submission.sort_values("ID").reset_index(drop=True)

    # Save
    submission.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"[+] Formatted submission saved to {output_path}")

    return submission


def upload_to_kaggle(submission_path: Path, message: str = "Automated submission"):
    """Upload submission to Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        logger.info(f"[*] Uploading submission to Kaggle: {KAGGLE_COMPETITION}")
        api.competition_submit(
            file_name=str(submission_path),
            message=message,
            competition=KAGGLE_COMPETITION,
        )
        logger.info("[+] Submission uploaded successfully!")
        logger.info(
            f"    Check your score at: https://kaggle.com/competitions/{KAGGLE_COMPETITION}/leaderboard"
        )

    except ImportError:
        logger.error("[-] kaggle package not installed. Run: pip install kaggle")
    except Exception as e:
        logger.error(f"[-] Upload failed: {e}")
        logger.info(
            "    You can upload manually at:\n"
            f"    https://kaggle.com/competitions/{KAGGLE_COMPETITION}/submit"
        )


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 -- Submission Manager")
    parser.add_argument(
        "--input",
        type=str,
        default=str(SUBMISSION_FILE),
        help="Path to submission CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SUBMISSION_FILE),
        help="Path for formatted submission",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to Kaggle after validation",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="KATHE 2026 submission",
        help="Submission message for Kaggle",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("KATHE 2026 -- Submission Manager")
    logger.info("=" * 60)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(
            f"[-] Submission file not found: {input_path}\n"
            "    Run 'python inference.py' first to generate translations."
        )
        return

    test_df = pd.read_csv(TEST_FILE)
    expected_count = len(test_df)

    format_submission(input_path, output_path)
    is_valid = validate_submission(output_path, expected_count=expected_count)

    if is_valid and args.upload:
        upload_to_kaggle(output_path, message=args.message)
    elif not is_valid:
        logger.error("[-] Submission validation failed -- fix errors before uploading")
    elif not args.upload:
        logger.info(
            "\n[+] To upload to Kaggle, run:\n"
            f"    python submission.py --input {output_path} --upload"
        )


if __name__ == "__main__":
    main()

