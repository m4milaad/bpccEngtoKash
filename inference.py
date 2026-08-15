"""
KATHE 2026 — Inference Script
Translate English sentences to Kashmiri using a pretrained model.

Supports both:
  - Zero-shot inference with pretrained NLLB/IndicTrans2
  - Inference with a fine-tuned model checkpoint

Usage:
    python inference.py                          # Zero-shot with pretrained model
    python inference.py --model_path models/best # Use fine-tuned model
    python inference.py --model_name facebook/nllb-200-distilled-600M
"""

import argparse
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import (
    MODEL_NAME,
    SRC_LANG,
    TGT_LANG,
    TEST_FILE,
    OUTPUT_DIR,
    INFERENCE_CONFIG,
    SUBMISSION_FILE,
)
from utils import setup_logging, get_device, load_test_data, save_submission, validate_submission

logger = setup_logging()


def load_model(model_name: str, model_path: str = None, device: torch.device = None):
    """
    Load model and tokenizer.
    
    If model_path is given, loads a fine-tuned checkpoint.
    Otherwise loads the pretrained model from HuggingFace.
    """
    load_from = model_path if model_path else model_name
    logger.info(f"[+] Loading model: {load_from}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        src_lang=SRC_LANG,
    )

    if model_path:
        # Load fine-tuned model
        try:
            from peft import PeftModel
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device and device.type == "cuda" else torch.float32,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
            logger.info("[+] Loaded fine-tuned model with LoRA adapter merged")
        except (ImportError, Exception):
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device and device.type == "cuda" else torch.float32,
            )
            logger.info("[+] Loaded fine-tuned model directly")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device and device.type == "cuda" else torch.float32,
        )
        logger.info("[+] Loaded pretrained model (zero-shot mode)")

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"    Parameters: {total_params / 1e6:.1f}M")

    return model, tokenizer


def translate_batch(
    model,
    tokenizer,
    sentences: list[str],
    device: torch.device,
    tgt_lang: str = TGT_LANG,
    max_length: int = INFERENCE_CONFIG["max_length"],
    num_beams: int = INFERENCE_CONFIG["num_beams"],
    **kwargs,
) -> list[str]:
    """Translate a batch of English sentences to Kashmiri."""
    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=kwargs.get("length_penalty", INFERENCE_CONFIG["length_penalty"]),
            no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", INFERENCE_CONFIG["no_repeat_ngram_size"]),
            early_stopping=kwargs.get("early_stopping", INFERENCE_CONFIG["early_stopping"]),
        )

    translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return translations


def run_inference(
    model,
    tokenizer,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int = INFERENCE_CONFIG["batch_size"],
) -> list[str]:
    """Run inference on all test sentences."""
    sentences = df["sentence"].tolist()
    all_translations = []

    logger.info(f"[+] Translating {len(sentences)} sentences (batch_size={batch_size})...")

    for i in tqdm(range(0, len(sentences), batch_size), desc="Translating"):
        batch = sentences[i : i + batch_size]
        translations = translate_batch(model, tokenizer, batch, device)
        all_translations.extend(translations)

        if (i + batch_size) % 100 < batch_size:
            logger.info(f"    Progress: {min(i + batch_size, len(sentences))}/{len(sentences)}")

    return all_translations


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 — English to Kashmiri Translation")
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help=f"Pretrained model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned model checkpoint (optional)",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=str(TEST_FILE),
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SUBMISSION_FILE),
        help="Path to output submission CSV",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=INFERENCE_CONFIG["batch_size"],
        help="Inference batch size",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=INFERENCE_CONFIG["num_beams"],
        help="Number of beams for beam search",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("KATHE 2026 -- English to Kashmiri Translation")
    logger.info("=" * 60)

    device = get_device()
    model, tokenizer = load_model(args.model_name, args.model_path, device)
    test_df = load_test_data(Path(args.test_file))

    start_time = time.time()
    translations = run_inference(model, tokenizer, test_df, device, args.batch_size)
    elapsed = time.time() - start_time

    logger.info(f"[+] Translation completed in {elapsed:.1f}s ({elapsed/len(translations):.2f}s/sentence)")

    logger.info("\n[*] Sample translations:")
    for i in range(min(5, len(translations))):
        logger.info(f"    EN: {test_df['sentence'].iloc[i]}")
        try:
            logger.info(f"    KS: {translations[i]}")
        except Exception:
            pass
        logger.info("")

    output_path = Path(args.output)
    save_submission(test_df["ID"].tolist(), translations, output_path)
    validate_submission(output_path, expected_count=len(test_df))

    logger.info("\n[+] Done!")
    logger.info(f"    Submission file: {output_path}")


if __name__ == "__main__":
    main()

