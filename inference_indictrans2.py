"""
KATHE 2026 — IndicTrans2 Inference Script
Translate English sentences to Kashmiri using fine-tuned IndicTrans2.

Usage:
    python inference_indictrans2.py                                    # Use fine-tuned model
    python inference_indictrans2.py --model_path models/indictrans2-best
"""

import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import TEST_FILE, OUTPUT_DIR, INFERENCE_CONFIG, SUBMISSION_FILE, MODEL_DIR
from utils import setup_logging, get_device, load_test_data, save_submission, validate_submission

logger = setup_logging()

IT2_MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
IT2_SRC_LANG = "eng_Latn"
IT2_TGT_LANG = "kas_Arab"
IT2_MODEL_DIR = MODEL_DIR / "indictrans2-best"


def load_model(model_path: str = None, device: torch.device = None):
    """Load IndicTrans2 model with optional fine-tuned LoRA adapter."""
    load_from = model_path if model_path else str(IT2_MODEL_DIR)
    logger.info(f"[+] Loading IndicTrans2 model")

    tokenizer = AutoTokenizer.from_pretrained(
        IT2_MODEL_NAME, trust_remote_code=True
    )

    if model_path:
        try:
            from peft import PeftModel
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                IT2_MODEL_NAME,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device and device.type == "cuda" else torch.float32,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
            logger.info(f"[+] Loaded fine-tuned IndicTrans2 with LoRA merged from {model_path}")
        except (ImportError, Exception) as e:
            logger.info(f"[!] Could not load as LoRA adapter ({e}), trying direct load...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=torch.float16 if device and device.type == "cuda" else torch.float32,
            )
            logger.info("[+] Loaded fine-tuned model directly")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            IT2_MODEL_NAME,
            trust_remote_code=True,
            dtype=torch.float16 if device and device.type == "cuda" else torch.float32,
        )
        logger.info("[+] Loaded pretrained IndicTrans2 (zero-shot mode)")

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"    Parameters: {total_params / 1e6:.1f}M")

    return model, tokenizer


def translate_batch(
    model, tokenizer, ip, sentences: list[str], device: torch.device,
    max_length: int = INFERENCE_CONFIG["max_length"],
    num_beams: int = INFERENCE_CONFIG["num_beams"],
    **kwargs,
) -> list[str]:
    """Translate a batch of English sentences to Kashmiri using IndicTrans2."""
    # Preprocess with IndicProcessor
    preprocessed = ip.preprocess_batch(sentences, src_lang=IT2_SRC_LANG, tgt_lang=IT2_TGT_LANG)

    inputs = tokenizer(
        preprocessed,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=kwargs.get("length_penalty", INFERENCE_CONFIG["length_penalty"]),
            no_repeat_ngram_size=kwargs.get("no_repeat_ngram_size", INFERENCE_CONFIG["no_repeat_ngram_size"]),
            early_stopping=kwargs.get("early_stopping", INFERENCE_CONFIG["early_stopping"]),
            use_cache=True,
        )

    raw_translations = tokenizer.batch_decode(generated, skip_special_tokens=True)

    # Postprocess with IndicProcessor
    translations = ip.postprocess_batch(raw_translations, lang=IT2_TGT_LANG)

    return translations


def run_inference(
    model, tokenizer, ip, df: pd.DataFrame, device: torch.device,
    batch_size: int = INFERENCE_CONFIG["batch_size"],
) -> list[str]:
    """Run inference on all test sentences."""
    sentences = df["sentence"].tolist()
    all_translations = []

    logger.info(f"[+] Translating {len(sentences)} sentences (batch_size={batch_size})...")

    for i in tqdm(range(0, len(sentences), batch_size), desc="Translating"):
        batch = sentences[i : i + batch_size]
        translations = translate_batch(model, tokenizer, ip, batch, device)
        all_translations.extend(translations)

        if (i + batch_size) % 100 < batch_size:
            logger.info(f"    Progress: {min(i + batch_size, len(sentences))}/{len(sentences)}")

    return all_translations


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 — IndicTrans2 Inference")
    parser.add_argument(
        "--model_path", type=str, default=str(IT2_MODEL_DIR),
        help="Path to fine-tuned IndicTrans2 checkpoint",
    )
    parser.add_argument(
        "--test_file", type=str, default=str(TEST_FILE),
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--output", type=str, default=str(SUBMISSION_FILE),
        help="Path to output submission CSV",
    )
    parser.add_argument(
        "--batch_size", type=int, default=INFERENCE_CONFIG["batch_size"],
        help="Inference batch size",
    )
    parser.add_argument(
        "--num_beams", type=int, default=INFERENCE_CONFIG["num_beams"],
        help="Number of beams for beam search",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("KATHE 2026 -- IndicTrans2 English->Kashmiri")
    logger.info("=" * 60)

    # Load IndicProcessor
    try:
        from IndicTransToolkit import IndicProcessor
        ip = IndicProcessor(inference=True)
        logger.info("[+] IndicProcessor loaded (inference mode)")
    except ImportError:
        logger.error("[-] IndicTransToolkit not installed! Run: pip install indictranstoolkit")
        sys.exit(1)

    device = get_device()
    model, tokenizer = load_model(args.model_path, device)
    test_df = load_test_data(Path(args.test_file))

    start_time = time.time()
    translations = run_inference(model, tokenizer, ip, test_df, device, args.batch_size)
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
