"""
KATHE 2026 — Multi-Model Ensemble Inference (MBR Decoding)
Combines predictions from multiple checkpoints/models at test time.

Usage:
    python ensemble_inference.py \
        --models models/best models/checkpoint-epoch-3 models/checkpoint-epoch-5 \
        --test_file englishdev.csv \
        --output outputs/submission_ensemble.csv \
        --num_beams 8 --n_candidates 10
"""

import argparse
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
import torch
import unicodedata
import re
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import sacrebleu

from config import SRC_LANG, TGT_LANG
from utils import get_device, setup_logging

logger = setup_logging()

NLLB_BASE = "facebook/nllb-200-distilled-600M"
IT2_BASE = "ai4bharat/indictrans2-en-indic-1B"
IT2_SRC = "eng_Latn"
IT2_TGT = "kas_Arab"


def clean_kashmiri(text: str) -> str:
    text = str(text)
    text = re.sub(r"^(?:kas[@_/\s0-9]*Arab|<2kas_Arab>|__kas_Arab__|\s+)+", "", text, flags=re.IGNORECASE)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ك", "ک").replace("ي", "ی").replace("ى", "ی")
    return re.sub(r"\s+", " ", text).strip()


def load_model(model_path: str, device: torch.device, model_type: str = "nllb"):
    """Load a fine-tuned model (NLLB or IndicTrans2)."""
    if model_type == "nllb":
        base = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_BASE,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        model = PeftModel.from_pretrained(base, model_path).merge_and_unload().to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(NLLB_BASE, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
        src_lang, tgt_lang = SRC_LANG, TGT_LANG
    else:  # indictrans2
        from IndicTransToolkit import IndicProcessor
        base = AutoModelForSeq2SeqLM.from_pretrained(
            IT2_BASE,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        model = PeftModel.from_pretrained(base, model_path).merge_and_unload().to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(IT2_BASE, trust_remote_code=True)
        src_lang, tgt_lang = IT2_SRC, IT2_TGT

    return model, tokenizer, src_lang, tgt_lang


def generate_candidates(model, tokenizer, sentences, src_lang, tgt_lang, device, batch_size, num_beams, n_candidates, max_length=128, model_type="nllb"):
    """Generate N-best candidates from a model."""
    model.eval()
    all_candidates = []

    for i in tqdm(range(0, len(sentences), batch_size), desc=f"Generating ({model_type})"):
        batch = sentences[i:i + batch_size]

        if model_type == "nllb":
            tokenizer.src_lang = src_lang
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=n_candidates,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        else:  # indictrans2
            from IndicTransToolkit import IndicProcessor
            ip = IndicProcessor(inference=True)
            preprocessed = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
            inputs = tokenizer(preprocessed, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=n_candidates,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            raw = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded = ip.postprocess_batch(raw, lang=tgt_lang)

        # Group by input sentence
        for j in range(len(batch)):
            cands = []
            for k in range(n_candidates):
                idx = j * n_candidates + k
                if idx < len(decoded):
                    cands.append(clean_kashmiri(decoded[idx]))
            all_candidates.append(cands)

    return all_candidates


def mbr_select(candidates_pool: list) -> str:
    """
    Minimum Bayes Risk selection using chrF++ as utility.
    Returns the candidate with highest expected chrF++ against the pool.
    """
    if not candidates_pool:
        return ""

    # Flatten all candidates
    all_cands = []
    for model_cands in candidates_pool:
        all_cands.extend(model_cands)

    # Deduplicate
    unique_cands = list(dict.fromkeys(all_cands))
    if len(unique_cands) == 1:
        return unique_cands[0]

    # Score each candidate by average chrF++ against all others
    best_score = -1
    best_cand = unique_cands[0]

    for cand in unique_cands:
        total_chrf = 0.0
        count = 0
        for other in unique_cands:
            if cand != other:
                chrf = sacrebleu.corpus_chrf([cand], [[other]], word_order=2).score
                total_chrf += chrf
                count += 1
        avg_chrf = total_chrf / count if count > 0 else 0
        if avg_chrf > best_score:
            best_score = avg_chrf
            best_cand = cand

    return best_cand


def run_ensemble_inference(
    model_paths: list,
    model_types: list,
    test_file: str,
    output_file: str,
    batch_size: int = 8,
    num_beams: int = 8,
    n_candidates: int = 5,
    max_length: int = 128,
):
    device = get_device()

    # Load test data
    df_test = pd.read_csv(test_file)
    sentences = df_test["sentence"].astype(str).str.strip().tolist()
    logger.info(f"[+] Loaded {len(sentences)} test sentences")

    # Generate candidates from each model
    all_model_candidates = []
    for model_path, model_type in zip(model_paths, model_types):
        logger.info(f"[*] Loading {model_type}: {model_path}")
        model, tokenizer, src_lang, tgt_lang = load_model(model_path, device, model_type)

        candidates = generate_candidates(
            model, tokenizer, sentences,
            src_lang, tgt_lang, device,
            batch_size, num_beams, n_candidates, max_length, model_type
        )
        all_model_candidates.append(candidates)

        # Clear GPU memory
        del model
        torch.cuda.empty_cache()

    # MBR selection per sentence
    logger.info("[*] Running MBR selection...")
    final_translations = []
    for i in tqdm(range(len(sentences)), desc="MBR decoding"):
        # Pool candidates from all models for this sentence
        sentence_pool = [model_cands[i] for model_cands in all_model_candidates]
        best = mbr_select(sentence_pool)
        final_translations.append(best)

    # Save submission
    df_out = pd.DataFrame({"ID": df_test["ID"], "kashmiri_text": final_translations})
    df_out.to_csv(output_file, index=False, encoding="utf-8")
    logger.info(f"[+] Ensemble submission saved: {output_file}")

    # Preview
    logger.info("\nSample translations:")
    for i in range(min(5, len(df_out))):
        logger.info(f"  [{i+1}] {df_out.loc[i, 'kashmiri_text']}")

    return df_out


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 Multi-Model Ensemble Inference")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to fine-tuned models")
    parser.add_argument("--model_types", nargs="+", default=None, help="Types: nllb or indictrans2 (default: auto-detect)")
    parser.add_argument("--test_file", type=str, default="englishdev.csv")
    parser.add_argument("--output", type=str, default="outputs/submission_ensemble.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--n_candidates", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    # Auto-detect model types if not provided
    if args.model_types is None:
        model_types = []
        for p in args.models:
            if "indictrans" in p.lower() or "it2" in p.lower():
                model_types.append("indictrans2")
            else:
                model_types.append("nllb")
    else:
        model_types = args.model_types

    logger.info("=" * 70)
    logger.info("KATHE 2026 — MULTI-MODEL ENSEMBLE INFERENCE")
    logger.info("=" * 70)
    for p, t in zip(args.models, model_types):
        logger.info(f"  Model: {p} (type: {t})")
    logger.info(f"  Candidates per model: {args.n_candidates}")
    logger.info(f"  Beams: {args.num_beams}")
    logger.info("=" * 70)

    run_ensemble_inference(
        model_paths=args.models,
        model_types=model_types,
        test_file=args.test_file,
        output_file=args.output,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        n_candidates=args.n_candidates,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()