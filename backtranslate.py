"""
KATHE 2026 — Iterative Back-Translation Pipeline
Generates synthetic parallel data from monolingual Kashmiri to close domain gap.

Pipeline:
1. Collect monolingual Kashmiri (Wikipedia, OSCAR, news)
2. Translate KASHMIRI → ENGLISH (reverse) using best model
3. Back-translate ENGLISH → KASHMIRI and compute chrF++ similarity
4. Filter: Keep pairs where chrF++ > threshold (e.g., 0.65)
5. Add to BPCC training data, retrain
6. Repeat 2-3 cycles
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
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

from config import MODEL_DIR, SRC_LANG, TGT_LANG
from utils import get_device, setup_logging

logger = setup_logging()

# ============================================================
# Monolingual Kashmiri Data Sources
# ============================================================
MONOLINGUAL_SOURCES = {
    "wikipedia": {
        "dataset": "wikipedia",
        "config": "20231101.ks",  # Kashmiri Wikipedia
        "text_column": "text",
        "max_samples": 50000,
    },
    "oscar": {
        "dataset": "oscar-corpus/OSCAR-2301",
        "config": "ks",  # Kashmiri
        "text_column": "text",
        "max_samples": 100000,
    },
    "cc100": {
        "dataset": "cc100",
        "config": "ks",
        "text_column": "text",
        "max_samples": 50000,
    },
}

# Alternative: Local Kashmiri text files
LOCAL_KASHMIRI_PATHS = [
    "data/monolingual_kashmiri.txt",
    "data/kashmiri_wiki.txt",
    "data/kashmiri_news.txt",
]


def clean_kashmiri(text: str) -> str:
    """Clean and normalize Kashmiri text."""
    text = str(text)
    # Remove any model prefixes
    text = re.sub(r"^(?:kas[@_/\s0-9]*Arab|<2kas_Arab>|__kas_Arab__|\s+)+", "", text, flags=re.IGNORECASE)
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
    # Standardize Arabic/Urdu character forms
    text = text.replace("ك", "ک").replace("ي", "ی").replace("ى", "ی")
    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove very short or very long
    if len(text) < 10 or len(text) > 500:
        return ""
    # Must contain Kashmiri/Arabic script characters
    if not re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", text):
        return ""
    return text


def load_monolingual_kashmiri(max_samples: int = 50000, source: str = "wikipedia") -> list:
    """Load monolingual Kashmiri sentences from various sources."""
    sentences = []

    # Try HF datasets first
    if source in MONOLINGUAL_SOURCES:
        src_cfg = MONOLINGUAL_SOURCES[source]
        try:
            logger.info(f"[*] Loading {source} Kashmiri data...")
            ds = load_dataset(src_cfg["dataset"], src_cfg["config"], split="train", streaming=True)
            count = 0
            for item in ds:
                text = clean_kashmiri(item.get(src_cfg["text_column"], ""))
                if text:
                    sentences.append(text)
                    count += 1
                if count >= src_cfg.get("max_samples", max_samples):
                    break
            logger.info(f"    Collected {len(sentences)} sentences from {source}")
        except Exception as e:
            logger.warning(f"    Failed to load {source}: {e}")

    # Try local files
    for path in LOCAL_KASHMIRI_PATHS:
        p = Path(path)
        if p.exists():
            logger.info(f"[*] Loading local file: {path}")
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    text = clean_kashmiri(line)
                    if text:
                        sentences.append(text)
            logger.info(f"    Total now: {len(sentences)}")

    # Deduplicate
    sentences = list(dict.fromkeys(sentences))  # preserves order
    logger.info(f"[+] Total unique monolingual Kashmiri sentences: {len(sentences)}")
    return sentences[:max_samples]


def translate_batch(model, tokenizer, sentences, src_lang, tgt_lang, device, batch_size=16, max_length=128, num_beams=5):
    """Translate a batch of sentences."""
    translations = []
    model.eval()

    for i in tqdm(range(0, len(sentences), batch_size), desc="Translating"):
        batch = sentences[i:i + batch_size]

        # Prepare inputs
        tokenizer.src_lang = src_lang
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translations.extend(decoded)

    return translations


def compute_chrf_similarity(hypothesis: str, reference: str) -> float:
    """Compute chrF++ similarity between two strings."""
    import sacrebleu
    chrf = sacrebleu.corpus_chrf([hypothesis], [[reference]], word_order=2).score
    return chrf / 100.0  # normalize to 0-1


def run_backtranslation_cycle(
    model_path: str,
    monolingual_sentences: list,
    output_path: str,
    batch_size: int = 16,
    num_beams: int = 5,
    max_length: int = 128,
    chrf_threshold: float = 0.65,
    reverse_model_path: str = None,
):
    """
    Run one back-translation cycle:
    1. KASHMIRI → ENGLISH (using reverse_model_path or base model)
    2. ENGLISH → KASHMIRI (using model_path)
    3. Filter by chrF++ similarity between original and back-translated Kashmiri
    """
    device = get_device()

    # Load forward model (English → Kashmiri)
    logger.info(f"[*] Loading forward model: {model_path}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    forward_model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload().to(device).eval()

    # Load reverse model (Kashmiri → English)
    if reverse_model_path and Path(reverse_model_path).exists():
        logger.info(f"[*] Loading reverse model: {reverse_model_path}")
        rev_base = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        reverse_model = PeftModel.from_pretrained(rev_base, reverse_model_path).merge_and_unload().to(device).eval()
    else:
        logger.info("[*] Using base NLLB for reverse direction (Kashmiri → English)")
        reverse_model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600M",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=TGT_LANG, tgt_lang=SRC_LANG)

    # Step 1: KASHMIRI → ENGLISH
    logger.info("[*] Step 1: Translating Kashmiri → English (reverse)...")
    english_sentences = translate_batch(
        reverse_model, tokenizer, monolingual_sentences,
        src_lang=TGT_LANG, tgt_lang=SRC_LANG,
        device=device, batch_size=batch_size, max_length=max_length, num_beams=num_beams
    )

    # Step 2: ENGLISH → KASHMIRI (back-translation)
    logger.info("[*] Step 2: Back-translating English → Kashmiri...")
    backtranslated_kashmiri = translate_batch(
        forward_model, tokenizer, english_sentences,
        src_lang=SRC_LANG, tgt_lang=TGT_LANG,
        device=device, batch_size=batch_size, max_length=max_length, num_beams=num_beams
    )

    # Step 3: Filter by quality
    logger.info("[*] Step 3: Filtering by chrF++ similarity...")
    filtered_pairs = []
    for orig_ks, eng, back_ks in zip(monolingual_sentences, english_sentences, backtranslated_kashmiri):
        orig_clean = clean_kashmiri(orig_ks)
        back_clean = clean_kashmiri(back_ks)

        if not orig_clean or not back_clean:
            continue

        chrf_score = compute_chrf_similarity(back_clean, orig_clean)

        if chrf_score >= chrf_threshold:
            filtered_pairs.append({
                "english": eng.strip(),
                "kashmiri": orig_clean,
                "chrf_score": chrf_score,
            })

    logger.info(f"[+] Kept {len(filtered_pairs)} / {len(monolingual_sentences)} pairs (threshold={chrf_threshold})")

    # Save synthetic parallel data
    df_synthetic = pd.DataFrame(filtered_pairs)
    df_synthetic.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"[+] Saved synthetic parallel data to: {output_path}")

    return df_synthetic


def merge_with_bpcc(synthetic_path: str, bpcc_train_path: str, output_path: str, max_synthetic: int = None):
    """Merge synthetic data with original BPCC training data."""
    logger.info("[*] Merging synthetic data with BPCC...")

    bpcc_df = pd.read_csv(bpcc_train_path)
    synth_df = pd.read_csv(synthetic_path)

    if max_synthetic and len(synth_df) > max_synthetic:
        # Keep highest quality synthetic pairs
        synth_df = synth_df.nlargest(max_synthetic, "chrf_score")

    # Combine
    merged = pd.concat([bpcc_df, synth_df[["english", "kashmiri"]]], ignore_index=True)
    merged = merged.drop_duplicates(subset=["english", "kashmiri"])
    merged = merged.dropna(subset=["english", "kashmiri"])

    merged.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"[+] Merged dataset: {len(merged)} pairs (BPCC: {len(bpcc_df)} + Synthetic: {len(synth_df)})")
    logger.info(f"    Saved to: {output_path}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 Back-Translation Pipeline")
    parser.add_argument("--model_path", type=str, default="models/best", help="Fine-tuned model (En→Ks)")
    parser.add_argument("--reverse_model_path", type=str, default=None, help="Fine-tuned reverse model (Ks→En)")
    parser.add_argument("--monolingual_source", type=str, default="wikipedia", choices=["wikipedia", "oscar", "cc100", "all"])
    parser.add_argument("--max_monolingual", type=int, default=30000, help="Max monolingual sentences to process")
    parser.add_argument("--output_synthetic", type=str, default="data/synthetic_parallel.csv")
    parser.add_argument("--output_merged", type=str, default="data/bpcc/train_augmented.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--chrf_threshold", type=float, default=0.65, help="chrF++ threshold for filtering")
    parser.add_argument("--max_synthetic", type=int, default=20000, help="Max synthetic pairs to add")
    parser.add_argument("--cycle", type=int, default=1, help="Cycle number (for iterative training)")

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info(f"KATHE 2026 — BACK-TRANSLATION CYCLE {args.cycle}")
    logger.info("=" * 70)

    # Load monolingual Kashmiri
    if args.monolingual_source == "all":
        all_sentences = []
        for src in ["wikipedia", "oscar", "cc100"]:
            all_sentences.extend(load_monolingual_kashmiri(args.max_monolingual // 3, src))
        monolingual = list(dict.fromkeys(all_sentences))[:args.max_monolingual]
    else:
        monolingual = load_monolingual_kashmiri(args.max_monolingual, args.monolingual_source)

    if len(monolingual) == 0:
        logger.error("[-] No monolingual data found!")
        sys.exit(1)

    # Run back-translation
    synthetic_df = run_backtranslation_cycle(
        model_path=args.model_path,
        monolingual_sentences=monolingual,
        output_path=args.output_synthetic,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_length=args.max_length,
        chrf_threshold=args.chrf_threshold,
        reverse_model_path=args.reverse_model_path,
    )

    # Merge with BPCC
    bpcc_train = Path("data/bpcc/train.csv")
    if bpcc_train.exists():
        merge_with_bpcc(
            args.output_synthetic,
            str(bpcc_train),
            args.output_merged,
            max_synthetic=args.max_synthetic,
        )
    else:
        logger.warning("[-] BPCC train.csv not found, skipping merge")

    logger.info("\n[+] Back-translation cycle complete!")
    logger.info(f"    Next: Retrain on {args.output_merged}")
    logger.info(f"    python train_optimized.py --epochs 3 --batch_size 8 --gradient_accumulation_steps 4")


if __name__ == "__main__":
    main()