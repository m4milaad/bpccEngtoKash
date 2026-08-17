"""
KATHE 2026 — Test Set Self-Evaluation (No Ground Truth)
Uses model confidence, self-consistency, and linguistic heuristics to predict test quality.

Run before submitting to estimate Kaggle score.
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

from config import SRC_LANG, TGT_LANG, BPCC_DIR
from utils import get_device, setup_logging

logger = setup_logging()

NLLB_BASE = "facebook/nllb-200-distilled-600M"


def clean_kashmiri(text: str) -> str:
    text = str(text)
    text = re.sub(r"^(?:kas[@_/\s0-9]*Arab|<2kas_Arab>|__kas_Arab__|\s+)+", "", text, flags=re.IGNORECASE)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ك", "ک").replace("ي", "ی").replace("ى", "ی")
    return re.sub(r"\s+", " ", text).strip()


def translate_with_confidence(model, tokenizer, sentences, src_lang, tgt_lang, device, batch_size=16, num_beams=5, max_length=128, return_sequences=5):
    """Generate N-best translations with confidence scores."""
    model.eval()
    all_results = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Generating candidates"):
        batch = sentences[i:i + batch_size]

        tokenizer.src_lang = src_lang
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=return_sequences,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode sequences
        sequences = outputs.sequences.view(len(batch), return_sequences, -1)
        scores = outputs.sequences_scores  # log probs

        for b_idx in range(len(batch)):
            candidates = []
            for s_idx in range(return_sequences):
                seq = sequences[b_idx, s_idx]
                decoded = tokenizer.decode(seq, skip_special_tokens=True)
                cleaned = clean_kashmiri(decoded)
                logprob = scores[b_idx * return_sequences + s_idx].item() if scores is not None else 0.0
                candidates.append((cleaned, logprob))
            all_results.append(candidates)

    return all_results


def self_consistency_score(candidates: list) -> float:
    """Compute self-consistency (chrF++ agreement) among top candidates."""
    if len(candidates) < 2:
        return 0.0

    texts = [c[0] for c in candidates if c[0]]
    if len(texts) < 2:
        return 0.0

    # Pairwise chrF++
    total_score = 0.0
    pairs = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            chrf = sacrebleu.corpus_chrf([texts[i]], [[texts[j]]], word_order=2).score
            total_score += chrf
            pairs += 1

    return (total_score / pairs) / 100.0 if pairs > 0 else 0.0


def linguistic_quality_score(text: str, english: str) -> float:
    """Heuristic linguistic quality checks."""
    score = 1.0
    text_clean = clean_kashmiri(text)

    # Penalty: Urdu pronoun leaks
    if re.search(r"\bہم\b", text_clean):
        score -= 0.15
    if re.search(r"\bاور\b", text_clean):
        score -= 0.1
    if re.search(r"\bلیکن\b", text_clean):
        score -= 0.1
    if re.search(r"\bکیونکہ\b", text_clean):
        score -= 0.1

    # Penalty: Missing Kashmiri script
    if not re.search(r"[\u0600-\u06FF]", text_clean):
        score -= 0.3

    # Penalty: Space before punctuation
    if re.search(r"\s+[۔،؛؟]", text_clean):
        score -= 0.05

    # Bonus: Kashmiri-specific characters
    kashmiri_chars = len(re.findall(r"[ؠٕٗٚٛ]", text_clean))
    score += min(kashmiri_chars * 0.02, 0.1)

    # Gender agreement heuristic (feminine)
    fem_eng = re.search(r"\b(she|her|hers|woman|girl|mother|sister|queen)\b", english, re.I)
    fem_ks = re.search(r"\b(سۄ|ٲس|چھی|کٔرمٕژ)\b", text_clean)
    masc_ks = re.search(r"\b(سہٕ|اوس|چھُ|کٔرمٕت)\b", text_clean)
    if fem_eng and masc_ks and not fem_ks:
        score -= 0.2
    if fem_eng and fem_ks:
        score += 0.1

    return max(0.0, min(1.0, score))


def predict_test_quality(model_path: str, test_file: str, output_file: str, batch_size=16, num_beams=6, num_candidates=5):
    """Run self-evaluation on test set."""
    device = get_device()

    logger.info(f"[*] Loading model: {model_path}")
    base = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_BASE,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, model_path).merge_and_unload().to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(NLLB_BASE, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

    # Load test data
    df_test = pd.read_csv(test_file)
    english_sentences = df_test["sentence"].astype(str).str.strip().tolist()
    logger.info(f"[+] Loaded {len(english_sentences)} test sentences")

    # Generate candidates with confidence
    logger.info("[*] Generating N-best candidates...")
    all_candidates = translate_with_confidence(
        model, tokenizer, english_sentences,
        src_lang=SRC_LANG, tgt_lang=TGT_LANG,
        device=device, batch_size=batch_size,
        num_beams=num_beams, max_length=128,
        return_sequences=num_candidates
    )

    # Analyze each sentence
    results = []
    for idx, (eng, cands) in enumerate(zip(english_sentences, all_candidates)):
        if not cands:
            results.append({
                "ID": df_test.loc[idx, "ID"],
                "english": eng,
                "prediction": "",
                "self_consistency": 0.0,
                "linguistic_quality": 0.0,
                "model_confidence": 0.0,
                "composite_score": 0.0,
            })
            continue

        best_text, best_logprob = cands[0]
        consistency = self_consistency_score(cands)
        ling_quality = linguistic_quality_score(best_text, eng)
        model_conf = min(1.0, max(0.0, (best_logprob + 10) / 10))  # normalize logprob

        # Weighted composite
        composite = 0.4 * consistency + 0.3 * ling_quality + 0.3 * model_conf

        results.append({
            "ID": df_test.loc[idx, "ID"],
            "english": eng,
            "prediction": best_text,
            "candidate_2": cands[1][0] if len(cands) > 1 else "",
            "candidate_3": cands[2][0] if len(cands) > 2 else "",
            "self_consistency": round(consistency, 4),
            "linguistic_quality": round(ling_quality, 4),
            "model_confidence": round(model_conf, 4),
            "composite_score": round(composite, 4),
        })

    df_results = pd.DataFrame(results)

    # Summary stats
    logger.info("\n" + "=" * 70)
    logger.info("TEST SET SELF-EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Sentences evaluated:     {len(df_results)}")
    logger.info(f"Avg self-consistency:    {df_results['self_consistency'].mean():.4f}")
    logger.info(f"Avg linguistic quality:  {df_results['linguistic_quality'].mean():.4f}")
    logger.info(f"Avg model confidence:    {df_results['model_confidence'].mean():.4f}")
    logger.info(f"Avg composite score:     {df_results['composite_score'].mean():.4f}")
    logger.info(f"Low quality (<0.4):      {(df_results['composite_score'] < 0.4).sum()} ({(df_results['composite_score'] < 0.4).mean()*100:.1f}%)")
    logger.info(f"High quality (>0.7):     {(df_results['composite_score'] > 0.7).sum()} ({(df_results['composite_score'] > 0.7).mean()*100:.1f}%)")
    logger.info("=" * 70)

    # Correlation with validation (if available)
    val_path = BPCC_DIR / "val.csv"
    if val_path.exists():
        logger.info("[*] Computing correlation with validation set performance...")
        # This would require running eval on val set too

    # Save detailed results
    df_results.to_csv(output_file, index=False, encoding="utf-8")
    logger.info(f"\n[+] Detailed results saved to: {output_file}")

    # Also save clean submission
    df_submission = df_results[["ID", "prediction"]].copy()
    df_submission.columns = ["ID", "kashmiri_text"]
    sub_path = output_file.replace(".csv", "_submission.csv")
    df_submission.to_csv(sub_path, index=False, encoding="utf-8")
    logger.info(f"[+] Ready-to-submit file: {sub_path}")

    return df_results


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 Test Set Self-Evaluation")
    parser.add_argument("--model_path", type=str, default="models/best", help="Fine-tuned model path")
    parser.add_argument("--test_file", type=str, default="englishdev.csv", help="Test file (no ground truth)")
    parser.add_argument("--output", type=str, default="outputs/test_self_eval.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=6)
    parser.add_argument("--num_candidates", type=int, default=5)
    args = parser.parse_args()

    predict_test_quality(
        model_path=args.model_path,
        test_file=args.test_file,
        output_file=args.output,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        num_candidates=args.num_candidates,
    )


if __name__ == "__main__":
    main()