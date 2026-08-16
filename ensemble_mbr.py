"""
KATHE 2026 — MBR Reranking Ensemble (Kaggle 2x T4 edition)
Combine IndicTrans2 + NLLB via Minimum Bayes Risk (MBR) decoding.

How it works:
  1. Generate N-best candidate translations from each model independently
  2. Pool all candidates for each source sentence
  3. Score every candidate against every other candidate using chrF++
  4. The candidate with the highest average chrF++ wins (MBR criterion)

On Kaggle 2x T4:  NLLB sits on GPU 0, IndicTrans2 sits on GPU 1.
Both stay loaded simultaneously — no load/unload overhead.
On single GPU: falls back to sequential load/generate/free/load/generate.

Usage (Kaggle 2x T4):
    !python ensemble_mbr.py

Usage (single GPU):
    python ensemble_mbr.py

Self-reranking (one model only):
    python ensemble_mbr.py --models nllb
    python ensemble_mbr.py --models indictrans2
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import sacrebleu
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import (
    MODEL_DIR,
    TEST_FILE,
    OUTPUT_DIR,
    SRC_LANG,
    TGT_LANG,
)
from utils import setup_logging, load_test_data, save_submission, validate_submission

logger = setup_logging()

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
NLLB_BASE = "facebook/nllb-200-distilled-600M"
NLLB_ADAPTER = MODEL_DIR / "best"

IT2_BASE = "ai4bharat/indictrans2-en-indic-1B"
IT2_ADAPTER = MODEL_DIR / "indictrans2-best"
IT2_SRC_LANG = "eng_Latn"
IT2_TGT_LANG = "kas_Arab"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_nllb(device: torch.device):
    """Load fine-tuned NLLB-200 with LoRA adapter merged."""
    logger.info(f"[*] Loading NLLB-200-distilled-600M + LoRA -> {device}")
    tokenizer = AutoTokenizer.from_pretrained(NLLB_BASE, src_lang=SRC_LANG)

    from peft import PeftModel

    base = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_BASE,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, str(NLLB_ADAPTER))
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()

    tgt_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    logger.info(f"    NLLB ready on {device}. Target: {TGT_LANG} -> id {tgt_lang_id}")
    return model, tokenizer, tgt_lang_id


def load_indictrans2(device: torch.device):
    """Load fine-tuned IndicTrans2 with LoRA adapter merged."""
    logger.info(f"[*] Loading IndicTrans2-en-indic-1B + LoRA -> {device}")
    tokenizer = AutoTokenizer.from_pretrained(IT2_BASE, trust_remote_code=True)

    from peft import PeftModel

    base = AutoModelForSeq2SeqLM.from_pretrained(
        IT2_BASE,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, str(IT2_ADAPTER))
    model = model.merge_and_unload()
    model = model.to(device)
    model.eval()

    logger.info(f"    IndicTrans2 ready on {device}.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_nllb_candidates(
    model, tokenizer, tgt_lang_id, sentences, device,
    n_candidates=5, num_beams=10, max_length=128,
):
    """Generate N-best translations from NLLB for a batch of sentences."""
    inputs = tokenizer(
        sentences, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=n_candidates,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
            use_cache=True,
        )

    all_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Reshape: flat list -> list of lists (one per input sentence)
    candidates_per_sentence = []
    for i in range(len(sentences)):
        start = i * n_candidates
        end = start + n_candidates
        candidates_per_sentence.append(all_translations[start:end])

    return candidates_per_sentence


def generate_it2_candidates(
    model, tokenizer, ip, sentences, device,
    n_candidates=5, num_beams=10, max_length=128,
):
    """Generate N-best translations from IndicTrans2 for a batch of sentences."""
    preprocessed = ip.preprocess_batch(
        sentences, src_lang=IT2_SRC_LANG, tgt_lang=IT2_TGT_LANG,
    )

    inputs = tokenizer(
        preprocessed, return_tensors="pt", padding=True, truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=n_candidates,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
            use_cache=True,
        )

    raw_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Postprocess in bulk, then reshape
    postprocessed = ip.postprocess_batch(raw_translations, lang=IT2_TGT_LANG)

    candidates_per_sentence = []
    for i in range(len(sentences)):
        start = i * n_candidates
        end = start + n_candidates
        candidates_per_sentence.append(postprocessed[start:end])

    return candidates_per_sentence


# ---------------------------------------------------------------------------
# MBR Reranking
# ---------------------------------------------------------------------------

def mbr_select(candidates: list[str]) -> str:
    """Select the candidate with the highest average chrF++ against all others.

    This is Minimum Bayes Risk decoding: we pick the hypothesis that maximises
    the expected utility (chrF++ score) under the model's implicit distribution
    over translations (approximated by the N-best list).

    For a pool of K candidates, this is O(K^2) chrF++ evaluations per sentence,
    which is fast since chrF++ is a lightweight string metric.
    """
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    # Deduplicate while preserving order (duplicates add no information to MBR)
    seen = set()
    unique = []
    for c in candidates:
        c_stripped = c.strip()
        if c_stripped and c_stripped not in seen:
            seen.add(c_stripped)
            unique.append(c_stripped)

    if len(unique) == 0:
        return candidates[0]
    if len(unique) == 1:
        return unique[0]

    # Compute pairwise chrF++ and average
    best_score = -1.0
    best_candidate = unique[0]

    for i, hyp in enumerate(unique):
        # Score this hypothesis against every OTHER candidate as pseudo-reference
        others = [unique[j] for j in range(len(unique)) if j != i]
        score = sacrebleu.corpus_chrf([hyp] * len(others), [others], word_order=2).score
        if score > best_score:
            best_score = score
            best_candidate = hyp

    return best_candidate


# ---------------------------------------------------------------------------
# GPU assignment
# ---------------------------------------------------------------------------

def assign_devices(use_nllb: bool, use_it2: bool):
    """Decide which GPU each model goes on.

    2 GPUs + 2 models -> one each (parallel, no memory contention).
    1 GPU  + 2 models -> both on cuda:0, sequential with memory freeing.
    """
    n_gpus = torch.cuda.device_count()

    if n_gpus == 0:
        return torch.device("cpu"), torch.device("cpu"), False

    if use_nllb and use_it2 and n_gpus >= 2:
        # Best case: one model per GPU
        return torch.device("cuda:0"), torch.device("cuda:1"), True
    else:
        # Single GPU: sequential
        return torch.device("cuda:0"), torch.device("cuda:0"), False


# ---------------------------------------------------------------------------
# Batch-level candidate generation (full dataset)
# ---------------------------------------------------------------------------

def generate_all_nllb(
    nllb_model, nllb_tok, nllb_tgt_id, sentences, device,
    batch_size, n_candidates, num_beams, max_length,
):
    """Generate NLLB candidates for all sentences."""
    all_candidates = [[] for _ in range(len(sentences))]
    for i in tqdm(range(0, len(sentences), batch_size), desc="NLLB candidates"):
        batch = sentences[i: i + batch_size]
        batch_cands = generate_nllb_candidates(
            nllb_model, nllb_tok, nllb_tgt_id, batch, device,
            n_candidates=n_candidates, num_beams=num_beams, max_length=max_length,
        )
        for j, cands in enumerate(batch_cands):
            all_candidates[i + j] = cands
    return all_candidates


def generate_all_it2(
    it2_model, it2_tok, ip, sentences, device,
    batch_size, n_candidates, num_beams, max_length,
):
    """Generate IndicTrans2 candidates for all sentences."""
    all_candidates = [[] for _ in range(len(sentences))]
    for i in tqdm(range(0, len(sentences), batch_size), desc="IT2 candidates"):
        batch = sentences[i: i + batch_size]
        batch_cands = generate_it2_candidates(
            it2_model, it2_tok, ip, batch, device,
            n_candidates=n_candidates, num_beams=num_beams, max_length=max_length,
        )
        for j, cands in enumerate(batch_cands):
            all_candidates[i + j] = cands
    return all_candidates


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_ensemble(args):
    use_nllb = "nllb" in args.models
    use_it2 = "indictrans2" in args.models

    nllb_device, it2_device, parallel = assign_devices(use_nllb, use_it2)

    n_gpus = torch.cuda.device_count()
    logger.info(f"[*] GPUs available: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"    GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    if parallel:
        logger.info(f"[+] Parallel mode: NLLB -> {nllb_device}, IT2 -> {it2_device}")
    elif use_nllb and use_it2:
        logger.info(f"[!] Single GPU: sequential mode (load/free/load)")
    else:
        logger.info(f"[*] Single model mode")

    # Load models
    nllb_model = nllb_tok = nllb_tgt_id = None
    it2_model = it2_tok = ip = None

    if use_nllb:
        nllb_model, nllb_tok, nllb_tgt_id = load_nllb(nllb_device)

    if use_it2:
        try:
            from IndicTransToolkit import IndicProcessor
            ip = IndicProcessor(inference=True)
        except ImportError:
            logger.error("[-] IndicTransToolkit not installed — skipping IndicTrans2.")
            use_it2 = False

        if use_it2:
            it2_model, it2_tok = load_indictrans2(it2_device)

    if not use_nllb and not use_it2:
        logger.error("[-] No models available. Exiting.")
        sys.exit(1)

    # Load test data
    test_df = load_test_data(Path(args.test_file))
    sentences = test_df["sentence"].tolist()
    ids = test_df["ID"].tolist()

    logger.info(f"\n[*] MBR Ensemble Config:")
    logger.info(f"    Models:          {args.models}")
    logger.info(f"    Parallel GPUs:   {parallel}")
    logger.info(f"    N candidates:    {args.n_candidates} per model")
    logger.info(f"    Num beams:       {args.num_beams}")
    logger.info(f"    Batch size:      {args.batch_size}")
    logger.info(f"    Total sentences: {len(sentences)}")
    if use_nllb and use_it2:
        logger.info(f"    Total pool/sent: {args.n_candidates * 2} candidates")
    else:
        logger.info(f"    Total pool/sent: {args.n_candidates} candidates (self-reranking)")

    all_nllb_candidates = [[] for _ in range(len(sentences))]
    all_it2_candidates = [[] for _ in range(len(sentences))]

    gen_kwargs = dict(
        batch_size=args.batch_size,
        n_candidates=args.n_candidates,
        num_beams=args.num_beams,
        max_length=args.max_length,
    )

    if parallel and use_nllb and use_it2:
        # ---- 2x T4: generate from both models simultaneously ----
        logger.info("\n[*] Generating candidates in parallel across 2 GPUs...")

        # Python threads release the GIL during CUDA kernels, so true
        # parallelism happens on the GPU side even with the GIL.
        with ThreadPoolExecutor(max_workers=2) as executor:
            nllb_future = executor.submit(
                generate_all_nllb,
                nllb_model, nllb_tok, nllb_tgt_id, sentences, nllb_device,
                **gen_kwargs,
            )
            it2_future = executor.submit(
                generate_all_it2,
                it2_model, it2_tok, ip, sentences, it2_device,
                **gen_kwargs,
            )
            all_nllb_candidates = nllb_future.result()
            all_it2_candidates = it2_future.result()

    else:
        # ---- Single GPU: sequential with optional memory freeing ----
        if use_nllb:
            logger.info("\n[*] Generating NLLB candidates...")
            all_nllb_candidates = generate_all_nllb(
                nllb_model, nllb_tok, nllb_tgt_id, sentences, nllb_device,
                **gen_kwargs,
            )
            if use_it2 and not parallel:
                logger.info("    Freeing NLLB from GPU memory...")
                del nllb_model
                torch.cuda.empty_cache()

        if use_it2:
            logger.info("\n[*] Generating IndicTrans2 candidates...")
            all_it2_candidates = generate_all_it2(
                it2_model, it2_tok, ip, sentences, it2_device,
                **gen_kwargs,
            )
            del it2_model
            torch.cuda.empty_cache()

    # --- MBR selection ---
    logger.info("\n[*] Running MBR reranking (chrF++ utility)...")
    translations = []
    for i in tqdm(range(len(sentences)), desc="MBR select"):
        pool = all_nllb_candidates[i] + all_it2_candidates[i]
        if not pool:
            translations.append("")
            continue
        best = mbr_select(pool)
        translations.append(best)

    return ids, translations


def main():
    parser = argparse.ArgumentParser(
        description="KATHE 2026 — MBR Reranking Ensemble (Kaggle 2x T4 edition)",
    )
    parser.add_argument(
        "--models", nargs="+", default=["nllb", "indictrans2"],
        choices=["nllb", "indictrans2"],
        help="Which models to use for candidate generation",
    )
    parser.add_argument("--test_file", type=str, default=str(TEST_FILE))
    parser.add_argument(
        "--output", type=str,
        default=str(OUTPUT_DIR / "submission_ensemble.csv"),
    )
    parser.add_argument(
        "--n_candidates", type=int, default=5,
        help="Number of candidate translations per model (total pool = n * num_models)",
    )
    parser.add_argument(
        "--num_beams", type=int, default=10,
        help="Beam width for generation (must be >= n_candidates)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    # num_beams must be >= n_candidates for num_return_sequences to work
    if args.num_beams < args.n_candidates:
        args.num_beams = args.n_candidates
        logger.info(f"[!] Bumped num_beams to {args.num_beams} (must be >= n_candidates)")

    logger.info("=" * 60)
    logger.info("KATHE 2026 -- MBR Reranking Ensemble")
    logger.info("=" * 60)

    start_time = time.time()
    ids, translations = run_ensemble(args)
    elapsed = time.time() - start_time

    logger.info(f"\n[+] MBR ensemble completed in {elapsed:.1f}s "
                f"({elapsed / len(translations):.2f}s/sentence)")

    # Show samples
    logger.info("\n[*] Sample MBR-selected translations:")
    for i in range(min(5, len(translations))):
        try:
            logger.info(f"    ID {ids[i]}: {translations[i]}")
        except Exception:
            pass

    # Save
    output_path = Path(args.output)
    save_submission(ids, translations, output_path)
    validate_submission(output_path, expected_count=len(ids))

    logger.info(f"\n[+] Done! Submission: {output_path}")
    logger.info("    Next: python evaluate.py --predictions " + str(output_path))


if __name__ == "__main__":
    main()
