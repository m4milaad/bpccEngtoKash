"""
KATHE 2026 — IndicTrans2 Inference Script (multi-GPU aware)
Translate English sentences to Kashmiri using fine-tuned IndicTrans2.

On a machine with 2+ GPUs (e.g. Kaggle's 2x T4), this automatically splits
the test set across all visible GPUs and runs independent inference
processes in parallel for a near-linear speedup. This does NOT need
accelerate/DDP -- inference has no gradients to sync, so each GPU just
translates its own slice of sentences with its own model copy.

Usage (Kaggle, 2x T4 -- auto-detected and used):
    !python inference_indictrans2.py --model_path models/indictrans2-best

Force single-GPU (e.g. to debug):
    python inference_indictrans2.py --single_gpu

IMPORTANT: run this as a script (`python inference_indictrans2.py`), not by
pasting the code into a notebook cell directly -- the multi-GPU path uses
torch.multiprocessing.spawn, which needs this file to be importable as a
normal module by the child processes.
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

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import TEST_FILE, INFERENCE_CONFIG, SUBMISSION_FILE, MODEL_DIR
from utils import setup_logging, get_device, load_test_data, save_submission, validate_submission

logger = setup_logging()

IT2_MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
IT2_SRC_LANG = "eng_Latn"
IT2_TGT_LANG = "kas_Arab"
IT2_MODEL_DIR = MODEL_DIR / "indictrans2-best"


def load_model(model_path: str, device: torch.device):
    """Load IndicTrans2 model with optional fine-tuned LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(IT2_MODEL_NAME, trust_remote_code=True)

    if model_path:
        try:
            from peft import PeftModel
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                IT2_MODEL_NAME, trust_remote_code=True,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
            logger.info(f"[+] Loaded fine-tuned IndicTrans2 with LoRA merged from {model_path}")
        except Exception as e:
            logger.info(f"[!] Could not load as LoRA adapter ({e}), trying direct load...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
            logger.info("[+] Loaded fine-tuned model directly")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            IT2_MODEL_NAME, trust_remote_code=True,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        logger.info("[+] Loaded pretrained IndicTrans2 (zero-shot mode)")

    model = model.to(device)
    model.eval()
    return model, tokenizer


def translate_batch(
    model, tokenizer, ip, sentences: list, device: torch.device,
    max_length: int = INFERENCE_CONFIG["max_length"],
    num_beams: int = INFERENCE_CONFIG["num_beams"],
    **kwargs,
) -> list:
    """Translate a batch of English sentences to Kashmiri using IndicTrans2."""
    preprocessed = ip.preprocess_batch(sentences, src_lang=IT2_SRC_LANG, tgt_lang=IT2_TGT_LANG)

    inputs = tokenizer(
        preprocessed, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
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
    return ip.postprocess_batch(raw_translations, lang=IT2_TGT_LANG)


def run_inference(
    model, tokenizer, ip, df: pd.DataFrame, device: torch.device,
    batch_size: int = INFERENCE_CONFIG["batch_size"],
    num_beams: int = INFERENCE_CONFIG["num_beams"],
    desc: str = "Translating",
) -> list:
    """Run inference on all sentences in df."""
    sentences = df["sentence"].tolist()
    all_translations = []

    for i in tqdm(range(0, len(sentences), batch_size), desc=desc):
        batch = sentences[i: i + batch_size]
        all_translations.extend(
            translate_batch(model, tokenizer, ip, batch, device, num_beams=num_beams)
        )

    return all_translations


def _worker(rank, world_size, model_path, test_file, batch_size, num_beams, tmp_dir):
    """One process per GPU. Translates only this rank's slice of the test set,
    then writes a partial CSV for the main process to stitch back together."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    try:
        from IndicTransToolkit import IndicProcessor
    except ImportError:
        print("[-] IndicTransToolkit not installed! Run: pip install indictranstoolkit")
        sys.exit(1)
    ip = IndicProcessor(inference=True)

    model, tokenizer = load_model(model_path, device)
    test_df = load_test_data(Path(test_file))

    chunks = np.array_split(np.arange(len(test_df)), world_size)
    idx = chunks[rank]
    chunk_df = test_df.iloc[idx].reset_index(drop=True)

    translations = run_inference(
        model, tokenizer, ip, chunk_df, device, batch_size, num_beams, desc=f"GPU{rank}"
    )

    out = pd.DataFrame({"ID": chunk_df["ID"].tolist(), "kashmiri_text": translations})
    out.to_csv(Path(tmp_dir) / f"_partial_rank{rank}.csv", index=False, encoding="utf-8")


def run_multi_gpu_inference(args, world_size: int) -> pd.DataFrame:
    """Spawn one process per GPU, each translating a slice of the test set,
    then merge the partial outputs back into ID order."""
    tmp_dir = Path(args.output).parent
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[+] Splitting inference across {world_size} GPUs")
    mp.spawn(
        _worker,
        args=(world_size, args.model_path, args.test_file, args.batch_size, args.num_beams, str(tmp_dir)),
        nprocs=world_size,
        join=True,
    )

    parts = []
    for rank in range(world_size):
        part_path = tmp_dir / f"_partial_rank{rank}.csv"
        parts.append(pd.read_csv(part_path))
        part_path.unlink()

    combined = pd.concat(parts, ignore_index=True).sort_values("ID").reset_index(drop=True)
    return combined


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 — IndicTrans2 Inference")
    parser.add_argument(
        "--model_path", type=str, default=str(IT2_MODEL_DIR),
        help="Path to fine-tuned IndicTrans2 checkpoint",
    )
    parser.add_argument("--test_file", type=str, default=str(TEST_FILE))
    parser.add_argument("--output", type=str, default=str(SUBMISSION_FILE))
    parser.add_argument("--batch_size", type=int, default=INFERENCE_CONFIG["batch_size"])
    parser.add_argument("--num_beams", type=int, default=INFERENCE_CONFIG["num_beams"])
    parser.add_argument(
        "--single_gpu", action="store_true",
        help="Force single-GPU inference even if more than one GPU is visible",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("KATHE 2026 -- IndicTrans2 English->Kashmiri")
    logger.info("=" * 60)

    n_gpus = torch.cuda.device_count()
    start_time = time.time()

    if not args.single_gpu and n_gpus > 1:
        logger.info(f"[+] Detected {n_gpus} GPUs -- running parallel inference")
        result_df = run_multi_gpu_inference(args, n_gpus)
        ids = result_df["ID"].tolist()
        translations = result_df["kashmiri_text"].tolist()
    else:
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
        translations = run_inference(
            model, tokenizer, ip, test_df, device, args.batch_size, args.num_beams
        )
        ids = test_df["ID"].tolist()

    elapsed = time.time() - start_time
    logger.info(f"[+] Translation completed in {elapsed:.1f}s ({elapsed/len(translations):.2f}s/sentence)")

    logger.info("\n[*] Sample translations:")
    for i in range(min(5, len(translations))):
        try:
            logger.info(f"    ID {ids[i]}: {translations[i]}")
        except Exception:
            pass

    output_path = Path(args.output)
    save_submission(ids, translations, output_path)
    validate_submission(output_path, expected_count=len(ids))

    logger.info("\n[+] Done!")
    logger.info(f"    Submission file: {output_path}")


if __name__ == "__main__":
    main()
