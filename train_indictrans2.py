"""
KATHE 2026 — IndicTrans2 Fine-Tuning Pipeline (Multi-GPU / Kaggle 2x T4 edition)
English-to-Kashmiri using ai4bharat/indictrans2-en-indic-1B with LoRA

This runs as a true multi-process DDP job across both Kaggle T4 GPUs via
HuggingFace Accelerate. On a machine with only one GPU (or CPU) it falls
back to normal single-process training automatically -- no code changes
needed either way.

Usage (Kaggle notebook, 2x T4):
    !pip install -q accelerate
    !NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch \
        --config_file accelerate_config.yaml \
        train_indictrans2.py --epochs 5 --batch_size 8 --gradient_accumulation_steps 2

Usage (single GPU / local, unchanged):
    python train_indictrans2.py --epochs 5 --batch_size 4
"""

import os

# Kaggle's virtualized 2x T4 instances don't reliably expose GPU-to-GPU P2P /
# InfiniBand access. NCCL's default probing for these can hang training
# entirely on Kaggle -- force it off before torch/NCCL initializes. This is
# a no-op (and harmless) on single-GPU runs or elsewhere.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import argparse
import pickle
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup,
)

from config import BPCC_DIR, MODEL_DIR, TRAIN_CONFIG
from utils import setup_logging

logger = setup_logging()

# IndicTrans2 specific config
IT2_MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
IT2_SRC_LANG = "eng_Latn"
IT2_TGT_LANG = "kas_Arab"
IT2_MODEL_DIR = MODEL_DIR / "indictrans2-best"


class IT2TranslationDataset(Dataset):
    """Dataset over already IndicProcessor-preprocessed sentence pairs."""

    def __init__(self, sources, targets, tokenizer, max_source_length=128, max_target_length=128):
        assert len(sources) == len(targets), "sources/targets length mismatch"
        self.sources = sources
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source_enc = self.tokenizer(
            self.sources[idx], max_length=self.max_source_length, truncation=True
        )
        target_enc = self.tokenizer(
            text_target=self.targets[idx], max_length=self.max_target_length, truncation=True
        )
        return {
            "input_ids": source_enc["input_ids"],
            "attention_mask": source_enc["attention_mask"],
            "labels": target_enc["input_ids"],
        }


def _preprocess_or_load_cache(data, ip, cache_path, accelerator, name):
    """Preprocess with IndicProcessor once (on the main process only) and let
    every other GPU process just load the cached result. Without this, each
    of the 2 T4 processes would redo the same CPU-bound preprocessing pass
    over the whole dataset, wasting Kaggle session time for no benefit."""
    cache_path = Path(cache_path)

    need_rebuild = True
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            need_rebuild = len(cached["sources"]) != len(data)
        except Exception:
            need_rebuild = True

    if accelerator.is_main_process and need_rebuild:
        logger.info(f"[*] Preprocessing {len(data)} {name} pairs with IndicProcessor...")
        sources = ip.preprocess_batch(data["english"].tolist(), src_lang=IT2_SRC_LANG, tgt_lang=IT2_TGT_LANG)
        targets = ip.preprocess_batch(data["kashmiri"].tolist(), src_lang=IT2_TGT_LANG, tgt_lang=IT2_TGT_LANG)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"sources": sources, "targets": targets}, f)

    accelerator.wait_for_everyone()
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    return cached["sources"], cached["targets"]


def load_training_data():
    """Load prepared BPCC training data."""
    train_path = BPCC_DIR / "train.csv"
    val_path = BPCC_DIR / "val.csv"

    if not train_path.exists():
        logger.error(f"[-] Training data not found at {train_path}. Run download_data.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if val_path.exists() else None

    train_df = train_df.dropna(subset=["english", "kashmiri"])
    train_df["english"] = train_df["english"].astype(str).str.strip()
    train_df["kashmiri"] = train_df["kashmiri"].astype(str).str.strip()
    train_df = train_df[(train_df["english"].str.len() > 0) & (train_df["kashmiri"].str.len() > 0)]

    if val_df is not None:
        val_df = val_df.dropna(subset=["english", "kashmiri"])
        val_df["english"] = val_df["english"].astype(str).str.strip()
        val_df["kashmiri"] = val_df["kashmiri"].astype(str).str.strip()
        val_df = val_df[(val_df["english"].str.len() > 0) & (val_df["kashmiri"].str.len() > 0)]

    logger.info(f"[+] Loaded {len(train_df)} training pairs")
    if val_df is not None:
        logger.info(f"[+] Loaded {len(val_df)} validation pairs")

    return train_df, val_df


def setup_model():
    """Load IndicTrans2 + LoRA. Deliberately no .to(device) here --
    accelerator.prepare() takes care of placement (and DDP wrapping)."""
    logger.info(f"[*] Loading IndicTrans2: {IT2_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(IT2_MODEL_NAME, trust_remote_code=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        IT2_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # T4 (Turing) has no bf16 tensor cores -- fp16 is correct here
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)

    # Keep LoRA weights in fp32 for optimizer stability; frozen base stays fp16.
    # accelerator.autocast() below reconciles the mixed dtypes during forward.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"    Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M ({100 * trainable / total:.2f}%)")

    return model, tokenizer


def evaluate_model(model, val_loader, accelerator):
    """Validation loss, correctly averaged across all GPU processes."""
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            with accelerator.autocast():
                outputs = model(**batch)
            bs = batch["input_ids"].shape[0]
            losses.append(accelerator.gather_for_metrics(outputs.loss.repeat(bs)))
    losses = torch.cat(losses)
    avg_loss = losses.mean().item()
    model.train()
    return avg_loss


def train(args):
    """Main training loop for IndicTrans2 (single or multi-GPU via Accelerate)."""
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if args.fp16 else "no",
    )
    set_seed(TRAIN_CONFIG.get("seed", 42))

    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("KATHE 2026 -- IndicTrans2 Fine-Tuning (Accelerate)")
        logger.info("=" * 60)
        logger.info(f"[*] GPU processes in use: {accelerator.num_processes}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"    GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    train_data, val_data = load_training_data()
    model, tokenizer = setup_model()

    try:
        from IndicTransToolkit import IndicProcessor
        ip = IndicProcessor(inference=False)
    except ImportError:
        logger.error("[-] IndicTransToolkit not installed! Run: pip install indictranstoolkit")
        sys.exit(1)

    train_sources, train_targets = _preprocess_or_load_cache(
        train_data, ip, BPCC_DIR / "it2_train_preprocessed.pkl", accelerator, "train"
    )
    train_dataset = IT2TranslationDataset(
        train_sources, train_targets, tokenizer,
        max_source_length=args.max_source_length, max_target_length=args.max_target_length,
    )

    val_dataset = None
    if val_data is not None:
        val_sources, val_targets = _preprocess_or_load_cache(
            val_data, ip, BPCC_DIR / "it2_val_preprocessed.pkl", accelerator, "val"
        )
        val_dataset = IT2TranslationDataset(
            val_sources, val_targets, tokenizer,
            max_source_length=args.max_source_length, max_target_length=args.max_target_length,
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True,
        pad_to_multiple_of=8, label_pad_token_id=-100,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=data_collator, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size * 2, shuffle=False,
            collate_fn=data_collator, num_workers=args.num_workers, pin_memory=True,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Prepare AFTER building the dataloaders -- accelerate shards them across
    # GPUs with a DistributedSampler under the hood (each of the 2 T4s sees
    # its own half of every epoch).
    if val_loader is not None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # len(train_loader) here is already the per-process (sharded) length, so
    # this total is correct without dividing by num_processes again.
    total_optimizer_steps = (len(train_loader) * args.epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimizer_steps,
    )
    scheduler = accelerator.prepare(scheduler)

    label_smoothing = args.label_smoothing
    effective_batch = args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes

    if accelerator.is_main_process:
        logger.info(f"\n[*] IndicTrans2 Training Config:")
        logger.info(f"    Model:                   {IT2_MODEL_NAME}")
        logger.info(f"    Total Dataset:           {len(train_data):,} pairs")
        logger.info(f"    GPUs:                    {accelerator.num_processes}")
        logger.info(f"    Epochs:                  {args.epochs}")
        logger.info(f"    Micro Batch Size/GPU:    {args.batch_size}")
        logger.info(f"    Grad Accumulation:       {args.gradient_accumulation_steps}")
        logger.info(f"    Effective Batch Size:    {effective_batch}")
        logger.info(f"    Optimizer Updates:       {total_optimizer_steps:,}")
        logger.info(f"    Warmup Steps:            {warmup_steps:,}")
        logger.info(f"    Label Smoothing:         {label_smoothing}")
        logger.info(f"    LR Schedule:             Cosine ({args.learning_rate})")

    best_val_loss = float("inf")
    opt_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in pbar:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                    if label_smoothing > 0:
                        logits = outputs.logits
                        vocab_size = logits.size(-1)
                        labels = batch["labels"]
                        smooth_labels = labels.clone()
                        pad_mask = smooth_labels == -100
                        smooth_labels[pad_mask] = 0
                        nll_loss = F.cross_entropy(
                            logits.view(-1, vocab_size), smooth_labels.view(-1),
                            reduction="none", ignore_index=-100,
                        )
                        smooth_loss = -F.log_softmax(logits.view(-1, vocab_size), dim=-1).sum(dim=-1)
                        non_pad = (~pad_mask).view(-1).float()
                        nll_loss = (nll_loss * non_pad).sum() / non_pad.sum().clamp(min=1)
                        smooth_loss = (smooth_loss * non_pad).sum() / non_pad.sum().clamp(min=1)
                        loss = (1 - label_smoothing) * nll_loss + label_smoothing * smooth_loss / vocab_size
                    else:
                        loss = outputs.loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                opt_step += 1

            epoch_loss += outputs.loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{outputs.loss.item():.4f}",
                "avg": f"{epoch_loss / num_batches:.4f}",
                "step": f"{opt_step}/{total_optimizer_steps}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # Validation
        if val_loader is not None:
            if accelerator.is_main_process:
                logger.info(f"\n[*] Validation epoch {epoch + 1}...")
            val_loss = evaluate_model(model, val_loader, accelerator)
            if accelerator.is_main_process:
                logger.info(f"    Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(model)
                if accelerator.is_main_process:
                    IT2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    unwrapped.save_pretrained(IT2_MODEL_DIR, save_function=accelerator.save)
                    tokenizer.save_pretrained(IT2_MODEL_DIR)
                    logger.info(f"    [+] New best model -> {IT2_MODEL_DIR}")
                accelerator.wait_for_everyone()

        # Save epoch checkpoint
        accelerator.wait_for_everyone()
        unwrapped = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            epoch_dir = MODEL_DIR / f"indictrans2-epoch-{epoch + 1}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(epoch_dir, save_function=accelerator.save)
            tokenizer.save_pretrained(epoch_dir)
            logger.info(f"    [+] Epoch checkpoint -> {epoch_dir}")
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("\n" + "=" * 60)
        logger.info("[+] IndicTrans2 Training Complete!")
        logger.info(f"    Best Val Loss: {best_val_loss:.4f}")
        logger.info(f"    Best Model:    {IT2_MODEL_DIR}")
        logger.info("=" * 60)
        logger.info(f"\nNext: python inference_indictrans2.py")


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 - IndicTrans2 Fine-Tuning (multi-GPU)")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument(
        "--batch_size", type=int, default=TRAIN_CONFIG["batch_size"],
        help="Micro batch size PER GPU. Effective batch = batch_size * grad_accum * num_gpus.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=TRAIN_CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--learning_rate", type=float, default=TRAIN_CONFIG["learning_rate"])
    parser.add_argument("--warmup_ratio", type=float, default=TRAIN_CONFIG["warmup_ratio"])
    parser.add_argument("--weight_decay", type=float, default=TRAIN_CONFIG["weight_decay"])
    parser.add_argument("--max_source_length", type=int, default=TRAIN_CONFIG["max_source_length"])
    parser.add_argument("--max_target_length", type=int, default=TRAIN_CONFIG["max_target_length"])
    parser.add_argument("--fp16", action="store_true", default=TRAIN_CONFIG["fp16"])
    parser.add_argument("--label_smoothing", type=float, default=TRAIN_CONFIG.get("label_smoothing", 0.1))
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader worker processes (per GPU process)")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
