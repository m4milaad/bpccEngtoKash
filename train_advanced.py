"""
KATHE 2026 — Advanced High-Capacity MT Fine-Tuning Pipeline
Architecture:
  - Supports NLLB-200-1.3B / Distilled-600M and IndicTrans2-1B
  - High-Capacity LoRA (r=64, alpha=128) across ALL linear projections
  - Trainable Language Embeddings (modules_to_save=['embed_tokens', 'lm_head'])
  - Strict Data Quality Filtering (Length ratio + Kashmiri Script Validation)
  - Native Multi-GPU DDP via Accelerate with FP16
"""

import os
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import argparse
import math
import re
import sys
import unicodedata
from pathlib import Path

# Force UTF-8 on Windows consoles
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

from config import BPCC_DIR, MODEL_DIR, SRC_LANG, TGT_LANG
from utils import setup_logging

logger = setup_logging()

# Supported models
MODEL_CONFIGS = {
    "nllb-600m": {
        "name": "facebook/nllb-200-distilled-600M",
        "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        "modules_to_save": ["embed_tokens", "lm_head"],
        "type": "nllb",
    },
    "nllb-1.3b": {
        "name": "facebook/nllb-200-1.3B",
        "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        "modules_to_save": ["embed_tokens", "lm_head"],
        "type": "nllb",
    },
    "indictrans2-1b": {
        "name": "ai4bharat/indictrans2-en-indic-1B",
        "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        "modules_to_save": ["shared", "lm_head"],
        "type": "it2",
    },
}


def clean_kashmiri_text(text: str) -> str:
    """Normalize Kashmiri text during data preparation."""
    text = str(text)
    text = re.sub(r"^(?:kas[@_/\s0-9]*Arab|<2kas_Arab>|__kas_Arab__|\s+)+", "", text, flags=re.IGNORECASE)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("ك", "ک").replace("ي", "ی").replace("ى", "ی")
    return re.sub(r"\s+", " ", text).strip()


def is_valid_kashmiri(text: str) -> bool:
    """Ensure target text contains Arabic/Persian/Kashmiri characters."""
    # Check for Arabic unicode block range
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", text))


class TranslationDataset(Dataset):
    """Tokenized Translation Dataset for dynamic padding."""

    def __init__(self, df: pd.DataFrame, tokenizer, model_type: str, max_source_length=128, max_target_length=128, ip=None):
        self.sources = df["english"].tolist()
        self.targets = df["kashmiri"].tolist()
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.ip = ip

        if model_type == "it2" and ip is not None:
            self.sources = ip.preprocess_batch(self.sources, src_lang="eng_Latn", tgt_lang="kas_Arab")
            self.targets = ip.preprocess_batch(self.targets, src_lang="kas_Arab", tgt_lang="kas_Arab")

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source_enc = self.tokenizer(
            self.sources[idx],
            max_length=self.max_source_length,
            truncation=True,
        )
        target_enc = self.tokenizer(
            text_target=self.targets[idx],
            max_length=self.max_target_length,
            truncation=True,
        )
        return {
            "input_ids": source_enc["input_ids"],
            "attention_mask": source_enc["attention_mask"],
            "labels": target_enc["input_ids"],
        }


def load_curated_data():
    """Load and strictly curate BPCC parallel training data."""
    train_path = BPCC_DIR / "train.csv"
    val_path = BPCC_DIR / "val.csv"

    if not train_path.exists():
        logger.error(f"[-] Training data not found at {train_path}. Run download_data.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path).dropna(subset=["english", "kashmiri"])
    val_df = pd.read_csv(val_path).dropna(subset=["english", "kashmiri"]) if val_path.exists() else None

    # Cleaning & Quality Filters
    logger.info("[*] Applying strict quality filters to training corpus...")
    initial_count = len(train_df)

    train_df["english"] = train_df["english"].astype(str).str.strip()
    train_df["kashmiri"] = train_df["kashmiri"].apply(clean_kashmiri_text)

    # 1. Filter out empty strings
    train_df = train_df[(train_df["english"].str.len() >= 3) & (train_df["kashmiri"].str.len() >= 3)]

    # 2. Filter out non-Arabic script Kashmiri
    train_df = train_df[train_df["kashmiri"].apply(is_valid_kashmiri)]

    # 3. Length-ratio filter (discard extreme length mismatches)
    eng_lens = train_df["english"].str.split().str.len()
    kas_lens = train_df["kashmiri"].str.split().str.len()
    ratio = eng_lens / kas_lens.replace(0, 1)
    train_df = train_df[(ratio >= 0.3) & (ratio <= 3.5)]

    logger.info(f"[+] High-quality curated pairs: {len(train_df):,} / {initial_count:,} ({len(train_df)/initial_count*100:.1f}%)")

    if val_df is not None:
        val_df["english"] = val_df["english"].astype(str).str.strip()
        val_df["kashmiri"] = val_df["kashmiri"].apply(clean_kashmiri_text)
        val_df = val_df[(val_df["english"].str.len() >= 3) & (val_df["kashmiri"].str.len() >= 3)]
        logger.info(f"[+] Validation pairs: {len(val_df):,}")

    return train_df, val_df


def train(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16",
    )
    set_seed(42)

    cfg = MODEL_CONFIGS[args.arch]
    model_name = cfg["name"]
    model_type = cfg["type"]

    if accelerator.is_main_process:
        logger.info("=" * 65)
        logger.info(f"KATHE 2026 -- High-Capacity Architecture Fine-Tuning: {args.arch}")
        logger.info("=" * 65)
        logger.info(f"Base Model:       {model_name}")
        logger.info(f"LoRA Rank:        {args.lora_r} (Alpha: {args.lora_alpha})")
        logger.info(f"Trainable Embeds: {args.train_embeddings}")
        logger.info(f"GPUs Detected:    {accelerator.num_processes}")

    # Tokenizer & IndicProcessor
    ip = None
    if model_type == "it2":
        from IndicTransToolkit import IndicProcessor
        ip = IndicProcessor(inference=False)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=SRC_LANG)
        tokenizer.tgt_lang = TGT_LANG

    # Load Data
    train_df, val_df = load_curated_data()

    train_dataset = TranslationDataset(train_df, tokenizer, model_type, args.max_source_length, args.max_target_length, ip=ip)
    val_dataset = TranslationDataset(val_df, tokenizer, model_type, args.max_source_length, args.max_target_length, ip=ip) if val_df is not None else None

    # Load Base Model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True if model_type == "it2" else False,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )

    # High-Capacity LoRA Config
    modules_to_save = cfg["modules_to_save"] if args.train_embeddings else None

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=cfg["target_modules"],
        modules_to_save=modules_to_save,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(base_model, lora_config)

    # Ensure trainable parameters stay in FP32 for stability
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    if accelerator.is_main_process:
        logger.info(f"[+] Trainable Parameters: {trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M ({100*trainable_params/total_params:.2f}%)")

    # DataLoaders
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    ) if val_dataset is not None else None

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_training_steps = num_update_steps_per_epoch * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Accelerate Prepare
    if val_loader is not None:
        model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, scheduler
        )
    else:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )

    # Training Loop
    save_dir = MODEL_DIR / f"{args.arch}-highcap-best"
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in pbar:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss

                    if args.label_smoothing > 0:
                        logits = outputs.logits
                        vocab_size = logits.size(-1)
                        labels = batch["labels"]
                        smooth_labels = labels.clone()
                        pad_mask = smooth_labels == -100
                        smooth_labels[pad_mask] = 0
                        nll_loss = F.cross_entropy(logits.view(-1, vocab_size), smooth_labels.view(-1), reduction="none", ignore_index=-100)
                        smooth_loss = -F.log_softmax(logits.view(-1, vocab_size), dim=-1).sum(dim=-1)
                        non_pad = (~pad_mask).view(-1).float()
                        nll_loss = (nll_loss * non_pad).sum() / non_pad.sum().clamp(min=1)
                        smooth_loss = (smooth_loss * non_pad).sum() / non_pad.sum().clamp(min=1)
                        loss = (1 - args.label_smoothing) * nll_loss + args.label_smoothing * smooth_loss / vocab_size

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{epoch_loss/num_batches:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        # Validation
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    with accelerator.autocast():
                        out = model(**batch)
                        val_losses.append(accelerator.gather_for_metrics(out.loss).mean().item())
            val_loss = sum(val_losses) / len(val_losses)

            if accelerator.is_main_process:
                logger.info(f"\n[*] Epoch {epoch + 1} Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(model)
                if accelerator.is_main_process:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    unwrapped.save_pretrained(save_dir, save_function=accelerator.save)
                    tokenizer.save_pretrained(save_dir)
                    logger.info(f"    [+] Saved new best high-capacity model -> {save_dir}")
                accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("\n" + "=" * 65)
        logger.info(f"[+] High-Capacity Training Complete! Best Model: {save_dir}")
        logger.info("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 Advanced MT Training")
    parser.add_argument("--arch", type=str, default="nllb-600m", choices=list(MODEL_CONFIGS.keys()), help="Model architecture")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--train_embeddings", action="store_true", default=True, help="Train vocabulary embeddings & LM head")
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
