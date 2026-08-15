"""
KATHE 2026 — IndicTrans2 Fine-Tuning Pipeline
English-to-Kashmiri using ai4bharat/indictrans2-en-indic-1B with LoRA

This model was specifically designed for Indic languages including Kashmiri
and should outperform NLLB-600M significantly.

Usage:
    python train_indictrans2.py [--epochs 5] [--batch_size 4]
"""

import argparse
import gc
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup,
)
import torch.nn.functional as F

from config import BPCC_DIR, MODEL_DIR, TRAIN_CONFIG
from utils import get_device, setup_logging

logger = setup_logging()

# IndicTrans2 specific config
IT2_MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
IT2_SRC_LANG = "eng_Latn"
IT2_TGT_LANG = "kas_Arab"
IT2_MODEL_DIR = MODEL_DIR / "indictrans2-best"


class IT2TranslationDataset(Dataset):
    """Dataset for IndicTrans2 with IndicProcessor preprocessing."""

    def __init__(self, data, tokenizer, ip, max_source_length=128, max_target_length=128):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.ip = ip
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # Preprocess all source and target sentences
        logger.info("[*] Preprocessing sentences with IndicProcessor...")
        self.sources = ip.preprocess_batch(
            data["english"].tolist(), src_lang=IT2_SRC_LANG, tgt_lang=IT2_TGT_LANG
        )
        self.targets = ip.preprocess_batch(
            data["kashmiri"].tolist(), src_lang=IT2_TGT_LANG, tgt_lang=IT2_TGT_LANG
        )
        logger.info(f"    Preprocessed {len(self.sources)} pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source = self.sources[idx]
        target = self.targets[idx]

        source_enc = self.tokenizer(
            source, max_length=self.max_source_length, truncation=True
        )
        target_enc = self.tokenizer(
            text_target=target, max_length=self.max_target_length, truncation=True
        )

        return {
            "input_ids": source_enc["input_ids"],
            "attention_mask": source_enc["attention_mask"],
            "labels": target_enc["input_ids"],
        }


def load_training_data():
    """Load prepared BPCC training data."""
    train_path = BPCC_DIR / "train.csv"
    val_path = BPCC_DIR / "val.csv"

    if not train_path.exists():
        logger.error(f"[-] Training data not found at {train_path}. Run download_data.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if val_path.exists() else None

    # Clean data
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


def setup_model(device):
    """Load IndicTrans2 model with LoRA adapters."""
    logger.info(f"[*] Loading IndicTrans2: {IT2_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        IT2_MODEL_NAME, trust_remote_code=True
    )

    # Load model - use sdpa attention for speed
    model = AutoModelForSeq2SeqLM.from_pretrained(
        IT2_MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    # LoRA config — same proven settings as NLLB but adapted for IndicTrans2
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model = model.to(device)

    # Keep LoRA weights in fp32 for optimizer stability
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"    Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M ({100 * trainable / total:.2f}%)")

    return model, tokenizer


def evaluate_model(model, val_loader, device):
    """Run validation loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda") if device.type == "cuda" else torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs.loss.item()
                num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    model.train()
    return avg_loss


def train(args):
    """Main training loop for IndicTrans2."""
    logger.info("=" * 60)
    logger.info("KATHE 2026 -- IndicTrans2 Fine-Tuning")
    logger.info("=" * 60)

    device = get_device()
    train_data, val_data = load_training_data()
    model, tokenizer = setup_model(device)

    # Load IndicProcessor for preprocessing
    try:
        from IndicTransToolkit import IndicProcessor
        ip = IndicProcessor(inference=False)
        logger.info("[+] IndicProcessor loaded (training mode)")
    except ImportError:
        logger.error("[-] IndicTransToolkit not installed! Run: pip install indictranstoolkit")
        sys.exit(1)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True,
        pad_to_multiple_of=8, label_pad_token_id=-100,
    )

    # Datasets
    train_dataset = IT2TranslationDataset(train_data, tokenizer, ip,
        max_source_length=args.max_source_length, max_target_length=args.max_target_length)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=data_collator, num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = None
    if val_data is not None:
        val_dataset = IT2TranslationDataset(val_data, tokenizer, ip,
            max_source_length=args.max_source_length, max_target_length=args.max_target_length)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size * 2, shuffle=False,
            collate_fn=data_collator, num_workers=0,
            pin_memory=True if device.type == "cuda" else False,
        )

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    total_optimizer_steps = (len(train_loader) * args.epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimizer_steps,
    )
    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None
    label_smoothing = args.label_smoothing

    logger.info(f"\n[*] IndicTrans2 Training Config:")
    logger.info(f"    Model:                   {IT2_MODEL_NAME}")
    logger.info(f"    Total Dataset:           {len(train_data):,} pairs")
    logger.info(f"    Epochs:                  {args.epochs}")
    logger.info(f"    Micro Batch Size:        {args.batch_size}")
    logger.info(f"    Grad Accumulation:       {args.gradient_accumulation_steps}")
    logger.info(f"    Effective Batch Size:    {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"    Optimizer Updates:        {total_optimizer_steps:,}")
    logger.info(f"    Warmup Steps:            {warmup_steps:,}")
    logger.info(f"    Label Smoothing:         {label_smoothing}")
    logger.info(f"    LR Schedule:             Cosine ({args.learning_rate})")

    best_val_loss = float("inf")
    opt_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    if label_smoothing > 0:
                        logits = outputs.logits
                        vocab_size = logits.size(-1)
                        smooth_labels = labels.clone()
                        pad_mask = smooth_labels == -100
                        smooth_labels[pad_mask] = 0
                        nll_loss = F.cross_entropy(
                            logits.view(-1, vocab_size), smooth_labels.view(-1),
                            reduction='none', ignore_index=-100,
                        )
                        smooth_loss = -F.log_softmax(logits.view(-1, vocab_size), dim=-1).sum(dim=-1)
                        non_pad = (~pad_mask).view(-1).float()
                        nll_loss = (nll_loss * non_pad).sum() / non_pad.sum().clamp(min=1)
                        smooth_loss = (smooth_loss * non_pad).sum() / non_pad.sum().clamp(min=1)
                        loss = (1 - label_smoothing) * nll_loss + label_smoothing * smooth_loss / vocab_size
                    else:
                        loss = outputs.loss
                    loss = loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    opt_step += 1
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
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
        if val_loader:
            logger.info(f"\n[*] Validation epoch {epoch + 1}...")
            val_loss = evaluate_model(model, val_loader, device)
            logger.info(f"    Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                IT2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(IT2_MODEL_DIR)
                tokenizer.save_pretrained(IT2_MODEL_DIR)
                logger.info(f"    [+] New best model -> {IT2_MODEL_DIR}")

        # Save epoch checkpoint
        epoch_dir = MODEL_DIR / f"indictrans2-epoch-{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        logger.info(f"    [+] Epoch checkpoint -> {epoch_dir}")

    logger.info("\n" + "=" * 60)
    logger.info("[+] IndicTrans2 Training Complete!")
    logger.info(f"    Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"    Best Model:    {IT2_MODEL_DIR}")
    logger.info("=" * 60)
    logger.info(f"\nNext: python inference_indictrans2.py")


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 - IndicTrans2 Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=TRAIN_CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--learning_rate", type=float, default=TRAIN_CONFIG["learning_rate"])
    parser.add_argument("--warmup_ratio", type=float, default=TRAIN_CONFIG["warmup_ratio"])
    parser.add_argument("--weight_decay", type=float, default=TRAIN_CONFIG["weight_decay"])
    parser.add_argument("--max_source_length", type=int, default=TRAIN_CONFIG["max_source_length"])
    parser.add_argument("--max_target_length", type=int, default=TRAIN_CONFIG["max_target_length"])
    parser.add_argument("--fp16", action="store_true", default=TRAIN_CONFIG["fp16"])
    parser.add_argument("--label_smoothing", type=float, default=TRAIN_CONFIG.get("label_smoothing", 0.1))
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
