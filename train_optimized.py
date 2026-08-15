"""
KATHE 2026 — Optimized LoRA Fine-Tuning Pipeline
English-to-Kashmiri Machine Translation on RTX 4060 GPU

Key Optimizations:
  1. Dynamic Batch Padding via DataCollatorForSeq2Seq (2x-3x speedup vs fixed 128 padding)
  2. PyTorch 2.x Native SDPA (Scaled Dot-Product Attention)
  3. Optimized Multi-worker DataLoader with pin_memory
  4. Explicit Optimizer Step Tracking vs Batch Iterations
  5. Best checkpoint saving based on validation loss

Usage:
    python train_optimized.py [--epochs 3] [--batch_size 8] [--grad_accum 4]
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
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)

from config import (
    BPCC_DIR,
    LORA_CONFIG,
    MODEL_DIR,
    MODEL_NAME,
    SRC_LANG,
    TGT_LANG,
    TRAIN_CONFIG,
)
from utils import get_device, setup_logging

logger = setup_logging()


class DynamicTranslationDataset(Dataset):
    """Tokenized dataset that stores raw token IDs for dynamic batch padding."""

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_source_length: int = 128,
        max_target_length: int = 128,
        src_lang: str = SRC_LANG,
        tgt_lang: str = TGT_LANG,
    ):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        source = str(row["english"]).strip()
        target = str(row["kashmiri"]).strip()

        # Tokenize source without padding (dynamic padding in collator)
        self.tokenizer.src_lang = self.src_lang
        source_enc = self.tokenizer(
            source,
            max_length=self.max_source_length,
            truncation=True,
        )

        # Tokenize target without padding
        target_enc = self.tokenizer(
            text_target=target,
            max_length=self.max_target_length,
            truncation=True,
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

    logger.info(f"[+] Loaded {len(train_df)} training pairs")
    if val_df is not None:
        logger.info(f"[+] Loaded {len(val_df)} validation pairs")

    return train_df, val_df


def setup_model_for_training(model_name: str, device: torch.device):
    """Load model with SDPA attention and apply LoRA adapters."""
    logger.info(f"[*] Loading model and tokenizer: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
    )

    # Enable native PyTorch SDPA attention
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            attn_implementation="sdpa",
        )
        logger.info("    [+] SDPA (Scaled Dot-Product Attention) enabled")
    except Exception:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias=LORA_CONFIG["bias"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"    Trainable params: {trainable_params / 1e6:.2f}M / {total_params / 1e6:.2f}M "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model, tokenizer


def evaluate_model(model, val_loader, device):
    """Run validation loss calculation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda") if device.type == "cuda" else torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                total_loss += outputs.loss.item()
                num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    model.train()
    return avg_loss


def train(args):
    """Main training routine with dynamic batch padding and GPU optimizations."""
    logger.info("=" * 60)
    logger.info("KATHE 2026 -- Optimized GPU Fine-Tuning")
    logger.info("=" * 60)

    device = get_device()
    train_data, val_data = load_training_data()
    model, tokenizer = setup_model_for_training(args.model_name, device)

    # Dynamic Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    train_dataset = DynamicTranslationDataset(
        train_data,
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = None
    if val_data is not None:
        val_dataset = DynamicTranslationDataset(
            val_data,
            tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=2,
            pin_memory=True if device.type == "cuda" else False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_optimizer_steps = (len(train_loader) * args.epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_optimizer_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None

    logger.info(f"\n[*] Training Metrics & Hardware Plan:")
    logger.info(f"    Total Dataset:           {len(train_data):,} pairs")
    logger.info(f"    Epochs:                  {args.epochs}")
    logger.info(f"    Micro Batch Size:        {args.batch_size}")
    logger.info(f"    Grad Accumulation Steps: {args.gradient_accumulation_steps}")
    logger.info(f"    Effective Batch Size:    {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"    Batches per Epoch:       {len(train_loader):,}")
    logger.info(f"    Total Optimizer Updates: {total_optimizer_steps:,}")
    logger.info(f"    Warmup Steps:            {warmup_steps:,}")
    logger.info(f"    Dynamic Padding:         Active (pad_to_multiple_of=8)")
    logger.info(f"    Mixed Precision (fp16):  {args.fp16}")

    best_val_loss = float("inf")
    opt_step = 0

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
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()

                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    opt_step += 1
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    opt_step += 1

            epoch_loss += outputs.loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{outputs.loss.item():.4f}",
                "avg_loss": f"{epoch_loss / num_batches:.4f}",
                "opt_step": f"{opt_step}/{total_optimizer_steps}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # End of epoch validation
        if val_loader:
            logger.info(f"\n[*] Running validation for epoch {epoch + 1}...")
            val_loss = evaluate_model(model, val_loader, device)
            logger.info(f"    Epoch {epoch + 1} Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f})")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_dir = MODEL_DIR / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                logger.info(f"    [+] New best model saved -> {best_dir}")

        # Save checkpoint per epoch
        epoch_dir = MODEL_DIR / f"checkpoint-epoch-{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        logger.info(f"    [+] Epoch checkpoint saved -> {epoch_dir}")

    logger.info("\n" + "=" * 60)
    logger.info("[+] Training Complete!")
    logger.info(f"    Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"    Best Model Saved At:  {MODEL_DIR / 'best'}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 Optimized Fine-Tuning")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=TRAIN_CONFIG["gradient_accumulation_steps"])
    parser.add_argument("--learning_rate", type=float, default=TRAIN_CONFIG["learning_rate"])
    parser.add_argument("--warmup_ratio", type=float, default=TRAIN_CONFIG["warmup_ratio"])
    parser.add_argument("--weight_decay", type=float, default=TRAIN_CONFIG["weight_decay"])
    parser.add_argument("--max_source_length", type=int, default=TRAIN_CONFIG["max_source_length"])
    parser.add_argument("--max_target_length", type=int, default=TRAIN_CONFIG["max_target_length"])
    parser.add_argument("--fp16", action="store_true", default=TRAIN_CONFIG["fp16"])
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
