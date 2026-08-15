"""
KATHE 2026 — Training Script
Fine-tune a pretrained model on BPCC English-Kashmiri parallel data using LoRA (PEFT).

Usage:
    python train.py                              # Default settings
    python train.py --epochs 5 --batch_size 4    # Custom settings
    python train.py --model_name facebook/nllb-200-distilled-600M
"""

import argparse
import math
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
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from config import (
    MODEL_NAME,
    SRC_LANG,
    TGT_LANG,
    BPCC_DIR,
    MODEL_DIR,
    LOG_DIR,
    TRAIN_CONFIG,
    LORA_CONFIG,
)
from utils import setup_logging, get_device

logger = setup_logging(log_file=str(LOG_DIR / "train.log"))


class TranslationDataset(Dataset):
    """Dataset for English-Kashmiri parallel sentences."""

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

        # Tokenize source
        self.tokenizer.src_lang = self.src_lang
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize target using text_target
        target_encoding = self.tokenizer(
            text_target=target,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = source_encoding["input_ids"].squeeze()
        attention_mask = source_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()

        # Replace padding token IDs with -100 so they are ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_training_data():
    """Load prepared BPCC training data."""
    train_path = BPCC_DIR / "train.csv"
    val_path = BPCC_DIR / "val.csv"

    if not train_path.exists():
        logger.error(
            f"[-] Training data not found at {train_path}\n"
            "    Run 'python download_data.py' first to download and prepare data."
        )
        raise FileNotFoundError(f"Training data not found: {train_path}")

    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path) if val_path.exists() else None

    logger.info(f"[*] Training data: {len(train_data)} pairs")
    if val_data is not None:
        logger.info(f"[*] Validation data: {len(val_data)} pairs")

    return train_data, val_data


def setup_model_for_training(model_name: str, device: torch.device):
    """Load model and apply LoRA adapters for efficient fine-tuning."""
    logger.info(f"[+] Loading base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    )

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
    """Run evaluation on the validation set and return average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

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
    """Main training loop."""
    logger.info("=" * 60)
    logger.info("KATHE 2026 -- Fine-Tuning")
    logger.info("=" * 60)

    device = get_device()

    train_data, val_data = load_training_data()
    model, tokenizer = setup_model_for_training(args.model_name, device)

    train_dataset = TranslationDataset(
        train_data,
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = None
    if val_data is not None:
        val_dataset = TranslationDataset(
            val_data,
            tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device.type == "cuda" else False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None

    logger.info(f"\n[*] Training Configuration:")
    logger.info(f"    Epochs:                  {args.epochs}")
    logger.info(f"    Batch size:              {args.batch_size}")
    logger.info(f"    Gradient accumulation:   {args.gradient_accumulation_steps}")
    logger.info(f"    Effective batch size:    {args.batch_size * args.gradient_accumulation_steps}")
    logger.info(f"    Learning rate:           {args.learning_rate}")
    logger.info(f"    Total steps:             {total_steps}")
    logger.info(f"    Warmup steps:            {warmup_steps}")
    logger.info(f"    Mixed precision (fp16):  {args.fp16}")

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for step, batch in enumerate(pbar):
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

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

            epoch_loss += outputs.loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{outputs.loss.item():.4f}",
                "avg_loss": f"{epoch_loss / num_batches:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            if global_step > 0 and global_step % args.logging_steps == 0:
                logger.info(
                    f"    Step {global_step}: loss={epoch_loss / num_batches:.4f}, "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        avg_train_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"\n[*] Epoch {epoch + 1}/{args.epochs} -- Train Loss: {avg_train_loss:.4f}")

        if val_loader:
            val_loss = evaluate_model(model, val_loader, device)
            logger.info(f"    Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = MODEL_DIR / "best"
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info(f"    [+] Best model saved to {save_path} (val_loss={val_loss:.4f})")
        else:
            save_path = MODEL_DIR / f"epoch_{epoch + 1}"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info(f"    [+] Model saved to {save_path}")

    final_path = MODEL_DIR / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"\n[+] Training complete! Final model saved to {final_path}")

    if val_loader:
        logger.info(f"    Best validation loss: {best_val_loss:.4f}")
        logger.info(f"    Best model at: {MODEL_DIR / 'best'}")

    logger.info(
        "\nNext steps:\n"
        f"    python inference.py --model_path {MODEL_DIR / 'best'}   # Run inference with fine-tuned model\n"
        "    python evaluate.py                                        # Evaluate on validation set"
    )


def main():
    parser = argparse.ArgumentParser(description="KATHE 2026 -- Fine-Tuning")
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
    parser.add_argument("--logging_steps", type=int, default=TRAIN_CONFIG["logging_steps"])

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

