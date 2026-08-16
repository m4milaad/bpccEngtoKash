"""
KATHE 2026 — IndicTrans2 Fresh Training (High Capacity)
LoRA r=64, alpha=128, trainable embeddings, 10 epochs.
"""

import argparse
import os
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

from config import BPCC_DIR, MODEL_DIR, SRC_LANG, TGT_LANG
from utils import get_device, setup_logging

logger = setup_logging()

IT2_MODEL = "ai4bharat/indictrans2-en-indic-1B"
IT2_SRC = "eng_Latn"
IT2_TGT = "kas_Arab"


class DynamicTranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_length=128, max_target_length=128):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        source = str(row["english"]).strip()
        target = str(row["kashmiri"]).strip()

        from IndicTransToolkit import IndicProcessor
        ip = IndicProcessor(inference=False)
        preprocessed = ip.preprocess_batch([source], src_lang=IT2_SRC, tgt_lang=IT2_TGT)[0]

        source_enc = self.tokenizer(
            preprocessed, max_length=self.max_source_length, truncation=True
        )
        target_enc = self.tokenizer(
            text_target=target, max_length=self.max_target_length, truncation=True
        )

        return {
            "input_ids": source_enc["input_ids"],
            "attention_mask": source_enc["attention_mask"],
            "labels": target_enc["input_ids"],
        }


def load_data():
    train_path = BPCC_DIR / "train.csv"
    val_path = BPCC_DIR / "val.csv"
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path) if val_path.exists() else None
    logger.info(f"[+] Loaded {len(train_df)} train, {len(val_df) if val_df is not None else 0} val")
    return train_df, val_df


def setup_model(device, lora_r=64, lora_alpha=128, lr=3e-5):
    logger.info(f"[*] Loading IndicTrans2: {IT2_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(IT2_MODEL, trust_remote_code=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        IT2_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )

    # High-capacity LoRA + trainable embeddings
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        modules_to_save=["embed_tokens", "lm_head"],  # train vocab embeddings
    )

    model = get_peft_model(model, lora_config).to(device)

    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"    Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")

    return model, tokenizer


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda"):
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            total_loss += loss.item()
    model.train()
    return total_loss / max(len(val_loader), 1)


def train(args):
    device = get_device()
    train_df, val_df = load_data()
    model, tokenizer = setup_model(device, args.lora_r, args.lora_alpha, args.learning_rate)

    from IndicTransToolkit import IndicProcessor
    ip = IndicProcessor(inference=False)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100)

    train_ds = DynamicTranslationDataset(train_df, tokenizer, args.max_source_length, args.max_target_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator, num_workers=0, pin_memory=True)

    val_loader = None
    if val_df is not None:
        val_ds = DynamicTranslationDataset(val_df, tokenizer, args.max_source_length, args.max_target_length)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size*2, shuffle=False, collate_fn=data_collator, num_workers=0, pin_memory=True)

    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(opt_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) * args.epochs) // args.gradient_accumulation_steps
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda") if args.fp16 and device.type == "cuda" else None

    best_loss = float("inf")
    step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for bi, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if scaler:
                with torch.amp.autocast("cuda"):
                    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                if (bi + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    step += 1
            else:
                loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss / args.gradient_accumulation_steps
                loss.backward()
                if (bi + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    step += 1

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            pbar.set_postfix(loss=f"{loss.item()*args.gradient_accumulation_steps:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        if val_loader:
            vloss = evaluate(model, val_loader, device)
            logger.info(f"  Val Loss: {vloss:.4f} (Best: {best_loss:.4f})")
            if vloss < best_loss:
                best_loss = vloss
                best_dir = MODEL_DIR / "indictrans2-best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                logger.info(f"  [+] Saved best to {best_dir}")

        ckpt = MODEL_DIR / f"indictrans2-checkpoint-epoch-{epoch+1}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

    logger.info(f"[+] Done. Best val loss: {best_loss:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_source_length", type=int, default=128)
    p.add_argument("--max_target_length", type=int, default=128)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()