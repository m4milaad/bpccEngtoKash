"""
Smoke test: validates new config end-to-end with VRAM optimization.
Trains 10 steps, measures VRAM per phase, clears cache, then tests inference.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass

import gc
import torch
import pandas as pd
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_cosine_schedule_with_warmup,
)
import torch.nn.functional as F

from config import (
    BPCC_DIR, LORA_CONFIG, MODEL_NAME, SRC_LANG, TGT_LANG, TRAIN_CONFIG,
)
from train_optimized import DynamicTranslationDataset
from utils import get_device


def vram_report(label):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        resv = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  [{label}] Allocated: {alloc:.2f} GB | Reserved: {resv:.2f} GB | Peak: {peak:.2f} GB")


print("=" * 60)
print("SMOKE TEST - New Config Validation (Optimized)")
print("=" * 60)

device = get_device()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# 1. Load a small subset of data
print("\n[1/6] Loading small data subset...")
train_df = pd.read_csv(BPCC_DIR / "train.csv", nrows=64)
print(f"  Loaded {len(train_df)} training pairs")

# 2. Load model with NEW LoRA config
print(f"\n[2/6] Loading model with NEW LoRA config...")
print(f"  r={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']}")
print(f"  modules={LORA_CONFIG['target_modules']}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device.type == "cuda" else torch.float32,
    attn_implementation="sdpa",
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

for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)")
vram_report("After model load")

# 3. Training with label smoothing + cosine schedule
ls = TRAIN_CONFIG.get("label_smoothing", 0.1)
print(f"\n[3/6] Training 10 steps (label_smoothing={ls}, cosine, LR={TRAIN_CONFIG['learning_rate']})...")

dataset = DynamicTranslationDataset(train_df, tokenizer)
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100)
loader = DataLoader(dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, collate_fn=collator, num_workers=0)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=TRAIN_CONFIG["learning_rate"])
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2, num_training_steps=10)
scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

model.train()
for step, batch in enumerate(loader):
    if step >= 10:
        break
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    if scaler:
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if ls > 0:
                logits = outputs.logits
                vocab_size = logits.size(-1)
                smooth_labels = labels.clone()
                pad_mask = smooth_labels == -100
                smooth_labels[pad_mask] = 0
                nll_loss = F.cross_entropy(logits.view(-1, vocab_size), smooth_labels.view(-1), reduction='none', ignore_index=-100)
                smooth_loss = -F.log_softmax(logits.view(-1, vocab_size), dim=-1).sum(dim=-1)
                non_pad = (~pad_mask).view(-1).float()
                nll_loss = (nll_loss * non_pad).sum() / non_pad.sum().clamp(min=1)
                smooth_loss = (smooth_loss * non_pad).sum() / non_pad.sum().clamp(min=1)
                loss = (1 - ls) * nll_loss + ls * smooth_loss / vocab_size
            else:
                loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
    
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    print(f"  Step {step+1}/10: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.2e}")

vram_report("After training")
print("  Training OK!")

# 4. Clear memory before inference
print("\n[4/6] Clearing training state for inference...")
del optimizer, scheduler, scaler, loader, dataset, collator
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
vram_report("After cleanup")

# 5. Test inference
print("\n[5/6] Testing inference (5 sentences, num_beams=8)...")
model.eval()
test_sentences = [
    "She was a true visionary.",
    "I go to my school daily.",
    "The weather is beautiful today.",
    "He is reading a book.",
    "We need more water."
]

tokenizer.src_lang = SRC_LANG
inputs = tokenizer(test_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
tgt_lang_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

with torch.no_grad():
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tgt_lang_id,
        max_length=128,
        num_beams=8,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )

translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
for en, ks in zip(test_sentences, translations):
    print(f"  EN: {en}")
    try:
        print(f"  KS: {ks}")
    except:
        print(f"  KS: [unicode display error]")
    print()
vram_report("After inference")

# 6. Final summary
print("[6/6] Final VRAM Summary:")
if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Peak memory used:  {peak:.2f} GB")
    print(f"  GPU total:         8.00 GB")
    print(f"  Headroom:          {8.0 - peak:.2f} GB")
    if peak > 7.5:
        print("  WARNING: Very tight on VRAM! Consider batch_size=4 + grad_accum=4")
    elif peak > 6.5:
        print("  OK: Tight but should work for full training")
    else:
        print("  GOOD: Plenty of headroom")

print("\n" + "=" * 60)
print("SMOKE TEST PASSED - Safe to run full training!")
print("=" * 60)
