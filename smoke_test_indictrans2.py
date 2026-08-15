"""
Smoke test: Validates IndicTrans2 pipeline on a tiny batch before committing to full training.
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

from config import BPCC_DIR
from utils import get_device

IT2_MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
IT2_SRC_LANG = "eng_Latn"
IT2_TGT_LANG = "kas_Arab"

def vram_report(label):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        resv = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  [{label}] Allocated: {alloc:.2f} GB | Reserved: {resv:.2f} GB | Peak: {peak:.2f} GB")

print("=" * 60)
print("SMOKE TEST - IndicTrans2-1B Validation")
print("=" * 60)

device = get_device()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

# 1. Test IndicTransToolkit Import
print("\n[1/6] Checking IndicTransToolkit...")
try:
    from IndicTransToolkit import IndicProcessor
    ip = IndicProcessor(inference=True)
    print("  [+] IndicTransToolkit and IndicProcessor loaded successfully!")
except ImportError as e:
    print(f"  [-] IndicTransToolkit import failed: {e}")
    print("      Make sure build tools are installed and run: pip install indictranstoolkit")
    sys.exit(1)

# 2. Load small training sample
print("\n[2/6] Loading sample data...")
df = pd.read_csv(BPCC_DIR / "train.csv", nrows=32)
print(f"  Loaded {len(df)} sample rows")

# 3. Load model and tokenizer
print(f"\n[3/6] Loading IndicTrans2-1B model & LoRA config...")
tokenizer = AutoTokenizer.from_pretrained(IT2_MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    IT2_MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
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
model = model.to(device)

for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)")
vram_report("After model load")

# 4. Run 5 training steps
print("\n[4/6] Training 5 steps on micro-batch (batch_size=4)...")
from train_indictrans2 import IT2TranslationDataset

ip_train = IndicProcessor(inference=False)
dataset = IT2TranslationDataset(df, tokenizer, ip_train, max_source_length=128, max_target_length=128)
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, pad_to_multiple_of=8, label_pad_token_id=-100)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator, num_workers=0)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-5)
scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

model.train()
for step, batch in enumerate(loader):
    if step >= 5:
        break
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    if scaler:
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    print(f"  Step {step+1}/5: loss={loss.item():.4f}")

vram_report("After training steps")

# 5. Test Inference
print("\n[5/6] Testing inference with IndicProcessor...")
del optimizer, loader, dataset, collator
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model.eval()
test_sentences = [
    "She was a true visionary.",
    "I go to my school daily.",
    "The weather is beautiful today."
]

preprocessed = ip.preprocess_batch(test_sentences, src_lang=IT2_SRC_LANG, tgt_lang=IT2_TGT_LANG)
inputs = tokenizer(preprocessed, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        use_cache=True,
    )

raw_trans = tokenizer.batch_decode(generated, skip_special_tokens=True)
translations = ip.postprocess_batch(raw_trans, lang=IT2_TGT_LANG)

for en, ks in zip(test_sentences, translations):
    print(f"  EN: {en}")
    try:
        print(f"  KS: {ks}")
    except:
        print(f"  KS: [unicode display error]")
    print()

vram_report("After inference")

# 6. Final VRAM Check
print("[6/6] VRAM Summary:")
if torch.cuda.is_available():
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Peak memory used:  {peak:.2f} GB")
    print(f"  GPU total:         8.00 GB")
    print(f"  Headroom:          {8.0 - peak:.2f} GB")

print("\n" + "=" * 60)
print("INDICTRANS2 SMOKE TEST COMPLETE!")
print("=" * 60)
