# Running the IndicTrans2 pipeline on Kaggle (2x T4)

## What changed

- **`train_indictrans2.py`** now uses HuggingFace **Accelerate** for real DDP
  training across both T4s (was single-GPU only). It still runs fine on one
  GPU with no flags changed.
- **`inference_indictrans2.py`** auto-detects multiple GPUs and splits the
  test set across them with independent processes (`torch.multiprocessing`) —
  no DDP needed for inference, just parallel copies of the model.
- **`smoke_test_indictrans2.py`** now reports actual detected VRAM instead of
  an assumed 8 GB, and prints how many GPUs it sees.
- **`utils.py`** gained `print_gpu_summary()`.
- New **`accelerate_config.yaml`** — launch config for 2 GPUs, fp16.

Notes on T4: it's Turing architecture, so it has fast **fp16** tensor cores
but **no bf16** support — `fp16=True` (already the project default) is the
right choice; don't switch to bf16 on Kaggle T4s.

## 1. Notebook settings

Settings → Accelerator → **GPU T4 x2**. Confirm with:

```python
!nvidia-smi
```

You should see two `Tesla T4` entries (16 GB each).

## 2. Install deps

```python
!pip install -q accelerate indictranstoolkit
```

## 3. Sanity check (single GPU, ~1 min)

```python
!python smoke_test_indictrans2.py
```

Confirms the model, LoRA config, and data pipeline all work before you
commit a full training run.

## 4. Full training — both T4s

Run as a script via `!`, not pasted into a cell directly — `accelerate
launch` spawns fresh processes, which plays much more reliably with CUDA
than trying to fork inside the notebook kernel.

```python
!NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch \
    --config_file accelerate_config.yaml \
    train_indictrans2.py \
    --epochs 5 \
    --batch_size 8 \
    --gradient_accumulation_steps 2
```

`--batch_size` is **per GPU**. With 2 GPUs and `grad_accum=2` that's an
effective batch of `8 * 2 * 2 = 32`. Each T4 has 16 GB — roughly double the
8 GB the original defaults were tuned for — so `batch_size 8-12` per GPU
should have comfortable headroom for IndicTrans2-1B + LoRA at
`max_length=128`. Push it up and rerun the smoke test's VRAM report if you
want to confirm before a long run.

`NCCL_P2P_DISABLE=1` / `NCCL_IB_DISABLE=1` avoid a known hang on Kaggle's
2xT4 instances, which don't expose reliable GPU-to-GPU peer access. The
training script also sets these itself as a fallback, but setting them in
the shell is more reliable since they must be set before NCCL initializes.

## 5. Inference — both T4s

```python
!python inference_indictrans2.py --model_path models/indictrans2-best
```

This automatically splits the test set across both GPUs. Add `--single_gpu`
to force single-GPU inference (e.g. for debugging).

## 6. Everything else

`evaluate.py` and `submission.py` are unchanged and don't need multi-GPU —
they're CPU-bound (scoring/formatting), not model inference.
