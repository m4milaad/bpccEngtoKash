import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset
import pandas as pd

configs = [
    'bpcc-seed-latest',
    'nllb-filtered',
    'samanantar-filtered',
    'bpcc-seed-v2',
    'nllb-seed',
    'massive',
    'daily',
    'comparable',
    'ilci'
]

print("Checking BPCC configs for Kashmiri...")
for cfg in configs:
    try:
        print(f"\n--- Checking config: {cfg} ---")
        ds = load_dataset("ai4bharat/BPCC", cfg, split="train", streaming=True)
        # Inspect first 5 examples
        for i, example in enumerate(ds):
            print(f"Sample {i}: keys = {list(example.keys())}")
            # check languages or content
            print(f"  Content: {example}")
            if i >= 2:
                break
    except Exception as e:
        print(f"Error checking {cfg}: {e}")
