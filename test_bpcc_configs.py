import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset
import pandas as pd

configs = ['bpcc-seed-latest', 'nllb-filtered', 'samanantar-filtered', 'massive', 'ilci']

for cfg in configs:
    try:
        print(f"\n================ Loading BPCC config: {cfg} ================")
        ds = load_dataset("ai4bharat/BPCC", cfg)
        print(f"Available splits/languages in {cfg}: {list(ds.keys())}")
        for k in list(ds.keys()):
            if 'kas' in k.lower() or 'ks' in k.lower():
                print(f"  --> Found Kashmiri split: {k} with {len(ds[k])} pairs!")
                print(f"      Sample: {ds[k][0]}")
    except Exception as e:
        print(f"Error on {cfg}: {e}")
