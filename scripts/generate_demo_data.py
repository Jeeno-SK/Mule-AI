"""Generate reproducible SYNTHETIC demo files in data/.

Usage:  python scripts/generate_demo_data.py [--seed 42] [--accounts 1200] [--tx 8000]
Writes: data/demo_transactions.csv, data/demo_ground_truth_accounts.csv, data/demo_ground_truth_transactions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.synthetic_generator import SyntheticConfig, generate_synthetic  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--accounts", type=int, default=1200)
    p.add_argument("--tx", type=int, default=8000)
    p.add_argument("--days", type=int, default=30)
    args = p.parse_args()
    ds = generate_synthetic(SyntheticConfig(seed=args.seed, n_accounts=args.accounts, n_legit_tx=args.tx, days=args.days))
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    ds.transactions.to_csv(out / "demo_transactions.csv", index=False)
    ds.ground_truth_accounts.to_csv(out / "demo_ground_truth_accounts.csv", index=False)
    ds.ground_truth_transactions.to_csv(out / "demo_ground_truth_transactions.csv", index=False)
    print(f"Wrote {len(ds.transactions):,} SYNTHETIC transactions ({int(ds.transactions.is_fraud.sum())} generator-labelled) to {out}")


if __name__ == "__main__":
    main()
