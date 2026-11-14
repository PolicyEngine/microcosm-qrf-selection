"""Coverage at each tau, for every fill and leaf floor the remedy runs tried.

``remedies.py`` reduces each configuration's coverage to one gap. This writes
the share of held-out positive donors at or below each conditional quantile,
mean over folds, to ``results/coverage_by_tau.csv``: one row per target, fill
and ``min_samples_leaf``, positive forest, pooled leaves.

Usage: python coverage_by_tau.py [--out ../results/coverage_by_tau.csv]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAG = "_remedies"
TAUS = ("0.1", "0.25", "0.5", "0.75", "0.9")


def table() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(ROOT / f"results/selection_*{TAG}.json"))):
        doc = json.loads(Path(path).read_text())
        key = Path(path).stem.removeprefix("selection_").removesuffix(TAG)
        for target in doc["targets"]:
            for leaf in doc.get("variants", [1]):
                for fill in ("adaptive", "honest"):
                    vals = {tau: [] for tau in TAUS}
                    for fold in doc["folds"]:
                        sign = (fold["targets"].get(target, {}).get("signs") or {}).get("1") or {}
                        q = sign.get("variants", {}).get(str(leaf), {}).get("quantile")
                        m = q and q[fill]["pooled"]
                        if not m:
                            continue
                        for tau in TAUS:
                            vals[tau].append(m["coverage"][tau])
                    if vals[TAUS[0]]:
                        rows.append({"key": key, "target": target, "fill": fill,
                                     "min_samples_leaf": int(leaf), "folds": len(vals[TAUS[0]]),
                                     **{f"coverage_{tau}": float(np.mean(vals[tau])) for tau in TAUS}})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/coverage_by_tau.csv")
    args = ap.parse_args()
    df = table()
    df.to_csv(args.out, index=False)
    print(f"{len(df)} rows, {df[['key', 'target']].drop_duplicates().shape[0]} targets")


if __name__ == "__main__":
    main()
