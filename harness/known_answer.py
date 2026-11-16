"""Synthetic data where the answer is known: what the same-tree difference measures.

The target is lognormal and independent of the predictors, so the predictors
hold no signal. The test runs on it twice. When the trees grow on the target
itself, their splits see the values that fill the leaves. When the trees grow
on separate noise, the splits never see those values, the catchments do not
depend on them, and adaptive and honest fills have the same expected total.

Writes ``results/public/known_answer.csv``: adaptive over honest minus one, %,
mean over seeds and its standard error, for each number of donors.

Usage: python known_answer.py [--out ../results/public/known_answer.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import selection as S  # noqa: E402

OUT = HERE.parent / "results/public/known_answer.csv"
SEEDS = 60
TREES = 15
RECIPIENTS = 600
_REAL_GROW = S.grow


def _grow_on_noise(X, y, seed, n_estimators, min_samples_leaf=1, store_leaf_samples=False):
    z = np.random.default_rng([int(seed), 99]).normal(size=len(y))
    return _REAL_GROW(X, z, seed, n_estimators, min_samples_leaf, store_leaf_samples)


def relative_difference(n: int, grown_on: str, seed: int) -> float:
    """Adaptive over honest minus one, %, for one synthetic sample."""
    S.grow = _grow_on_noise if grown_on == "noise" else _REAL_GROW
    try:
        rng = np.random.default_rng([seed, 5])
        X = rng.normal(size=(n, 3))
        y = np.exp(rng.normal(size=n))
        w = rng.uniform(1, 4, size=n)
        X_rec = rng.normal(size=(RECIPIENTS, 3))
        res = S.same_tree_sign(X, y, w, X_rec, np.ones(RECIPIENTS), seed=seed, n_estimators=TREES)
    finally:
        S.grow = _REAL_GROW
    totals = res["variants"]["1"]["totals"]
    return 100 * (totals["adaptive"]["pooled"] / totals["honest"]["pooled"] - 1)


def table() -> pd.DataFrame:
    rows = []
    for n in (1200, 4800):
        for grown_on in ("noise", "target"):
            d = np.array([relative_difference(n, grown_on, seed) for seed in range(SEEDS)])
            rows.append({"donors": n, "trees grown on": grown_on, "trees": TREES, "seeds": SEEDS,
                         "adaptive_over_honest_pct": d.mean(),
                         "se": d.std(ddof=1) / np.sqrt(len(d))})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    df = table()
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
