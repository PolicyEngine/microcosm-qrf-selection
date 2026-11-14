"""Selection in production's expected draws, beside selection in pooled means.

The same-tree test compares each fill's pooled mean. Production draws one value
at a uniform quantile of a curve read off quantile-forest's weighted quantiles,
whose mean differs from the pooled mean. ``run_selection.py --draw-means
--tag _draws`` computes each fill's expected total under that draw; this script
summarizes it. quantile-forest's formula reads masses as frequencies, so its
curve depends on their scale. The honest fill here therefore takes integer
masses built as production builds in-bag ones (``honest_resample``).

Writes ``results/draw_means.csv``, one row per target, percentages of the
held-out reported total:

- ``sel_mean``: selection in pooled means, as the report states it (the
  honest fill at survey-weight mass);
- ``sel_boot_mean``: selection in pooled means against the integer-mass honest
  fill;
- ``sel_draw``: selection in expected draws against that fill;
- ``own_draw_gap``, ``adaptive_draw_gap``, ``honest_draw_gap``: expected draw
  over pooled mean, minus one, for the fold's own forest and each fill;
- ``honest_weight_draw_gap``: the same for the survey-weight honest fill;
- ``mass_effect``: ``sel_boot_mean`` minus ``sel_mean``, what the integer-mass
  fill's resampling moves selection in means by.

Usage: python draw_means.py [--out ../results/draw_means.csv]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TAG = "_draws"


def _fold_totals(target: dict, variant: str = "1") -> dict[str, float] | None:
    """One fold's totals for a target, summed over signs."""
    signs = target["signs"]
    if not signs or any("skipped" in s for s in signs.values()):
        return None
    out = {k: 0.0 for k in ("adaptive_mean", "honest_weight_mean", "honest_mean", "adaptive_draw",
                            "honest_draw", "honest_weight_draw", "own_mean", "own_draw")}
    for s in signs.values():
        totals = s["variants"][variant]["totals"]
        pooled = totals["adaptive"]["pooled"]
        out["adaptive_mean"] += pooled
        out["honest_weight_mean"] += totals["honest"]["pooled"]
        # A constant sign forest has no draw curve: every draw is the one value.
        draws = s.get("draw_totals") or {f: pooled for f in ("adaptive", "honest", "honest_mean",
                                                             "honest_weight")}
        out["adaptive_draw"] += draws["adaptive"]
        out["honest_draw"] += draws["honest"]
        out["honest_mean"] += draws["honest_mean"]
        out["honest_weight_draw"] += draws["honest_weight"]
        out["own_mean"] += s["own_forest"]["pooled"]
        out["own_draw"] += s.get("own_forest_draw_total", s["own_forest"]["pooled"])
    return out


def _mean_se(values: list[float]) -> tuple[float, float]:
    v = np.array([x for x in values if math.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def table() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(ROOT / f"results/selection_*{TAG}.json"))):
        doc = json.loads(Path(path).read_text())
        key = Path(path).stem.removeprefix("selection_").removesuffix(TAG)
        for target in doc["targets"]:
            folds = [f["targets"][target] for f in doc["folds"] if target in f["targets"]]
            pairs = [(t, _fold_totals(t)) for t in folds]
            pairs = [(t, x) for t, x in pairs if x is not None and t["reported"]]
            if not pairs:
                continue
            row = {"key": key, "target": target, "folds": len(pairs)}
            for name, fn in {
                "sel_mean": lambda t, x: 100 * (x["adaptive_mean"] - x["honest_weight_mean"]) / t["reported"],
                "sel_boot_mean": lambda t, x: 100 * (x["adaptive_mean"] - x["honest_mean"]) / t["reported"],
                "sel_draw": lambda t, x: 100 * (x["adaptive_draw"] - x["honest_draw"]) / t["reported"],
                "own_draw_gap": lambda t, x: 100 * (x["own_draw"] / x["own_mean"] - 1),
                "adaptive_draw_gap": lambda t, x: 100 * (x["adaptive_draw"] / x["adaptive_mean"] - 1),
                "honest_draw_gap": lambda t, x: 100 * (x["honest_draw"] / x["honest_mean"] - 1),
                "honest_weight_draw_gap": lambda t, x: 100 * (x["honest_weight_draw"] / x["honest_weight_mean"] - 1),
                "mass_effect": lambda t, x: 100 * (x["honest_weight_mean"] - x["honest_mean"]) / t["reported"],
            }.items():
                row[name], row[f"{name}_se"] = _mean_se([fn(t, x) for t, x in pairs])
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/draw_means.csv")
    args = ap.parse_args()
    df = table()
    df.to_csv(args.out, index=False)
    print(f"{len(df)} targets across {df['key'].nunique()} fits")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
