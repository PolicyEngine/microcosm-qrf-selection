"""Each remedy against the forest production would grow on the fold's donors.

The same-tree test grows half-size forests so that an honest half remains, so
``remedies.py`` compares every remedy with a half-size adaptive forest.
``run_full_size.sh`` also grows each fold's forest at production's size, at
``min_samples_leaf`` 1, 5 and 20 (``selection.full_size_forests``). This script
compares, with that full-size leaf-1 pooled forest as the baseline:

- the leaf floors and ``weighted_leaves=True`` at full size;
- honest fills as the same-tree test builds them (trees grown on one half,
  filled with the other half, the halves swapped and averaged), which is the
  cross-fitted estimator option 3 would ship;
- the half-size adaptive fill, the baseline ``remedies.py`` uses.

Writes ``results/full_size.csv``, one row per target and configuration, each
measure a mean over folds with its standard error:

- ``effect``: the configuration's total minus the baseline's, % of the
  held-out reported total;
- ``vs_reported``: the configuration's total against the held-out reported
  total, %;
- ``prod_effect``: the configuration's total over the baseline's on the
  production recipients, minus one, % (a ratio of two model totals, which does
  not depend on how far the forest's scale sits from the release's draws);
- ``pinball_change``: the change in mean pinball loss of the positive forest's
  quantiles on the held-out positive donors, relative to the baseline, %;
- ``coverage_error``: the mean absolute gap between each tau and the share of
  held-out positive donors at or below that quantile; ``coverage_<tau>`` holds
  that share at each tau.

Usage: python full_size.py [--out ../results/full_size.csv]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from remedies import _mean_se  # noqa: E402

ROOT = HERE.parent
TAG = "_fullsize"
BASE = "production"
#: Configuration: which forest, which entry of it, how the pool reads it.
CONFIGS = {
    BASE: ("full", "1", "pooled"),
    "honest fills": ("half", "honest", "pooled"),
    "min_samples_leaf=5": ("full", "5", "pooled"),
    "min_samples_leaf=20": ("full", "20", "pooled"),
    "weighted_leaves=True": ("full", "1", "tree"),
    "half-size adaptive": ("half", "adaptive", "pooled"),
}
#: The leaf-1 forest regrown here must equal the fold's own forest this closely.
OWN_FOREST_TOLERANCE = 1e-9
TAUS = ("0.1", "0.25", "0.5", "0.75", "0.9")


def _config_total(sign: dict, kind: str, entry: str, mode: str) -> float:
    if kind == "full":
        return sign["full_size"][entry]["totals"][mode]
    return sign["variants"]["1"]["totals"][entry][mode]


def _config_production(sign: dict, kind: str, entry: str, mode: str) -> float:
    if kind == "full":
        return sign["full_size"][entry]["extra"]["production"][mode]
    return sign["extra"]["production"]["variants"]["1"]["totals"][entry][mode]


def _config_quantile(sign: dict, kind: str, entry: str, mode: str) -> dict | None:
    if kind == "full":
        return (sign.get("full_size", {}).get(entry, {}).get("quantile") or {}).get(mode)
    return (sign.get("variants", {}).get("1", {}).get("quantile") or {}).get(entry, {}).get(mode)


def fold_target(target: dict, where: str = "") -> dict | None:
    """Per-configuration totals for one target in one fold, signs summed."""
    signs = target.get("signs") or {}
    if not signs or any("skipped" in s or "full_size" not in s for s in signs.values()):
        return None
    for s in signs.values():
        ours, own = s["full_size"]["1"]["totals"]["pooled"], s["own_forest"]["pooled"]
        if abs(ours - own) > OWN_FOREST_TOLERANCE * max(abs(own), 1.0):
            raise ValueError(f"{where}: the regrown leaf-1 forest totals {ours}, the fold's own {own}")
    has_prod = all("extra" in s for s in signs.values())
    pos = signs.get("1", {})
    tot, prod, quant = {}, {}, {}
    for name, cfg in CONFIGS.items():
        tot[name] = sum(_config_total(s, *cfg) for s in signs.values())
        if has_prod:
            prod[name] = sum(_config_production(s, *cfg) for s in signs.values())
        quant[name] = _config_quantile(pos, *cfg)
    return {"tot": tot, "prod": prod if has_prod else None, "quant": quant,
            "reported": target["reported"]}


def summarize_key(key: str, doc: dict) -> list[dict]:
    rows = []
    for target in doc["targets"]:
        parsed = [fold_target(f["targets"][target], f"{key}/{target}/fold {f.get('fold')}")
                  for f in doc["folds"] if target in f["targets"]]
        parsed = [p for p in parsed if p is not None and p["reported"]]
        if not parsed:
            continue
        for name in CONFIGS:
            def pct(fn):
                vals = []
                for p in parsed:
                    try:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            vals.append(fn(p))
                    except (ZeroDivisionError, TypeError, KeyError):
                        vals.append(None)
                return _mean_se(vals)

            row = {"key": key, "target": target, "folds": len(parsed), "config": name}
            row["effect"], row["effect_se"], _ = pct(
                lambda p: 100 * (p["tot"][name] - p["tot"][BASE]) / p["reported"])
            row["vs_reported"], row["vs_reported_se"], _ = pct(
                lambda p: 100 * (p["tot"][name] / p["reported"] - 1))
            if all(p["prod"] is not None for p in parsed):
                row["prod_effect"], row["prod_effect_se"], _ = pct(
                    lambda p: 100 * (p["prod"][name] / p["prod"][BASE] - 1))
            row["pinball_change"], row["pinball_change_se"], row["pinball_folds"] = pct(
                lambda p: 100 * (p["quant"][name]["mean_pinball"]
                                 / p["quant"][BASE]["mean_pinball"] - 1))
            row["coverage_error"], row["coverage_error_se"], _ = pct(
                lambda p: float(np.mean([abs(c - float(tau)) for tau, c in
                                         p["quant"][name]["coverage"].items()])))
            for tau in TAUS:
                row[f"coverage_{tau}"], _, _ = pct(lambda p: p["quant"][name]["coverage"][tau])
            rows.append(row)
    return rows


def table() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(ROOT / f"results/selection_*{TAG}.json"))):
        key = Path(path).stem.removeprefix("selection_").removesuffix(TAG)
        doc = json.loads(Path(path).read_text())
        if doc.get("folds"):
            rows.extend(summarize_key(key, doc))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/full_size.csv")
    args = ap.parse_args()
    df = table()
    df.to_csv(args.out, index=False)
    print(f"{df[['key', 'target']].drop_duplicates().shape[0]} targets across {df['key'].nunique()} fits")


if __name__ == "__main__":
    main()
