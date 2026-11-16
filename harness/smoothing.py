"""Each target's adaptive-minus-honest contrast, split into selection and smoothing.

``run_smoothing.sh`` reruns every fit's same-tree test with the same seeds and,
for each sign, fits ``mean_model`` on that sign's held-out donors (which no tree
sees) and splits the contrast with ``contrast_parts``. This script summarizes
those runs. Per target, mean over folds and its standard error:

- ``contrast``: adaptive minus honest, % of the held-out reported total (the
  base run's ``sel_pooled``; ``_tree`` columns use per-tree averaging);
- ``smoothing``: the estimated smoothing part, the two fills' different
  averaging of the estimated conditional mean, same units;
- ``selection_est``: contrast minus estimated smoothing;
- ``prod_*``: the same over the production recipients, % of the drawn total,
  and ``prod_*_vs_honest``, % of the honest fill's model total there (the PUF
  chain's production measure).

A fold whose sign had fewer than ``selection.MEAN_FIT_MIN_ROWS`` held-out
donors has no estimate, and the target's estimate needs every fold.

It refuses a run whose contrast differs from the base run's adaptive minus
honest in the same fold (``CONTRAST_TOLERANCE``, relative): the two runs share
seeds, halves, bootstraps and trees.

Writes ``results/smoothing.csv``.

Usage: python smoothing.py [--out ../results/smoothing.csv]
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
TAG = "_smoothing"
MODES = ("pooled", "tree")
PARTS = ("contrast", "smoothing", "selection")
CONTRAST_TOLERANCE = 1e-9


def _base_contrasts(key: str) -> dict:
    """(fold, target, sign, mode) -> adaptive minus honest in the base run."""
    doc = json.loads((ROOT / f"results/selection_{key}.json").read_text())
    out = {}
    for f in doc["folds"]:
        for target, t in f["targets"].items():
            for sign, s in (t.get("signs") or {}).items():
                if "variants" not in s:
                    continue
                tot = s["variants"]["1"]["totals"]
                for m in MODES:
                    out[(f["fold"], target, sign, m)] = tot["adaptive"][m] - tot["honest"][m]
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= CONTRAST_TOLERANCE * max(abs(a), abs(b), 1.0)


def fold_parts(target: dict, fold: int, name: str, base: dict) -> dict | None:
    """Signs summed: each part per mode, held out and on production; None if unmeasured."""
    signs = target.get("signs") or {}
    if not signs or any("skipped" in s for s in signs.values()):
        return None
    sets = ["held_out"] + (["production"] if all("extra" in s for s in signs.values()) else [])
    out = {e: {m: dict.fromkeys(PARTS, 0.0) for m in MODES} for e in sets}
    honest_prod = dict.fromkeys(MODES, 0.0)
    estimated = True
    for sign, s in signs.items():
        tot = s["variants"]["1"]["totals"]
        for m in MODES:
            direct = tot["adaptive"][m] - tot["honest"][m]
            ref = base.get((fold, name, sign, m))
            if ref is not None and not _close(direct, ref):
                raise ValueError(f"{name} fold {fold} sign {sign} {m}: contrast {direct} "
                                 f"against the base run's {ref}")
            if "production" in sets:
                honest_prod[m] += s["extra"]["production"]["variants"]["1"]["totals"]["honest"][m]
        parts = s.get("smoothing")
        if parts is None:
            estimated = False
            continue
        for e in sets:
            for m in MODES:
                for part in PARTS:
                    out[e][m][part] += parts[e][m][part]
        for m in MODES:
            if not _close(parts["held_out"][m]["contrast"], tot["adaptive"][m] - tot["honest"][m]):
                raise ValueError(f"{name} fold {fold} sign {sign} {m}: the split's contrast "
                                 "differs from the run's own totals")
    if not estimated:
        return {"estimated": False}
    return {"estimated": True, "parts": out, "honest_prod": honest_prod,
            "reported": target["reported"],
            "drawn": (target.get("production") or {}).get("drawn_total"),
            "n_fit": min(s["smoothing"]["n_fit"] for s in signs.values() if s.get("smoothing"))}


def summarize_key(key: str, doc: dict) -> list[dict]:
    base = _base_contrasts(key)
    rows = []
    for target in doc["targets"]:
        parsed = [fold_parts(f["targets"][target], f["fold"], target, base)
                  for f in doc["folds"] if target in f["targets"]]
        parsed = [p for p in parsed if p is not None]
        if not parsed:
            continue
        row = {"key": key, "target": target, "folds": len(parsed),
               "estimated": all(p["estimated"] for p in parsed)}
        if row["estimated"]:
            row["n_fit_min"] = min(p["n_fit"] for p in parsed)
            for m in MODES:
                sfx = "" if m == "pooled" else "_tree"
                for part in PARTS:
                    name = "selection_est" if part == "selection" else part
                    vals = [100 * p["parts"]["held_out"][m][part] / p["reported"]
                            if p["reported"] else None for p in parsed]
                    row[f"{name}{sfx}"], row[f"{name}{sfx}_se"], _ = _mean_se(vals)
            if all("production" in p["parts"] for p in parsed):
                for part in PARTS:
                    name = "selection_est" if part == "selection" else part
                    vals = [100 * p["parts"]["production"]["pooled"][part] / p["drawn"]
                            if p["drawn"] else None for p in parsed]
                    row[f"prod_{name}"], row[f"prod_{name}_se"], _ = _mean_se(vals)
                    vals = [100 * p["parts"]["production"]["pooled"][part] / p["honest_prod"]["pooled"]
                            if p["honest_prod"]["pooled"] else None for p in parsed]
                    row[f"prod_{name}_vs_honest"], row[f"prod_{name}_vs_honest_se"], _ = _mean_se(vals)
        rows.append(row)
    return rows


def table() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(ROOT / f"results/selection_*{TAG}.json"))):
        key = Path(path).stem.removeprefix("selection_").removesuffix(TAG)
        doc = json.loads(Path(path).read_text())
        if len(doc.get("folds", [])) == 5:
            rows.extend(summarize_key(key, doc))
    df = pd.DataFrame(rows)
    if len(df) and "selection_est" in df:
        est, se = df["selection_est"].abs(), df["selection_est_se"]
        df["material_selection_est"] = df["estimated"] & (est > se) & (est > 0.5)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/smoothing.csv")
    args = ap.parse_args()
    df = table()
    df.to_csv(args.out, index=False)
    print(f"{len(df)} targets across {df['key'].nunique() if len(df) else 0} fits")


if __name__ == "__main__":
    main()
