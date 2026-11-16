"""Summarize the remedy runs: what each remedy does to the total and the fit.

Reads every ``results/selection_<key>_remedies.json`` (``run_remedies.sh``
writes them: variants ``min_samples_leaf`` 1, 5 and 20, with quantile
metrics). Each fold holds, for every target and sign forest, totals over the
held-out recipients (and over the production recipients, where the stage has
them) for each variant x fill (adaptive: the trees' own in-bag rows; honest:
the other half's donors) x mode (pooled: count pooling, which is how
Microcosm's ``draw`` reads the forest, ``quantile_forest`` ``predict`` with its
default ``weighted_leaves=False``; tree: per-tree leaf weights, which is
``predict(weighted_leaves=True)``).

Production's configuration is variant 1, adaptive, pooled. For every other
configuration this reports, per target and as a mean over folds with its
standard error:

- ``effect``: the configuration's total minus production's, as % of the
  held-out reported total (same trees, same halves, same recipients);
- ``residual_selection``: adaptive minus honest under the configuration's
  variant and mode, as % of the held-out reported total;
- ``vs_reported``: the configuration's total against the held-out reported
  total, % (this carries the gate's error and the half-sample forests' own
  error, which every configuration shares);
- ``prod_effect``: the same shift over the production recipients, as % of
  their drawn total;
- ``pinball_change``: the change in mean pinball loss (taus 0.1-0.9) of the
  positive forest's conditional quantiles on the held-out positive donors,
  relative to production's configuration, %; and ``coverage_error``, the
  mean absolute gap between each tau and the share of held-out donors at or
  below that quantile.

Usage: python remedies.py [--out ../results/remedies.csv]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODES = ("pooled", "tree")
FILLS = ("adaptive", "honest")
BASE = ("1", "adaptive", "pooled")


def _mean_se(vals):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    if len(vals) == 0:
        return float("nan"), float("nan"), 0
    se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
    return float(vals.mean()), se, int(len(vals))


def _fold_target(t):
    """Per-configuration totals for one target in one fold, signs summed."""
    signs = t.get("signs") or {}
    if not signs or any("skipped" in s for s in signs.values()):
        return None
    first = next(iter(signs.values()))
    variants = list(first["variants"])
    tot, prod, quant = {}, {}, {}
    has_prod = all("extra" in s for s in signs.values())
    for v in variants:
        for f in FILLS:
            for m in MODES:
                tot[(v, f, m)] = sum(s["variants"][v]["totals"][f][m] for s in signs.values())
                if has_prod:
                    prod[(v, f, m)] = sum(s["extra"]["production"]["variants"][v]["totals"][f][m]
                                          for s in signs.values())
                pos = signs.get("1", {})
                q = (pos.get("variants", {}).get(v, {}).get("quantile") or {}).get(f, {}).get(m)
                quant[(v, f, m)] = q
    return {"variants": variants, "tot": tot, "prod": prod if has_prod else None,
            "quant": quant, "reported": t["reported"],
            "drawn": (t.get("production") or {}).get("drawn_total")}


def summarize_key(key: str, doc: dict) -> list[dict]:
    rows = []
    for target in doc["targets"]:
        folds = [f["targets"].get(target) for f in doc["folds"]]
        folds = [t for t in folds if t is not None]
        if not folds:
            continue
        regime = folds[0].get("regime")
        parsed = [_fold_target(t) for t in folds]
        parsed = [p for p in parsed if p is not None]
        if not parsed:
            rows.append({"key": key, "target": target, "regime": regime, "status":
                         "not_measured"})
            continue
        variants = parsed[0]["variants"]
        for v in variants:
            for f in FILLS:
                for m in MODES:
                    cfg = (v, f, m)
                    row = {"key": key, "target": target, "regime": regime,
                           "folds": len(parsed), "min_samples_leaf": int(v), "fill": f,
                           "mode": m, "production_config": cfg == BASE}

                    def pct(fn):
                        vals = []
                        for p in parsed:
                            try:
                                with np.errstate(divide="ignore", invalid="ignore"):
                                    vals.append(fn(p))
                            except (ZeroDivisionError, TypeError, KeyError):
                                vals.append(None)
                        return _mean_se(vals)

                    def rep(p):
                        return p["reported"] if p["reported"] else float("nan")

                    row["effect"], row["effect_se"], _ = pct(
                        lambda p: 100 * (p["tot"][cfg] - p["tot"][BASE]) / rep(p))
                    row["residual_selection"], row["residual_selection_se"], _ = pct(
                        lambda p: 100 * (p["tot"][(v, "adaptive", m)]
                                         - p["tot"][(v, "honest", m)]) / rep(p))
                    row["vs_reported"], row["vs_reported_se"], _ = pct(
                        lambda p: 100 * (p["tot"][cfg] / rep(p) - 1))
                    if all(p["prod"] is not None and p["drawn"] for p in parsed):
                        row["prod_effect"], row["prod_effect_se"], _ = pct(
                            lambda p: 100 * (p["prod"][cfg] - p["prod"][BASE]) / p["drawn"])
                        row["prod_residual_selection"], row["prod_residual_selection_se"], _ = pct(
                            lambda p: 100 * (p["prod"][(v, "adaptive", m)]
                                             - p["prod"][(v, "honest", m)]) / p["drawn"])
                    row["pinball_change"], row["pinball_change_se"], row["pinball_folds"] = pct(
                        lambda p: 100 * (p["quant"][cfg]["mean_pinball"]
                                         / p["quant"][BASE]["mean_pinball"] - 1))
                    row["coverage_error"], row["coverage_error_se"], _ = pct(
                        lambda p: float(np.mean([abs(c - float(tau)) for tau, c in
                                                 p["quant"][cfg]["coverage"].items()])))
                    row["reported_total_mean"] = float(np.mean([p["reported"] for p in parsed]))
                    rows.append(row)
    return rows


def table() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "results/selection_*_remedies.json"))):
        key = Path(path).stem.removeprefix("selection_").removesuffix("_remedies")
        doc = json.loads(Path(path).read_text())
        if not doc.get("folds"):
            continue
        rows.extend(summarize_key(key, doc))
    return pd.DataFrame(rows)


REMEDIES = {
    "honest fills": ("1", "honest", "pooled"),
    "min_samples_leaf=5": ("5", "adaptive", "pooled"),
    "min_samples_leaf=20": ("20", "adaptive", "pooled"),
    "weighted_leaves=True": ("1", "adaptive", "tree"),
}


def compact(df: pd.DataFrame) -> pd.DataFrame:
    """One row per target: production's selection, then each remedy's effect."""
    out = []
    for (key, target), g in df.dropna(subset=["fill"]).groupby(["key", "target"], sort=False):
        base = g.loc[g["production_config"]].iloc[0]
        row = {"key": key, "target": target, "folds": int(base["folds"]),
               "selection": base["residual_selection"],
               "selection_se": base["residual_selection_se"],
               "coverage_error": base["coverage_error"],
               "vs_reported": base["vs_reported"]}
        for name, (v, f, m) in REMEDIES.items():
            r = g.loc[(g["min_samples_leaf"] == int(v)) & (g["fill"] == f)
                      & (g["mode"] == m)].iloc[0]
            row[f"{name}: effect"] = r["effect"]
            row[f"{name}: effect_se"] = r["effect_se"]
            row[f"{name}: residual"] = (r["residual_selection"] if f == "adaptive"
                                        else 0.0)
            row[f"{name}: pinball"] = r["pinball_change"]
            row[f"{name}: pinball_se"] = r["pinball_change_se"]
            row[f"{name}: coverage_error"] = r["coverage_error"]
            row[f"{name}: vs_reported"] = r["vs_reported"]
            if "prod_effect" in r and pd.notna(r.get("prod_effect")):
                row[f"{name}: prod_effect"] = r["prod_effect"]
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/remedies.csv")
    args = ap.parse_args()
    df = table()
    df.to_csv(args.out, index=False)
    c = compact(df)
    c.to_csv(args.out.with_name("remedies_compact.csv"), index=False)
    cols = ["key", "target", "selection", "selection_se"] + [
        f"{n}: {k}" for n in REMEDIES for k in ("effect", "residual", "pinball")]
    with pd.option_context("display.width", 300, "display.max_rows", 500,
                           "display.max_columns", 40):
        print(c[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


if __name__ == "__main__":
    main()
