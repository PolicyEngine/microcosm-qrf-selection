"""Run the same-tree selection test on one captured production fit.

Loads ``donors/<key>`` (the donor table, weights and fit settings that
reproduce production) and, when present, its household groups
(``<key>__groups.npy``) and production recipients (``<key>__rec0.parquet``,
restricted to the rows whose draw reaches the released column). Then runs
``selection.run_stage`` with the fit's own seed and forest size on 5 folds and
writes ``results/selection_<key><tag>.json`` plus a summary CSV.

Usage:
  python run_selection.py scf_net_worth
  python run_selection.py scf_net_worth --variants 1 5 20 --quantiles --tag _remedies
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import capture  # noqa: E402
import selection as S  # noqa: E402

ROOT = HERE.parent


def load_inputs(key: str, production: bool = True):
    table, meta = capture.load(key)
    if not meta["weighted"]:
        raise ValueError(f"{key}: production fit was unweighted; the harness assumes weights")
    gpath = capture.DONORS / f"{key}__groups.npy"
    groups = np.load(gpath, allow_pickle=False) if gpath.exists() else None
    if groups is not None and len(groups) != len(table):
        raise ValueError(f"{key}: groups and donor rows differ in length")
    prod = None
    if production and meta.get("recipients"):
        if len(meta["recipients"]) != 1:
            raise ValueError(f"{key}: expected one predict call, found {len(meta['recipients'])}")
        rec = capture.load_recipients(key, 0)
        keep = rec["__in_production__"].to_numpy(dtype=bool)
        rec = rec.loc[keep].reset_index(drop=True)
        prod = (rec, rec["__weight__"].to_numpy(dtype=np.float64))
    return table, meta, groups, prod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--variants", type=int, nargs="+", default=[1])
    ap.add_argument("--quantiles", action="store_true")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--no-production", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="thin each fold's training donors to this share first")
    args = ap.parse_args()

    table, meta, groups, prod = load_inputs(args.key, not args.no_production)
    out = ROOT / f"results/selection_{args.key}{args.tag}.json"
    run_meta = {"key": args.key, "capture": {k: v for k, v in meta.items() if k != "recipients"},
                "groups": None if groups is None else {
                    "rows": int(len(groups)), "households": int(len(np.unique(groups)))},
                "production_recipients": None if prod is None else {
                    "n": int(len(prod[0])), "weight_total": float(prod[1].sum()),
                    **{k: meta["recipients"][0].get(k) for k in ("entity", "weight_source",
                                                                 "method")}}}
    print(json.dumps(run_meta, default=str), flush=True)
    doc = S.run_stage(
        args.key, table, meta["predictors"], meta["targets"], "__weight__",
        seed=int(meta["seed"]), n_estimators=int(meta["n_estimators"]), out_path=out,
        groups=groups, folds=args.folds, variants=tuple(args.variants),
        quantiles=args.quantiles, only=args.only,
        max_samples_leaf=meta.get("max_samples_leaf"), meta=run_meta, production=prod,
        train_fraction=args.train_fraction)
    for v in args.variants:
        summary = S.summarize(doc, variant=str(v))
        summary.to_csv(out.with_name(out.stem + f"_v{v}.csv"), index=False)
        print(f"variant min_samples_leaf={v}", flush=True)
        print(summary.T.to_string(), flush=True)


if __name__ == "__main__":
    main()
