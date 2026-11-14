"""Production recipients for the PUF tax-detail chain, rebuilt from ASEC twins.

``capture_us_puf_chain.py`` builds the chain's production recipients from the
release's PUF-channel tax units, with the 8 predictors that
``_tax_unit_feature_frame`` (``puf_support.py:1169-1196`` at 6d06410bb)
computes from the released frame. That function reads the income predictors
from the person columns the chain itself writes (``employment_income_before_lsr``
for ``employment_income`` through ``_PREDICTOR_LEAF_ALIASES``,
``puf_support.py:339-352``; ``taxable_interest_income``; the dividend leaves),
so on a PUF-channel unit of the release they hold the chain's draws, not the
CPS values the chain conditioned on in the base build.

Support cloning copies each base household into an ASEC half and a PUF half
that share ``*_source_id`` (the release's ``tax_unit_source_id`` is unique
within each channel). Where calibration kept both halves, the ASEC twin still
carries the CPS values: the chain draws only for the PUF half. This script
takes the 8 predictors from the twin, the chained target columns from the PUF
unit's released values (the production draws, after finalization), and the
weight from the PUF unit's household. It covers only PUF units whose twin the
release kept; the coverage and the subset's share of each released total are
written to the report so the subset's representativeness is visible.

The donor table is production's own (hard-linked from ``donors/puf_chain``).

Usage: python puf_chain_twin.py --out ../results/capture_us_puf_chain_twin.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from capture_us_puf_chain import release_frame  # noqa: E402  (sets 6d06410bb)

import capture  # noqa: E402

SRC, DST = "puf_chain", "puf_chain_twin"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from populace.build.us_runtime import puf_support as PS

    table, meta = capture.load(SRC)
    old = capture.load_recipients(SRC, 0)
    predictors, targets = list(meta["predictors"]), list(meta["targets"])
    tables, frame = release_frame()
    tu = tables["tax_unit"]
    channel = tu[PS.support_channel_column("tax_unit")].astype(str).to_numpy()
    features = PS._tax_unit_feature_frame(frame, predictors).reset_index(drop=True)
    puf_rows = np.flatnonzero(channel == PS.PUF_TAX_DETAIL_SUPPORT_CHANNEL)
    # The earlier capture's recipients are exactly these rows, in this order.
    assert np.array_equal(old["__index__"].to_numpy(), puf_rows)
    source = tu["tax_unit_source_id"].to_numpy()
    asec_rows = np.flatnonzero(channel != PS.PUF_TAX_DETAIL_SUPPORT_CHANNEL)
    asec_of = pd.Series(asec_rows, index=source[asec_rows])
    assert asec_of.index.is_unique and pd.Index(source[puf_rows]).is_unique
    twin = asec_of.reindex(source[puf_rows]).to_numpy()
    has = ~np.isnan(twin)
    keep = np.flatnonzero(has)
    twin_rows = twin[has].astype(np.int64)

    rec = old.iloc[keep].reset_index(drop=True).copy()
    released = rec[predictors].to_numpy(dtype=np.float64)  # features at released values
    rec[predictors] = features.iloc[twin_rows][predictors].to_numpy(dtype=np.float64)
    weight = rec["__weight__"].to_numpy(dtype=np.float64)
    all_w = old["__weight__"].to_numpy(dtype=np.float64)

    coverage = {
        "puf_units": int(len(puf_rows)), "with_asec_twin": int(len(keep)),
        "unit_share": float(len(keep) / len(puf_rows)),
        "weight_share": float(weight.sum() / all_w.sum()),
        "released_total_share": {
            t: (float((weight * rec[t].to_numpy()).sum()
                      / (all_w * old[t].to_numpy()).sum())
                if (all_w * old[t].to_numpy()).sum() else None)
            for t in targets},
        "predictor_mean_twin_over_released": {
            p: (float(np.average(rec[p].to_numpy(), weights=weight)
                      / np.average(released[:, i], weights=weight))
                if np.average(released[:, i], weights=weight) else None)
            for i, p in enumerate(predictors)},
    }
    spec = capture.CapturedSpec(
        table=table.drop(columns="__weight__"), predictors=predictors, targets=targets,
        weights=table["__weight__"].to_numpy(dtype=np.float64),
        n_estimators=int(meta["n_estimators"]), seed=int(meta["seed"]),
        zero_atol=float(meta["zero_atol"]), max_samples_leaf=meta["max_samples_leaf"],
        module=meta["module"])
    # Recipients: index = position in the release tax_unit table of the PUF unit.
    spec.recipients = [{
        "table": rec.drop(columns=[c for c in rec.columns if c.startswith("__")]),
        "index": puf_rows[keep], "entity": "tax_unit",
        "method": "chain_reconstructed_from_release_asec_twin_features",
        "weight": weight, "in_production": np.ones(len(rec), dtype=bool),
        "weight_source": "release household_weight of the PUF tax unit's household",
        "check": coverage}]
    extra = {k: v for k, v in meta.items()
             if k not in ("recipients", "predictors", "targets", "n_estimators", "seed",
                          "zero_atol", "max_samples_leaf", "module", "n_rows", "weighted")}
    extra["recipient_features"] = "ASEC twin (same tax_unit_source_id)"
    capture.save(spec, DST, extra)
    # The donor parquet duplicates SRC's: replace it with a hard link (no copy).
    src_p, dst_p = capture.DONORS / f"{SRC}.parquet", capture.DONORS / f"{DST}.parquet"
    a, b = pd.read_parquet(src_p), pd.read_parquet(dst_p)
    assert a.equals(b)
    dst_p.unlink()
    os.link(src_p, dst_p)
    args.out.write_text(json.dumps(coverage, indent=1))
    print(json.dumps({k: v for k, v in coverage.items() if k != "released_total_share"},
                     indent=1))
    top = sorted(coverage["released_total_share"].items(),
                 key=lambda kv: -abs(kv[1] or 0))[:12]
    print("released total share of the twin subset (first 12):", top)


if __name__ == "__main__":
    main()
