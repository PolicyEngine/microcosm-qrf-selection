"""Capture the US PUF tax-detail QRF chain with the base builder's code at 6d06410bb.

The US release's base pool (``base_h5_sha256`` 5d8a3162 in
``parent_build_manifest.json``; 337,704 candidate households) was built at
commit 6d06410bb with seed 0 and 32 trees (``BUILDM_STATE.json``
``build_commit``; ``buildm_base.sh`` passes ``--seed 0 --n-estimators 32``).
Its clone-feature stage builds the PUF donor with
``puf_tax_unit_donor_from_arrays`` on ``puf_2024.h5``
(``tools/build_us_puf_support_base.py:1810-1830``) and starts one QRF chain
over 64 targets (``puf_qrf_chain.py:68-160``: ``start_chain(donor_frame,
predictors, target_order, weights="design")``). The code between 6d06410bb
and the release commit cae8640 changed the donor (#515 E19200 carve, #516
outlier screen), so this script imports 6d06410bb.

It runs the same functions on the same PUF file (sha256 7669f5b5..., the
materialized pin) and captures the ``start_chain`` inputs: the donor table,
the 8 predictors, the 64 targets in production order (checked against
``PRIMARY_QRF_TARGET_ORDER_SHA256``) and the design weights. The donor table is
production's own.

Production recipients. The base pool no longer exists, so the chain's
recipients are rebuilt from the release: its PUF-channel tax units, with the
8 features computed by the chain's own ``_tax_unit_feature_frame``
(``puf_support.py:1169``) on the released frame, chained targets set to the
released values summed to the tax unit (``_person_tax_unit_sum``), and
weights the release household weight of each tax unit's household. Later
stages edit some feature columns and finalization post-processes the draws
(``finalize_us_puf_tax_detail_predictions``), so the recipient set is a
reconstruction.

Usage: python capture_us_puf_chain.py --out ../results/capture_us_puf_chain.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prodpath  # noqa: E402

prodpath.us("6d06410bb")

import capture  # noqa: E402

ROOT = HERE.parent
PUF_H5 = Path.home() / ("PolicyEngine/_recovered/scratch-backup/893/route-a-rerun-20261018/"
                        "inputs/storage/puf_2024.h5")
PUF_SHA256 = "7669f5b5"  # prefix of the materialized pin; full hash recorded in the report
RELEASE_H5 = ROOT / "data/prod/us/populace_us_2024.h5"
SEED, N_ESTIMATORS = 0, 32


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_h5_arrays(path: Path) -> dict:
    """``tools/build_us_puf_support_base.py:2523-2527`` at 6d06410bb."""
    import h5py

    with h5py.File(path, "r") as h5:
        return {name: np.asarray(dataset) for name, dataset in h5.items()}


def release_frame():
    from populace.frame import Frame, WeightKind, Weights
    from populace.frame.units import US_SCHEMA

    tables = {k: pd.read_hdf(RELEASE_H5, key=k) for k in
              ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")}
    t = {k: v.copy() for k, v in tables.items()}
    weights = t["household"].pop("household_weight").to_numpy(dtype=np.float64)
    return tables, Frame(t, US_SCHEMA, {"household": Weights(weights, WeightKind.CALIBRATED)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    from populace.build.us_runtime import puf_qrf_chain as PC
    from populace.build.us_runtime import puf_support as PS
    from populace.fit import QRF

    puf_sha = sha256(PUF_H5)
    assert puf_sha.startswith(PUF_SHA256), puf_sha
    donor_tax_units = PS.puf_tax_unit_donor_from_arrays(read_h5_arrays(PUF_H5))
    tables, frame = release_frame()
    inputs = PS.prepare_us_puf_tax_detail_chain_inputs(frame, donor_tax_units)
    order = inputs.target_order
    assert PC._ordered_strings_sha256(order) == PC.PRIMARY_QRF_TARGET_ORDER_SHA256, \
        "target order differs from production"
    with capture.recording() as caps:
        QRF(n_estimators=N_ESTIMATORS, seed=SEED).start_chain(
            inputs.donor_frame, list(inputs.predictors), list(order), weights="design")
    assert len(caps) == 1, len(caps)
    spec = caps[0]

    # Reconstructed production recipients: the release's PUF-channel tax units.
    tu = tables["tax_unit"]
    person = tables["person"]
    puf_rows = np.flatnonzero(tu[PS.support_channel_column("tax_unit")].to_numpy()
                              == PS.PUF_TAX_DETAIL_SUPPORT_CHANNEL)
    rec = inputs.recipient_features.reset_index(drop=True).copy()
    assert len(rec) == len(puf_rows)
    released = {}
    for target in inputs.person_outputs:
        released[target] = PS._person_tax_unit_sum(frame, target)[puf_rows]
    for target in inputs.tax_unit_outputs:
        released[target] = pd.to_numeric(tu[target], errors="coerce").fillna(0.0) \
            .to_numpy(dtype=np.float64)[puf_rows]
    for target in order:
        rec[target] = released[target]
    hh_of_tu = person.groupby("person_tax_unit_id")["person_household_id"].first()
    hh_weight = pd.Series(tables["household"]["household_weight"].to_numpy(dtype=np.float64),
                          index=tables["household"]["household_id"].to_numpy())
    weight = hh_weight.reindex(hh_of_tu.reindex(
        tu["tax_unit_id"].to_numpy()[puf_rows]).to_numpy()).to_numpy(dtype=np.float64)
    assert np.isfinite(weight).all() and (weight > 0).all()
    donor_w = spec.weights
    check = {t: {"donor_weighted_total": float(np.sum(donor_w * spec.table[t].to_numpy())),
                 "released_puf_half_total": float(np.sum(weight * rec[t].to_numpy()))}
             for t in order}
    spec.recipients = [{"table": rec, "index": puf_rows, "entity": "tax_unit",
                        "method": "chain_reconstructed_from_release", "weight": weight,
                        "in_production": np.ones(len(rec), dtype=bool),
                        "weight_source": "release household_weight of the tax unit's household",
                        "check": check}]
    capture.save(spec, "puf_chain", {
        "stage": "puf_tax_detail", "code": "6d06410bb", "puf_sha256": puf_sha,
        "target_order_sha256": PC.PRIMARY_QRF_TARGET_ORDER_SHA256,
        "reconstructed_recipients": True})
    report = {"puf_sha256": puf_sha, "donor_rows": int(len(spec.table)),
              "donor_weight_total": float(donor_w.sum()), "predictors": spec.predictors,
              "targets": spec.targets, "n_estimators": spec.n_estimators, "seed": spec.seed,
              "max_samples_leaf": str(spec.max_samples_leaf), "zero_atol": spec.zero_atol,
              "recipients": int(len(rec)), "recipient_weight_total": float(weight.sum()),
              "check": check, "seconds": time.time() - t0}
    args.out.write_text(json.dumps(report, indent=1, default=str))
    big = sorted(check.items(), key=lambda kv: -abs(kv[1]["released_puf_half_total"]))[:20]
    for t, c in big:
        print(f"{t}: donor {c['donor_weighted_total']:.4g}  released PUF half "
              f"{c['released_puf_half_total']:.4g}", flush=True)
    print(f"captured puf_chain ({len(spec.table)} donors, {len(order)} targets) in "
          f"{report['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
