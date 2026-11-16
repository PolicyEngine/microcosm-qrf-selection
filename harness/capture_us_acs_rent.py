"""Capture the US ACS rent QRF fit with the base builder's code at 6d06410bb.

The base builder restores ``pre_subsidy_rent`` before support cloning with
``with_us_housing_inputs`` (``tools/build_us_puf_support_base.py:880-895`` at
6d06410bb), which fits one ``QRF`` (``RegimeGatedQRF``) on a seeded sample of
the pinned ACS 2022 donor and draws rent once per CPS household head
(``housing_inputs.py:815-890``: 100 trees, the build seed 0, weights the ACS
``household_weight``). The clone then copies each household's rent to its PUF
half unchanged (``housing_inputs.py:974-990``).

This script loads the same pinned donor (``acs_2022.h5``, sha256 0b319b49...,
checked by ``load_acs_2022_rent_donor``) and runs
``impute_us_pre_subsidy_rent`` unchanged under ``capture.recording()``. The
training sample is a deterministic function of the donor
(``_archived_joint_training_sample``), so the captured donor table is
production's own.

Recipients: the release's household heads with the head features
``_recipient_head_features`` builds from the released frame, weighted by the
release household weight. The base pool's heads are gone, and later stages
edit some head features, so the recipient set is a reconstruction; the check
compares the reconstructed draws' weighted total with the released
``pre_subsidy_rent`` total on the same heads.

Usage: python capture_us_acs_rent.py --out ../results/capture_us_acs_rent.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from capture_us_puf_chain import release_frame, sha256  # noqa: E402  (sets 6d06410bb)

import capture  # noqa: E402

ACS_H5 = Path.home() / ("PolicyEngine/_recovered/scratch-backup/893/route-a-rerun-20261018/"
                        "inputs/storage/acs_2022.h5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    from populace.build.us_runtime import housing_inputs as H

    donor = H.load_acs_2022_rent_donor(ACS_H5)
    tables, frame = release_frame()
    with capture.recording() as caps:
        rent = H.impute_us_pre_subsidy_rent(frame, donor, seed=0,
                                           n_estimators=H._DEFAULT_N_ESTIMATORS)
    assert len(caps) == 1 and len(caps[0].recipients) == 1, (len(caps),)
    spec = caps[0]
    rec = spec.recipients[0]
    person = tables["person"]
    _, head_mask = H._recipient_head_features(frame)
    heads = person.loc[head_mask]
    assert len(heads) == len(rec["table"])
    weight = pd.Series(tables["household"]["household_weight"].to_numpy(dtype=np.float64),
                       index=tables["household"]["household_id"].to_numpy()).reindex(
        heads["person_household_id"].to_numpy()).to_numpy(dtype=np.float64)
    drawn = np.maximum(rec["table"]["rent"].to_numpy(dtype=np.float64), 0.0)
    released = pd.to_numeric(heads["pre_subsidy_rent"], errors="coerce").fillna(0.0) \
        .to_numpy(dtype=np.float64)
    check = {"rent": {
        "reconstructed_total": float((weight * drawn).sum()),
        "production_total": float((weight * released).sum()),
        "positive_share_reconstructed": float(np.average(drawn > 0, weights=weight)),
        "positive_share_production": float(np.average(released > 0, weights=weight)),
        "share_exact": float(np.mean(drawn == released))}}
    # Chained columns: none (one target). The recipient table's target column
    # holds the released value, the value production drew for these heads.
    rec["table"]["rent"] = released
    rec.update({"weight": weight, "in_production": np.ones(len(heads), dtype=bool),
                "entity": "person (household head)", "method": "predict_reconstructed",
                "weight_source": "release household_weight of the head's household",
                "check": check})
    capture.save(spec, "acs_rent", {"stage": "acs_rent", "code": "6d06410bb",
                                    "acs_sha256": sha256(ACS_H5),
                                    "reconstructed_recipients": True})
    report = {**spec.meta(), "check": check, "acs_sha256": sha256(ACS_H5),
              "donor_weight_total": float(spec.weights.sum()), "seconds": time.time() - t0}
    args.out.write_text(json.dumps(report, indent=1, default=str))
    c = check["rent"]
    print(f"rent: reconstructed/released total {c['reconstructed_total'] / c['production_total']:.3f}; "
          f"positive share {c['positive_share_reconstructed']:.3f} vs "
          f"{c['positive_share_production']:.3f}; exact {c['share_exact']:.3f}", flush=True)
    print(f"captured acs_rent ({len(spec.table)} donors) in {report['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
