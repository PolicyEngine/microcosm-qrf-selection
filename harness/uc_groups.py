"""Household groups for the UC reporter redraw's training benefit units.

``capture`` records the training table's index (``__source_index.npy``); for
this fit it indexes the stage-30 frame's benefit-unit table. This script maps
each training unit to its household and checks the mapping: the training
target must equal the unit's reported UC summed from its members.

Usage: python uc_groups.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import capture_uk_release as R  # noqa: E402
import capture  # noqa: E402

KEY = "uk_uc_reporter_redraw"


def main() -> None:
    frame, _ = R.stage30_frame()
    benunit = frame.table("benunit")
    person = frame.table("person")
    idx = np.load(capture.DONORS / f"{KEY}__source_index.npy", allow_pickle=False)
    table = pd.read_parquet(capture.DONORS / f"{KEY}.parquet")
    units = benunit["benunit_id"].to_numpy()[idx]
    hh_of_unit = person.drop_duplicates("person_benunit_id").set_index(
        "person_benunit_id")["person_household_id"]
    groups = hh_of_unit.reindex(units).to_numpy()
    assert not pd.isna(groups).any()
    uc = pd.to_numeric(person["universal_credit_reported"], errors="coerce").fillna(0.0)
    uc_unit = uc.groupby(person["person_benunit_id"].to_numpy()).sum().reindex(units)
    target = table["universal_credit_reported_amount"].to_numpy(dtype=float)
    match = float(np.mean(np.isclose(uc_unit.to_numpy(dtype=float), target, atol=1e-6)))
    print(f"training units {len(idx)}; households {len(np.unique(groups))}; "
          f"target equals members' reported UC on {match:.4f} of units")
    if match < 0.99:
        raise SystemExit("the source index does not map to the benefit-unit table")
    np.save(capture.DONORS / f"{KEY}__groups.npy", groups.astype(np.int64),
            allow_pickle=False)


if __name__ == "__main__":
    main()
