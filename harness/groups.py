"""Household ids for the captured US donor tables, so folds and halves keep households whole.

The production loaders drop their survey ids before the fit, so each captured
donor table carries none. This module recovers them and checks the alignment
row by row against the captured table before saving
``donors/<key>__groups.npy`` (record-level ids: local only).

- SCF (net worth, financial assets, auto loans): every household appears as
  five implicate rows. ``load_scf_2022_financial_asset_donor`` keeps the
  summary extract's row order and drops non-positive weights
  (``scf_wealth.py:560-608``); ``load_scf_2022_auto_loan_donor`` restores the
  summary order after its join and applies the same filter
  (``scf_auto_loans.py:218-318``). The group is the extract's ``yy1``.
- SIPP financial assets and tips: December person rows, several per household.
  The loaders' own calls run under a recorder that keeps the raw table they
  build (``pd.concat`` / ``pd.read_csv``) and the row labels each
  ``reset_index(drop=True)`` keeps; composing them gives each donor row's raw
  row, whose ``SSUID`` is the group. A check on age confirms the alignment. Each fit trains on a known subset of the loader's rows:
  the target's observed rows for financial assets
  (``sipp_financial_assets.py:676-688``), and a seeded subsample for tips when
  the donor exceeds its cap (``sipp_tips.py:416-423``).
- SIPP vehicles: one row per household already (``sipp_vehicles.py:430-455``).
- CPS ORG: the pinned wage file carries no household or person ids, so ORG
  folds and halves split rows.

Usage: python groups.py
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prodpath  # noqa: E402

prodpath.us()

import capture  # noqa: E402
import capture_us as C  # noqa: E402


@contextlib.contextmanager
def recording_pandas():
    """Record pd.concat / pd.read_csv results and pre-reset indexes (drop=True)."""
    log = {"concat": [], "read_csv": [], "reset": []}
    real_concat, real_read, real_reset = pd.concat, pd.read_csv, pd.DataFrame.reset_index

    def concat(*a, **k):
        out = real_concat(*a, **k)
        log["concat"].append(out)
        return out

    def read_csv(*a, **k):
        out = real_read(*a, **k)
        if isinstance(out, pd.DataFrame):
            log["read_csv"].append(out)
        return out

    def reset_index(self, *a, **k):
        if k.get("drop"):
            log["reset"].append(self.index.copy())
        return real_reset(self, *a, **k)

    pd.concat, pd.read_csv, pd.DataFrame.reset_index = concat, read_csv, reset_index
    try:
        yield log
    finally:
        pd.concat, pd.read_csv, pd.DataFrame.reset_index = real_concat, real_read, real_reset


def compose_resets(n_raw: int, resets: list, n_out: int) -> np.ndarray:
    """Raw-table position of each output row, through every recorded reset in order.

    Each ``reset_index(drop=True)`` renumbers rows 0..n-1, so the labels the
    next reset keeps are positions in the previous output. The SIPP asset
    loader resets twice: after its weight filter and after its
    target-balanced cap to 20,000 rows (``sipp_financial_assets.py:494-509``).
    """
    pos = np.arange(n_raw)
    for ix in resets:
        pos = pos[np.asarray(ix, dtype=np.int64)]
    if len(pos) != n_out:
        raise AssertionError(f"reset chain ends at {len(pos)} rows, loader returned {n_out}")
    return pos


def check_rows(captured: pd.DataFrame, source: pd.DataFrame, label: str) -> None:
    cols = [c for c in captured.columns if c != "__weight__"]
    a = captured.loc[:, cols].to_numpy(dtype=np.float64)
    b = source.loc[:, cols].to_numpy(dtype=np.float64)
    if a.shape != b.shape or not np.array_equal(a, b):
        raise AssertionError(f"{label}: captured table and loader rows differ")
    print(f"  {label}: {len(a)} rows match the loader row for row", flush=True)


def scf_groups() -> dict[str, np.ndarray]:
    from populace.build.us_runtime import scf_wealth as M

    raw = pd.read_stata(C.SCF_SUMMARY, convert_categoricals=False)
    pred = M._scf_summary_predictor_table(raw)
    keep = pred[M._DONOR_WEIGHT_COLUMN].to_numpy(dtype=np.float64) > 0
    yy1 = raw["yy1"].to_numpy()[keep]
    loader = M.load_scf_2022_financial_asset_donor(C.SCF_SUMMARY)
    if not np.array_equal(pd.to_numeric(raw["networth"], errors="coerce").fillna(0.0)
                          .to_numpy()[keep], loader["net_worth"].to_numpy()):
        raise AssertionError("SCF yy1 alignment failed on net worth")
    out = {}
    for key in ("scf_net_worth", "scf_financial_assets", "scf_auto_loans"):
        table, _ = capture.load(key)
        preds = [c for c in M.SCF_WEALTH_PREDICTORS]
        check_rows(table.loc[:, preds], loader.loc[:, preds], key)
        out[key] = yy1
    return out


def sipp_financial_asset_groups() -> dict[str, np.ndarray]:
    from populace.build.us_runtime import sipp_financial_assets as M

    with recording_pandas() as log:
        donor = M.load_sipp_2023_financial_asset_donor(
            C.SIPP, expected_sha256=M.SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256,
            expected_size_bytes=M.SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES)
    raw = next(t for t in reversed(log["concat"]) if "SSUID" in t.columns)
    kept = compose_resets(len(raw), log["reset"], len(donor))
    ssuid = raw["SSUID"].to_numpy()[kept]
    if not np.array_equal(pd.to_numeric(raw["TAGE"]).to_numpy()[kept],
                          donor["age"].to_numpy(dtype=np.float64)):
        raise AssertionError("SIPP financial-asset SSUID alignment failed on age")
    out = {}
    prior: list[str] = []
    for i, target in enumerate(M.SIPP_FINANCIAL_ASSET_OUTPUT_COLUMNS):
        key = f"sipp_financial_assets__{i}"
        table, meta = capture.load(key)
        assert meta["targets"] == [target], (key, meta["targets"])
        observed = donor[f"{target}{M._OBSERVED_SUFFIX}"].astype(bool).to_numpy()
        cols = [*M.SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS, *prior, target]
        check_rows(table.loc[:, cols], donor.loc[observed, cols].reset_index(drop=True), key)
        out[key] = ssuid[observed]
        prior.append(target)
    return out


def sipp_tip_groups() -> dict[str, np.ndarray]:
    from populace.build.us_runtime import sipp_tips as M

    with recording_pandas() as log:
        donor = M.load_sipp_2023_tip_donor(C.SIPP_SLIM, expected_sha256=M.SIPP_2023_TIP_DONOR_SHA256)
    raw = next(t for t in reversed(log["read_csv"]) if "SSUID" in t.columns)
    assert isinstance(raw.index, pd.RangeIndex)
    kept = compose_resets(len(raw), log["reset"], len(donor))
    ssuid = raw["SSUID"].astype(str).to_numpy()[kept]
    if not np.array_equal(pd.to_numeric(raw["TAGE"], errors="coerce").fillna(0.0)
                          .to_numpy()[kept], donor["age"].to_numpy(dtype=np.float64)):
        raise AssertionError("SIPP tip SSUID alignment failed on age")
    rows = np.arange(len(donor))
    if len(donor) > M._MAX_TRAIN_SAMPLES:
        rows = np.sort(np.random.default_rng(M._TIP_TRAINING_SAMPLE_SEED).choice(
            len(donor), size=M._MAX_TRAIN_SAMPLES, replace=False))
    table, _ = capture.load("sipp_tips")
    cols = [*M.SIPP_TIP_PREDICTORS, "tip_income"]
    src = donor.loc[:, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    check_rows(table.loc[:, cols], src.iloc[rows].reset_index(drop=True), "sipp_tips")
    return {"sipp_tips": ssuid[rows]}


def main() -> None:
    names = sys.argv[1:] or ["scf_groups", "sipp_financial_asset_groups", "sipp_tip_groups"]
    path = C.ROOT / "results/groups_summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for fn in (globals()[n] for n in names):
        print(fn.__name__, flush=True)
        for key, groups in fn().items():
            np.save(capture.DONORS / f"{key}__groups.npy", np.asarray(groups).astype(str))
            codes = pd.Series(groups).value_counts()
            summary[key] = {"rows": int(len(groups)), "groups": int(len(codes)),
                            "max_rows_per_group": int(codes.max())}
            print(f"  {key}: {summary[key]}", flush=True)
    path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
