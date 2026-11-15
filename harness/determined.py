"""Find targets that their chained features determine exactly, per sign.

Microcosm fits target ``k`` of a chain on its predictors plus every earlier
target (``qrf.py`` ``RegimeGatedQRF.fit``: ``chained = (*predictors,
*targets[:position])``), and grows one amount forest per sign on the donors of
that sign. When a sign's donor amounts are an exact linear function of the
chained features, every donor's residual about its conditional mean is zero,
so that sign's selection is zero by construction. Its estimated selection then
measures only the fitted mean's error, which makes it a control for the
estimator in ``selection.contrast_parts``.

The scan covers exact linear functions with an intercept, fitted by least
squares on each sign's donors. On one sign's donors that includes a target
equal to a feature, a difference or sum of features (non-qualified dividends
are the dividend predictor less qualified dividends), and a positive or
negative part of either. It does not search nonlinear functions.

A sign counts as determined when its largest absolute residual is at most
``REL_TOL`` of its largest absolute amount and it has at least ``MIN_DOF``
more donors than fitted columns, so an exact fit cannot come from having as
many columns as donors. On the 45 fits no residual falls between 1e-12 and
1e-4 of the largest amount, so the tolerance decides no borderline case.

Usage:
  python determined.py results/smoothing_plan.txt --out results/determined.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import capture  # noqa: E402

REL_TOL = 1e-8
#: Residual degrees of freedom an exact fit needs before it counts.
MIN_DOF = 20
#: A column whose scaled coefficient is at least this share of the largest
#: one is named in the identity.
NAME_SHARE = 1e-6


def sign_fit(X: np.ndarray, y: np.ndarray, columns: list[str]) -> dict:
    """Least-squares fit of ``y`` on ``[1, X]`` with columns scaled to unit size."""
    scale = np.max(np.abs(X), axis=0)
    scale[scale == 0] = 1.0
    A = np.column_stack([np.ones(len(y)), X / scale])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    y_max = float(np.max(np.abs(y)))
    rel = float(np.max(np.abs(resid)) / y_max) if y_max > 0 else 0.0
    raw = coef[1:] / scale
    weight = np.abs(coef[1:])
    top = weight.max() if len(weight) else 0.0
    named = [f"{raw[j]:+.6g}*{columns[j]}" for j in np.argsort(-weight)
             if top > 0 and weight[j] >= NAME_SHARE * top]
    intercept = float(coef[0])
    return {"max_rel_resid": rel, "intercept": intercept, "identity": " ".join(named[:6])}


def scan_key(key: str) -> list[dict]:
    table, meta = capture.load(key)
    return scan_table(key, table, list(meta["predictors"]), list(meta["targets"]),
                      float(meta.get("zero_atol") or 1e-6))


def scan_table(key: str, table: pd.DataFrame, predictors: list[str], targets: list[str],
               atol: float) -> list[dict]:
    """Every target-sign of one chain, with whether its chained features fix it."""
    rows = []
    for position, target in enumerate(targets):
        columns = [*predictors, *targets[:position]]
        y_all = table[target].to_numpy(dtype=np.float64)
        X_all = table[columns].to_numpy(dtype=np.float64)
        for sign in (1, -1):
            mask = y_all > atol if sign == 1 else y_all < -atol
            keep = mask & np.isfinite(y_all) & np.all(np.isfinite(X_all), axis=1)
            n, p = int(keep.sum()), len(columns) + 1
            row = {"key": key, "target": target, "sign": sign, "n": n, "n_columns": p,
                   "n_dropped_nonfinite": int(mask.sum() - n)}
            if n < 2:
                rows.append({**row, "status": "too_few_donors"})
                continue
            y = y_all[keep]
            if np.all(y == y[0]):
                rows.append({**row, "status": "constant", "max_rel_resid": 0.0})
                continue
            fit = sign_fit(X_all[keep], y, columns)
            enough = n >= p + MIN_DOF
            exact = fit["max_rel_resid"] <= REL_TOL
            status = ("determined" if exact and enough
                      else "exact_but_few_donors" if exact else "not_determined")
            rows.append({**row, **fit, "status": status})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    keys = [line.split()[0] for line in Path(args.plan).read_text().splitlines() if line.strip()]
    rows = []
    for key in keys:
        found = scan_key(key)
        hits = [r for r in found if r["status"] in ("determined", "exact_but_few_donors")]
        print(f"{key}: {len(found)} target-signs, {len(hits)} exact", flush=True)
        for r in hits:
            print(f"  {r['target']} sign {r['sign']:+d} n={r['n']} p={r['n_columns']} "
                  f"rel={r['max_rel_resid']:.2e} {r['status']}: {r['identity']}", flush=True)
        rows += found
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
