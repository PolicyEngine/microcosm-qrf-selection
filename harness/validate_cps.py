"""Check the generic harness against the earlier CPS MOOP same-tree test.

Runs ``selection.run_stage`` on the inputs, folds and seeds
``qrf-shortfall-2026-11-07/harness/microcosm_cps.py`` used and compares every
total with ``results/microcosm_cps.json`` there: the half-sample adaptive and
honest totals in both modes, and the fold's own Microcosm forest read as the
pool mean (``own_gate_pool_mean``).

Usage:
  python validate_cps.py --out ../results/validate_cps.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import selection as S  # noqa: E402

PRIOR = Path.home() / "reviews/qrf-shortfall-2026-11-07"
SEED = 20261105  # harness/cps.py SEED


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = np.load(PRIOR / "results/cps_inputs.npz", allow_pickle=False)
    X = pd.DataFrame(data["X"], columns=[str(c) for c in data["columns"]])
    table = X.copy()
    table["moop"] = data["moop"]
    predictors = list(X.columns)
    doc = S.run_stage("cps_moop", table, predictors, ["moop"], data["weight"],
                      seed=SEED, n_estimators=100, out_path=args.out,
                      fold_ids=data["folds"], fold_seeds=[SEED + k for k in range(5)])
    prior = {r["fold"]: r for r in json.loads((PRIOR / "results/microcosm_cps.json").read_text())}
    worst = 0.0
    for f in doc["folds"]:
        sign = f["targets"]["moop"]["signs"]["1"]
        tot = sign["variants"]["1"]["totals"]
        ref = prior[f["fold"]]
        pairs = {
            "ahalf/pooled": tot["adaptive"]["pooled"], "ahalf/tree": tot["adaptive"]["tree"],
            "honest/pooled": tot["honest"]["pooled"], "honest/tree": tot["honest"]["tree"],
        }
        for key, mine in pairs.items():
            rel = abs(mine / ref["totals"][key] - 1)
            worst = max(worst, rel)
            print(f"fold {f['fold']} {key}: {mine:.6e} vs {ref['totals'][key]:.6e} (rel {rel:.1e})")
        rel = abs(sign["own_forest"]["pooled"] / ref["own_gate_pool_mean"] - 1)
        rel2 = abs(sign["own_forest_pool_mean_check"] / ref["own_gate_pool_mean"] - 1)
        worst = max(worst, rel, rel2)
        print(f"fold {f['fold']} own forest pooled fill vs saved pool mean: rel {rel:.1e}; "
              f"its predict(quantiles='mean'): rel {rel2:.1e}")
    print(f"largest relative difference: {worst:.1e}")
    summary = S.summarize(doc)
    print(summary.T.to_string())


if __name__ == "__main__":
    main()
