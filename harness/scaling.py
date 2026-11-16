"""How selection moves when each fold's training donors halve again.

The same-tree test grows each forest on half of a sign's training donors, so
that an honest half remains. Production grows on all of them. This compares,
on the same held-out donors, the test as run (``selection_<key>.json``) with
the test after thinning every fold's training donors to half
(``selection_<key>_halved.json``, from ``run_selection.py --train-fraction
0.5 --tag _halved``). If selection shrinks as donors grow, the test's
half-sample forests overstate what production's full forests carry; if it
holds steady, they do not.

Writes ``results/scaling.csv``: one row per target, with selection at each
size (% of the held-out total), and ``ratio`` = selection on the thinned
donors over selection as run.

Usage: python scaling.py [--out ../results/scaling.csv]
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

import selection as S  # noqa: E402

ROOT = HERE.parent
TAG = "_halved"


def table() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(str(ROOT / f"results/selection_*{TAG}.json"))):
        key = Path(path).stem.removeprefix("selection_").removesuffix(TAG)
        thin_doc = json.loads(Path(path).read_text())
        base_path = ROOT / f"results/selection_{key}.json"
        if not thin_doc.get("folds") or not base_path.exists():
            continue
        base = S.summarize(json.loads(base_path.read_text()), variant="1").set_index("target")
        thin = S.summarize(thin_doc, variant="1")
        thin = thin.loc[thin["regime"] != "not_run"].set_index("target")
        for target, r in thin.iterrows():
            b = base.loc[target]
            ratio = (r["sel_pooled"] / b["sel_pooled"]
                     if b["sel_pooled"] and np.isfinite(b["sel_pooled"]) else float("nan"))
            rows.append({"key": key, "target": target, "folds": int(r["folds"]),
                         "selection": b["sel_pooled"], "selection_se": b["sel_pooled_se"],
                         "selection_thinned": r["sel_pooled"],
                         "selection_thinned_se": r["sel_pooled_se"], "ratio": ratio})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/scaling.csv")
    args = ap.parse_args()
    df = table()
    df.to_csv(args.out, index=False)
    ratio = df["ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"{len(df)} targets across {df['key'].nunique()} fits")
    print(f"ratio of selection on thinned donors to selection as run: median {ratio.median():.2f}, "
          f"quartiles {ratio.quantile(.25):.2f} to {ratio.quantile(.75):.2f}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
