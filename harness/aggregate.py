"""One table of every measured target: selection, its fold SE, and materiality.

Reads each ``results/selection_<key>.json`` (one per production fit; trace and
remedy runs, whose tag starts with an underscore after the key, are skipped
unless named) through ``selection.summarize`` and writes
``results/all_targets.csv``. A target is material when its selection exceeds
its fold standard error and 0.5% of its total (the brief's threshold), on the
held-out donors (``material``) or on the production recipients
(``material_production``).

Usage: python aggregate.py [--out ../results/all_targets.csv]
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
SKIP_TAGS = ("_trace", "_remedies", "_traceprod", "_trace_scorp", "_rest", "_halved", "_rerun", "_draws", "_check")
THRESHOLD = 0.5  # percent of total
#: A fit measured again with better recipients: the newer key replaces the
#: older one once it has every fold, and stays out of the table until then.
SUPERSEDED_BY = {"puf_chain": "puf_chain_twin"}
FOLDS = 5


def keys() -> list[str]:
    out = []
    for path in sorted(glob.glob(str(ROOT / "results/selection_*.json"))):
        key = Path(path).stem.removeprefix("selection_")
        if any(key.endswith(t) for t in SKIP_TAGS):
            continue
        out.append(key)
    for old, new in SUPERSEDED_BY.items():
        if new in out:
            done = len(json.loads((ROOT / f"results/selection_{new}.json").read_text())["folds"])
            out.remove(old if done >= FOLDS and old in out else new)
    return out


def _median_nonzero(doc: dict, target: str) -> float:
    """Median over folds of the training donors the sign forests grew on.

    A held-out fold holds about a quarter as many nonzero donors as a training
    fold (5 folds), so a target with a few hundred is one whose held-out total
    rests on a few dozen donors.
    """
    counts = []
    for f in doc["folds"]:
        t = f["targets"].get(target)
        if not t or not t.get("signs"):
            continue
        signs = [s for s in t["signs"].values() if "n" in s]
        if signs:  # a constant forest or a skipped sign records no count
            counts.append(sum(int(s["n"]) for s in signs))
    return float(np.median(counts)) if counts else float("nan")


def _gap_share(doc: dict, target: str, production: bool) -> float:
    """Largest share (%) of the adaptive total held by recipients no honest donor reached."""
    out = []
    for f in doc["folds"]:
        for s in (f["targets"].get(target) or {}).get("signs", {}).values():
            rec = ((s.get("extra") or {}).get("production") or {}) if production else s
            out.append(float((rec.get("max_empty_adaptive_share") or {}).get("pooled", 0.0)))
    return 100 * max(out) if out else float("nan")


def table() -> pd.DataFrame:
    rows = []
    for key in keys():
        doc = json.loads((ROOT / f"results/selection_{key}.json").read_text())
        if not doc.get("folds"):
            continue
        s = S.summarize(doc, variant="1")
        s.insert(0, "key", key)
        s["n_folds_done"] = len(doc["folds"])
        s["n_train_nonzero"] = s["target"].map(lambda t: _median_nonzero(doc, t))
        s["honest_gap_share"] = s["target"].map(lambda t: _gap_share(doc, t, False))
        s["prod_honest_gap_share"] = s["target"].map(lambda t: _gap_share(doc, t, True))
        rows.append(s)
    df = pd.concat(rows, ignore_index=True)
    ran = df["regime"] != "not_run"
    df = df.loc[ran].copy()

    def material(sel, se):
        return (sel.abs() > se) & (sel.abs() > THRESHOLD)

    df["material"] = material(df["sel_pooled"], df["sel_pooled_se"])
    if "prod_sel_pooled" in df:
        df["material_production"] = material(df["prod_sel_pooled"], df["prod_sel_pooled_se"])
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/all_targets.csv")
    args = ap.parse_args()
    df = table()
    df.to_csv(args.out, index=False)
    cols = [c for c in ("key", "target", "regime", "n_folds_done", "reported_total_mean",
                        "sel_pooled", "sel_pooled_se", "prod_sel_pooled", "prod_sel_pooled_se",
                        "material", "material_production") if c in df]
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(df.loc[df["material"] | df.get("material_production", False), cols]
              .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"\n{len(df)} targets measured; {int(df['material'].sum())} material on held-out "
          f"donors; {int(df.get('material_production', pd.Series(dtype=bool)).sum())} on "
          "production recipients")


if __name__ == "__main__":
    main()
