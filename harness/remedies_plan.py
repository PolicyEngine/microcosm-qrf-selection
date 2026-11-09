"""Write the remedy plan: every target whose selection is material.

One line per fit, ``<key> <target> [<target> ...]``, read by
``run_remedies.sh``. A target enters when its selection exceeds its fold
standard error and 0.5% of the total on held-out donors or on production
recipients (``aggregate.py``). ``puf_chain`` gives way to ``puf_chain_twin``:
the twin fits the same donors and forests and adds production recipients.

Usage: python remedies_plan.py [--out ../results/remedies_plan.txt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPLACED = {"puf_chain"}  # measured again, with production recipients, as puf_chain_twin


def plan(df: pd.DataFrame) -> dict[str, list[str]]:
    material = df["material"].fillna(False).astype(bool)
    if "material_production" in df:
        material |= df["material_production"].fillna(False).astype(bool)
    keep = df.loc[material & ~df["key"].isin(REPLACED), ["key", "target"]]
    return {key: list(rows["target"]) for key, rows in keep.groupby("key", sort=False)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/remedies_plan.txt")
    args = ap.parse_args()
    lines = plan(pd.read_csv(ROOT / "results/all_targets.csv"))
    args.out.write_text("".join(f"{k} {' '.join(v)}\n" for k, v in lines.items()))
    print(f"{len(lines)} fits, {sum(map(len, lines.values()))} targets -> {args.out}")


if __name__ == "__main__":
    main()
