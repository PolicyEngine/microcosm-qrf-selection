"""Write the plan for ``run_smoothing.sh``: every reported fit, cheapest first.

Each line names a fit and, when its base run had no production recipients,
``--no-production``, so the split runs on exactly the base run's inputs.

Usage: python smoothing_plan.py [--out ../results/smoothing_plan.txt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import aggregate  # noqa: E402

ROOT = HERE.parent


def plan() -> list[str]:
    rows = []
    for key in aggregate.keys():
        doc = json.loads((ROOT / f"results/selection_{key}.json").read_text())
        seconds = sum(f.get("seconds", 0.0) for f in doc["folds"])
        flags = "" if doc.get("meta", {}).get("production_recipients") else " --no-production"
        rows.append((seconds, f"{key}{flags}"))
    return [line for _, line in sorted(rows)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/smoothing_plan.txt")
    args = ap.parse_args()
    lines = plan()
    args.out.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} fits -> {args.out}")


if __name__ == "__main__":
    main()
