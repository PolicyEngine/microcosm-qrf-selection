"""Count recipients an honest or adaptive fill left with no donor in any tree.

``fill_forest`` drops such a recipient from a total. Every run records the most
any half-forest left empty, per sign, fill and mode. Prints each target with a
nonzero count and the largest count overall.

Usage: python empty_check.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def walk():
    for path in sorted(glob.glob(str(ROOT / "results/selection_*.json"))):
        doc = json.loads(Path(path).read_text())
        for fold in doc.get("folds", []):
            for target, t in fold["targets"].items():
                for sign, s in (t.get("signs") or {}).items():
                    for variant, v in s.get("variants", {}).items():
                        # max_empty_recipients: {fill: {mode: n}}, on the sign or variant
                        pass
                    empty = s.get("max_empty_recipients")
                    if isinstance(empty, dict):
                        for fill, modes in empty.items():
                            for mode, n in modes.items():
                                yield Path(path).stem, fold.get("fold"), target, sign, fill, mode, n
                    prod = (s.get("extra") or {}).get("production") or {}
                    if isinstance(prod, dict) and "max_empty" in prod:
                        yield Path(path).stem, fold.get("fold"), target, sign, "production", "any", prod["max_empty"]


def main() -> None:
    rows, seen = [], 0
    for r in walk():
        seen += 1
        if r[-1]:
            rows.append(r)
    print(f"{seen} counts read; {len(rows)} nonzero")
    for r in sorted(rows, key=lambda x: -x[-1])[:60]:
        print(r)


if __name__ == "__main__":
    main()
