"""Count recipients an honest or adaptive fill left with no donor in any tree.

``complete_honest`` in ``selection.py`` gives such a recipient its adaptive
estimate, so it adds nothing to the difference. Every run records the most any
half-forest left empty, per sign, fill and mode. This prints each nonzero count
and the largest counts overall.

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
