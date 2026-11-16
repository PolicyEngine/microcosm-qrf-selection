"""Compare two result files number by number.

Walks the ``folds`` of two ``selection_*.json`` files and counts every number
that differs at all, separating quantile metrics from everything else (totals,
residuals, catchment checks). Run times do not count.

Usage: python compare_runs.py [--common] <old.json> <new.json>
``--common`` compares only what both files hold (a remedy run against the base
run of the same fit, say: the remedy run tests fewer targets and adds variants).
Exits 0 when nothing outside the quantile metrics differs, 1 otherwise.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SKIP = {"seconds"}


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _walk(a, b, path, out, common=False) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP:
                continue
            if k not in a or k not in b:
                if not common:
                    out["unmatched"].append([*path, k])
            else:
                _walk(a[k], b[k], [*path, k], out, common)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out["unmatched"].append(path)
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, [*path, i], out, common)
    elif _is_number(a) and _is_number(b):
        out["numbers"] += 1
        if math.isnan(a) or math.isnan(b):
            if not (math.isnan(a) and math.isnan(b)):
                out["differing"].append((path, a, b))
        elif a != b:
            out["differing"].append((path, a, b))
            if "quantile" in path:
                out["max_rel_quantile"] = max(out["max_rel_quantile"],
                                              abs(a - b) / max(abs(a), abs(b)))
    elif a != b:
        out["unmatched"].append(path)


def compare(old: dict, new: dict, common: bool = False) -> dict:
    out = {"numbers": 0, "differing": [], "unmatched": [], "max_rel_quantile": 0.0}
    _walk(old["folds"], new["folds"], [], out, common)
    out["outside_quantile"] = [d for d in out["differing"] if "quantile" not in d[0]]
    return out


def main() -> int:
    args = sys.argv[1:]
    common = "--common" in args
    old_path, new_path = (Path(a) for a in args if a != "--common")
    out = compare(json.loads(old_path.read_text()), json.loads(new_path.read_text()), common)
    print(f"{old_path.stem}: {out['numbers']} numbers; {len(out['outside_quantile'])} differ "
          f"outside quantile metrics; "
          f"{len(out['differing']) - len(out['outside_quantile'])} quantile metrics differ "
          f"(largest relative difference {out['max_rel_quantile']:.2e}); "
          f"{len(out['unmatched'])} unmatched")
    for item in out["outside_quantile"][:5] + out["unmatched"][:5]:
        print("  ", item)
    return 0 if not out["outside_quantile"] and not out["unmatched"] else 1


if __name__ == "__main__":
    sys.exit(main())
