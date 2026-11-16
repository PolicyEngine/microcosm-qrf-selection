"""Rewrite every results/selection_*.json summary CSV with the current summarize()."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import selection as S  # noqa: E402

ROOT = HERE.parent

for path in sorted(glob.glob(str(ROOT / "results/selection_*.json"))):
    doc = json.loads(Path(path).read_text())
    if not doc.get("folds"):
        print("no folds:", path)
        continue
    for v in doc.get("variants", [1]):
        out = Path(path).with_name(Path(path).stem + f"_v{v}.csv")
        S.summarize(doc, variant=str(v)).to_csv(out, index=False)
    print(f"{Path(path).name}: {len(doc['folds'])} folds", flush=True)
