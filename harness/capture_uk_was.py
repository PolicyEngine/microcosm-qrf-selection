"""Capture the UK WAS wealth chain fits with the production code at 67ee2ddf3.

The ``was_wealth`` stage (stage 19 of the UK national build) fits eleven
segmented QRF chains on the WAS round-8 household tab and draws 19 targets
(``was_wealth.py:847-1150``). This script reads the same pinned tab
(``was_round_8_hhold_eul_may_2025_230525.tab``, sha256 ``18b3eb98...``, the
``was_qrf_donor`` artifact the stage declares), cleans it with
``clean_was_household_table`` (``was_wealth.py:501``) and runs
``impute_was_wealth`` unchanged with the stage's declared seed (0) and forest
size (100 trees, ``fit_weighted_qrf_chain`` in the stage spec) under
``capture.recording()``. Each segment's ``start_chain`` is captured as a fit:
a chain fits exactly what ``fit`` fits on the same inputs
(``tests/test_selection.py::test_chain_fit_matches_fit``).

Recipients. The held-out test needs none. A segment fits nothing when it has
no recipient (``was_wealth.py:944-955``), so the chain runs with the donor's
own predictor rows as recipients and every segment fits. The production
recipients are the build frame's households at stage 19, whose engine
predictors (household net income, capital income) the release does not carry;
``capture_uk_was_production.py`` reconstructs them separately.

Usage: python capture_uk_was.py --out ../results/capture_uk_was.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prodpath  # noqa: E402

prodpath.uk()

import capture  # noqa: E402

ROOT = HERE.parent
WAS_TAB = Path.home() / (
    "reviews/uk-data-spi-reported-2026-10-01/prereqs/was_2006_22/"
    "was_round_8_hhold_eul_may_2025_230525.tab")


def stage_spec() -> dict:
    stages = json.loads((ROOT / "inventory/source_stages_uk_67ee2ddf3.json").read_text())
    return next(s for s in stages["stages"] if s["stage"] == "was_wealth")


def load_donor():
    from microcosm.build.uk_runtime import was_wealth as W
    from microcosm.build.uk_runtime.frs_spine import read_pinned_tab

    spec = stage_spec()
    artifact = next(a for a in spec["artifacts"] if a["role"] == "was_qrf_donor")
    donor = W.clean_was_household_table(read_pinned_tab(WAS_TAB, artifact))
    chain = next(op for op in spec["operations"] if op["kind"] == "fit_weighted_qrf_chain")
    return W, donor, int(chain["seed"]), int(chain["n_estimators"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()
    W, donor, seed, n_estimators = load_donor()
    assert seed == W.UK_WAS_WEALTH_DECLARED_SEEDS["was_wealth"], seed
    recipients = donor.loc[:, [*W.UK_WAS_WEALTH_PREDICTORS,
                               W.UK_WAS_TENURE_CATEGORY_COLUMN]].copy()
    ids = np.arange(1, len(recipients) + 1, dtype=np.int64)
    with capture.recording() as caps:
        result = W.impute_was_wealth(donor, recipients, seed=seed,
                                     n_estimators=n_estimators, record_ids=ids)
    seeds = W.was_wealth_segment_seeds(seed)
    report = {"donor_rows": int(len(donor)), "donor_weight_total": float(donor["weight"].sum()),
              "seed": seed, "n_estimators": n_estimators, "segment_seeds": list(seeds),
              "fits": [], "seconds": None}
    chains = [c for c in caps if c.module.endswith(":chain")]
    assert len(chains) == W.UK_WAS_CHAIN_SEGMENTS, [c.targets for c in caps]
    for i, spec in enumerate(chains):
        assert spec.seed == seeds[i], (i, spec.seed, seeds[i])
        assert spec.n_estimators == n_estimators
        key = f"uk_was__{i}"
        spec.recipients = []  # donor rows standing in as recipients: not production
        capture.save(spec, key, {"stage": "was_wealth", "segment": i,
                                 "country": "uk", "code": "67ee2ddf3",
                                 "donor_file": WAS_TAB.name})
        weights = spec.weights
        totals = {t: float(np.sum(weights * spec.table[t].to_numpy(dtype=np.float64)))
                  for t in spec.targets}
        report["fits"].append({"key": key, **spec.meta(), "donor_weighted_totals": totals})
        print(f"{key}: {spec.targets} rows={len(spec.table)} totals="
              + ", ".join(f"{t}={v:.4g}" for t, v in totals.items()), flush=True)
    report["draw_columns"] = list(result.draws.columns)
    report["seconds"] = time.time() - t0
    args.out.write_text(json.dumps(report, indent=1, default=str))
    print(f"captured {len(chains)} chains in {report['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
