"""Tests for the remedy summary that the methodology decision reads."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import report_tables as RT  # noqa: E402


def compact_rows():
    base = {"folds": 5, "selection_se": 0.1, "coverage_error": 0.05, "vs_reported": 3.0}
    def remedies(residual, pinball, coverage, gap):
        out = {}
        for n in RT.REMEDY_NAMES:
            out.update({f"{n}: effect": -1.0, f"{n}: effect_se": 0.1, f"{n}: residual": residual,
                        f"{n}: pinball": pinball, f"{n}: pinball_se": 0.2,
                        f"{n}: coverage_error": coverage, f"{n}: vs_reported": gap})
        return out
    return pd.DataFrame([
        {"key": "uk_lcfs", "target": "a", "selection": 10.0, **base, **remedies(1.0, -10.0, 0.03, 2.0)},
        {"key": "uk_lcfs", "target": "b", "selection": -4.0, **base, **remedies(-5.0, 2.0, 0.07, 4.0)},
        {"key": "scf_net_worth", "target": "c", "selection": 2.0, **base, **remedies(0.2, -1.0, 0.04, 1.0)},
        {"key": "scf_net_worth", "target": "not_material", "selection": 0.1, **base,
         **remedies(0.0, 0.0, 0.05, 3.0)},
    ])


def targets_rows():
    return pd.DataFrame({"key": ["uk_lcfs", "uk_lcfs", "scf_net_worth", "scf_net_worth"],
                         "target": ["a", "b", "c", "not_material"],
                         "material": [True, "True", True, False]})


def test_summary_uses_only_material_targets_and_splits_by_country():
    out = RT.remedy_summary(compact_rows(), targets_rows())
    row = out.loc[out["group"].eq("All") & out["remedy"].eq("honest fills")].iloc[0]
    assert row["targets"] == 3
    assert out.loc[out["group"].eq("US"), "targets"].unique().tolist() == [1]
    assert out.loc[out["group"].eq("UK"), "targets"].unique().tolist() == [2]


def test_summary_counts_and_medians():
    out = RT.remedy_summary(compact_rows(), targets_rows())
    row = out.loc[out["group"].eq("All") & out["remedy"].eq("min_samples_leaf=20")].iloc[0]
    assert row["median selection (absolute)"] == 4.0
    assert row["median residual (absolute)"] == 1.0
    assert row["targets with a smaller residual"] == 2  # 1 < 10 and 0.2 < 2, not 5 < 4
    assert row[f"targets with residual below {RT.REMOVED_BELOW}%"] == 1
    assert row["median pinball change, %"] == -1.0
    assert row["targets with lower pinball"] == 2
    assert row["targets scored"] == 3
    assert abs(row["median change in coverage gap"] - (-0.01)) < 1e-12
    assert row["median change in gap to reported total"] == -1.0
