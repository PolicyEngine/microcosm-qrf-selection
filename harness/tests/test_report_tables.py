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
    assert row["median contrast (absolute)"] == 4.0
    assert row["median residual (absolute)"] == 1.0
    assert row["targets with a smaller residual"] == 2  # 1 < 10 and 0.2 < 2, not 5 < 4
    assert row[f"targets with residual below {RT.REMOVED_BELOW}%"] == 1
    assert row["median pinball change, %"] == -1.0
    assert row["targets with lower pinball"] == 2
    assert row[f"targets with pinball up more than {RT.HAZARD_ABOVE}%"] == 1
    assert row["targets scored"] == 3
    assert abs(row["median change in coverage gap"] - (-0.01)) < 1e-12
    assert row["median change in gap to reported total"] == -1.0


def closer_rows():
    return pd.DataFrame({
        "key": ["uk_lcfs", "uk_lcfs", "scf_net_worth", "scf_net_worth", "scf_net_worth"],
        "target": ["a", "b", "c", "zero", "unmeasured"],
        "sel_pooled": [8.0, -2.0, 4.0, 0.0, float("nan")],
        "material": [True, "False", True, False, False],
        "own_vs_reported": [9.0, -3.0, 5.0, 0.0, float("nan")],
        "adaptive_vs_reported": [10.0, -3.0, 4.0, 0.0, float("nan")],
        "honest_vs_reported": [2.0, -1.0, 8.0, 0.0, float("nan")],
    })


def test_closer_summary_counts_measured_nonzero_targets():
    out = RT.closer_summary(closer_rows()).set_index("targets")
    row = out.loc["All"]
    assert row["n"] == 3  # the zero-selection and unmeasured targets drop out
    assert row["adaptive fill"] == 4.0 and row["honest fill"] == 2.0
    assert row["forest as fitted"] == 5.0
    assert row["honest nearer than adaptive"] == 2  # |2|<|10| and |-1|<|-3|, not |8|<|4|
    assert row["honest nearer than forest as fitted"] == 2
    assert out.loc["US", "n"] == 1 and out.loc["UK", "n"] == 2
    assert out.loc["All, material targets only", "n"] == 2


def test_scaling_summary_ignores_undefined_ratios():
    df = pd.DataFrame({"key": ["a", "a", "b", "b"], "target": list("wxyz"),
                       "ratio": [0.5, 1.5, float("nan"), float("inf")]})
    text = RT.scaling_summary(df)
    assert "| 2 | 3 |" in text  # two fits; three defined ratios, the infinite one kept in the count
    assert "1.00" in text        # the median of the two finite ratios


def test_summaries_split_the_puf_chain_from_other_us_fits():
    compact = pd.concat([compact_rows(), compact_rows().iloc[[2]].assign(key="puf_chain_twin", target="p")])
    targets = pd.concat([targets_rows(), pd.DataFrame({"key": ["puf_chain_twin"], "target": ["p"],
                                                       "material": [True]})])
    out = RT.remedy_summary(compact, targets)
    n = out.drop_duplicates("group").set_index("group")["targets"]
    assert (n["US"], n["US outside the PUF chain"], n["PUF chain"], n["UK"], n["All"]) == (2, 1, 1, 2, 4)


def test_closer_summary_splits_the_puf_chain():
    extra = closer_rows().iloc[[2]].assign(key="puf_chain_twin", target="p")
    out = RT.closer_summary(pd.concat([closer_rows(), extra], ignore_index=True)).set_index("targets")
    assert out.loc["PUF chain", "n"] == 1 and out.loc["US outside the PUF chain", "n"] == 1
    assert out.loc["US", "n"] == 2


def test_production_table_uses_the_ratio_only_on_its_own_puf_row():
    rows = pd.DataFrame({
        "key": ["scf_net_worth"] * 3 + ["puf_chain_twin"] * 3,
        "target": list("abcdef"),
        "sel_pooled": [1.0, 2.0, -3.0, 4.0, 5.0, -6.0],
        "prod_sel_pooled": [1.5, 2.5, 3.5, 40.0, 50.0, 60.0],
        "prod_sel_vs_honest": [9.0, 9.0, 9.0, 0.4, 0.5, -0.6],
    })
    lines = RT.production_vs_held_out(rows).splitlines()
    outside = next(l for l in lines if l.startswith("| US outside"))
    ratio = next(l for l in lines if l.startswith("| PUF chain, adaptive"))
    drawn = next(l for l in lines if l.startswith("| PUF chain, %"))
    assert "| 3 | 2 | 2.00 | 2.50 |" in outside  # no ratio values in the outside-chain row
    assert "| 3 | 3 | 5.00 | 0.50 |" in ratio
    assert "| 3 | 2 | 5.00 | 50.00 |" in drawn


def test_hazards_list_each_remedy_that_worsens_pinball():
    text = RT.remedy_hazards(compact_rows(), targets_rows())
    assert len([l for l in text.splitlines() if "| uk_lcfs | b |" in l]) == len(RT.REMEDY_NAMES)
    assert "not_material" not in text
