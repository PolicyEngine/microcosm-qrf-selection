"""REPORT.md quotes the public result files exactly."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import report_rows as R  # noqa: E402

REPORT = Path(__file__).resolve().parents[2] / "REPORT.md"


@pytest.mark.parametrize("table", ["selection", "closer", "remedies", "lcfs", "coverage",
                                   "known_answer", "draws", "distribution", "full_size",
                                   "smoothing_check"])
def test_every_generated_row_appears_in_the_report(table):
    lines = {line.strip() for line in REPORT.read_text().splitlines()}
    missing = [row for row in R.all_rows()[table] if row not in lines]
    assert not missing, "\n".join(missing)


def flat() -> str:
    return " ".join(REPORT.read_text().split())


CONFIGS = ("honest fills", "min_samples_leaf=5", "min_samples_leaf=20", "weighted_leaves=True")


def production_size():
    d = R.load()
    return R.full_size_numbers(d["full_size"], d["targets"]), d


def test_report_states_the_lcfs_coverage_ranges():
    r = R.lcfs_coverage_ranges(R.load()["full_size"])
    phrase = (f"Across the {r['n']} LCFS targets the remedies cover, {r['a10lo']}-{r['a10hi']}% of "
              f"held-out donors fall at or below the production-size forest's 10th percentile and "
              f"{r['a90lo']}-{r['a90hi']}% at or below its 90th; honest fills give "
              f"{r['h10lo']}-{r['h10hi']}% and {r['h90lo']}-{r['h90hi']}%.")
    assert phrase in flat(), phrase


def test_report_states_the_lcfs_coverage_gaps():
    text = flat()
    g = R.lcfs_gap_ranges(R.load()["full_size"])
    for phrase in [f"Across the {g['n']} LCFS targets the remedies ran on, the production-size "
                   f"forest's mean absolute gap between each tau and its coverage runs "
                   f"{g['prod_lo']:.3f}-{g['prod_hi']:.3f}.",
                   f"(to {g['honest_lo']:.3f}-{g['honest_hi']:.3f})",
                   f"(to {g['leaf20_lo']:.3f}-{g['leaf20_hi']:.3f})"]:
        assert phrase in text, phrase


def test_report_states_the_production_size_numbers():
    text = flat()
    n, _ = production_size()
    hon, l5, l20, tree = (n[k] for k in CONFIGS)
    for phrase in [
        f"honest fills lower mean pinball loss by a median {abs(hon['pinball']):.1f}% "
        f"(lower on {hon['lower']} of the {hon['n']})",
        f"`min_samples_leaf=20` lowers pinball by a median {abs(l20['pinball']):.1f}% "
        f"(lower on {l20['lower']})",
        f"`min_samples_leaf=5` lowers pinball by {abs(l5['pinball']):.1f}% and "
        f"`weighted_leaves=True` by {abs(tree['pinball']):.1f}%",
        f"Honest fills raise pinball by 100% or more on {hon['big']} targets and "
        f"`min_samples_leaf=20` on {l20['big']}.",
        f"{R.signed(hon['coverage'], 3)} for honest fills, {R.signed(l20['coverage'], 3)} for "
        f"`min_samples_leaf=20`, {R.signed(l5['coverage'], 3)} for `min_samples_leaf=5` and "
        f"{R.signed(tree['coverage'], 3)} for `weighted_leaves=True`",
    ]:
        assert phrase in text, phrase


def test_the_options_table_quotes_the_production_size_numbers():
    lines = {line.strip() for line in REPORT.read_text().splitlines()}
    n, d = production_size()
    s = R.RT.remedy_summary(d["compact"], d["targets"][["key", "target", "material"]])
    all_rows = s.loc[s["group"].eq("All")].set_index("remedy")
    contrast = all_rows["median contrast (absolute)"].iloc[0]
    assert (f"| 1. Keep adaptive fills and document them | none | {contrast:.2f}% of the total | "
            f"none | none | none | none | none |") in lines

    def cells(c):
        return (f"{R.signed(c['pinball'], 1)}% | {R.signed(c['coverage'], 3)} | "
                f"{R.signed(c['gap'], 1)} points | {c['up']} | {c['big']} |")

    l20, hon = n["min_samples_leaf=20"], n["honest fills"]
    residual = all_rows.loc["min_samples_leaf=20", "median residual (absolute)"]
    assert (f"| 2. `min_samples_leaf=20` in `_fit_forest` | one argument (`qrf.py:675-683`) | "
            f"{residual:.2f}% | {cells(l20)}") in lines
    assert f"| 3. Cross-fitted honest fills | new fill code | 0 by construction | {cells(hon)}" in lines


def test_report_states_each_hazard_count_and_every_large_one():
    text = flat()
    n, d = production_size()
    for phrase in [
        f"Honest fills raise it on {n['honest fills']['up']} of the {n['honest fills']['n']} "
        f"material targets, by 100% or more on {n['honest fills']['big']}:",
        f"`min_samples_leaf=5` raises it on {n['min_samples_leaf=5']['up']} targets",
        f"`min_samples_leaf=20` raises it on {n['min_samples_leaf=20']['up']} targets, by 100% or "
        f"more on {n['min_samples_leaf=20']['big']}:",
        f"`weighted_leaves=True` raises it on {n['weighted_leaves=True']['up']} targets",
    ]:
        assert phrase in text, phrase
    for config in ("honest fills", "min_samples_leaf=20"):
        hazards = R.full_size_hazards(d["full_size"], d["targets"], config)
        for r in hazards.loc[hazards["pinball_change"] >= R.RT.LARGE_HAZARD].itertuples():
            assert R.plus_minus(r.pinball_change, r.pinball_change_se, 1) in text, (config, r.target)


@pytest.mark.parametrize("config", CONFIGS)
def test_the_two_hazard_counts_agree(config):
    n, d = production_size()
    hazards = R.full_size_hazards(d["full_size"], d["targets"], config)
    assert n[config]["up"] == len(hazards)
    assert n[config]["big"] == int((hazards["pinball_change"] >= R.RT.LARGE_HAZARD).sum())


def test_the_production_size_baseline_compares_with_itself():
    full = R.load()["full_size"]
    base = full.loc[full["config"].eq("production")]
    assert base["effect"].dropna().eq(0).all()
    assert base["pinball_change"].dropna().eq(0).all()


def test_report_states_the_draw_numbers():
    text = " ".join(REPORT.read_text().split())
    d = R.RT.draw_summary_row(R.load()["draws"])
    assert d["targets"] == 36 and d["same sign"] == 36 and d["fits"] == 13
    for phrase in [f"with the sign kept on all {d['targets']} targets",
                   f"a median {d['median shift']:.2f} points and at most {d['largest shift']:.2f}",
                   f"keep the sign on {d['same sign as survey-weight means']} of {d['targets']}",
                   f"move a median {d['median shift from survey-weight means']:.2f} points, up to "
                   f"{d['largest shift from survey-weight means']:.2f} on SCF bank accounts",
                   f"runs a median {-d['median draw gap']:.2f}% below the pooled mean",
                   f"{-d['smallest draw gap']:.2f}% below it on LCFS health",
                   f"by a median {d['median mass effect']:.2f} points in absolute terms",
                   f"({d['median contrast at survey-weight mass']:.2f} at survey-weight mass, "
                   f"{d['median contrast in means']:.2f} at integer mass)"]:
        assert phrase in text, phrase


def test_report_states_the_counts_in_the_files():
    text = " ".join(REPORT.read_text().split())
    h = R.headline(R.load()["targets"])
    phrases = [
        f"{h['us_material']} of the {h['us_reported']} US targets and "
        f"{h['uk_material']} of the {h['uk_reported']} UK targets",
        f"{h['us_material'] + h['uk_material']} carry a material held-out contrast: "
        f"{h['us_material']} of {h['us_reported']} in the US ({h['outside_material']} of "
        f"{h['outside_reported']} outside the PUF chain, {h['puf_material']} of its "
        f"{h['puf_reported']}) and {h['uk_material']} of {h['uk_reported']} in the UK",
        f"same sign on {h['outside_same_sign']} of {h['outside_prod']} targets",
        f"a Spearman rank correlation of {h['outside_spearman']:.2f}",
        f"{h['outside_median_prod']:.1f}% of the drawn total against "
        f"{h['outside_median_held']:.1f}% of the held-out total",
        f"{h['puf_material']} show a material held-out contrast, {h['chain_material_drawn']} a "
        f"material production contrast as a percent of the drawn total and "
        f"{h['chain_material_ratio']} on the ratio",
    ]
    missing = [p for p in phrases if p not in text]
    assert not missing, "\n".join(missing)
    assert h["zero_selection"] == 10 and "Ten targets whose nonzero values all match" in text


def test_report_states_the_split_counts():
    text = flat()
    d = R.load()
    s = R.smoothing_numbers(d["targets"], d["smoothing"])
    assert s["estimated"] + s["missing"] == s["reported"]
    for phrase in [
        f"Of the {s['reported']} reported targets, {s['estimated']} have an estimated split "
        f"({s['us_estimated']} US, {s['uk_estimated']} UK)",
        f"Estimated selection is material on {s['material_selection']} of them: "
        f"{s['us_material_selection']} of {s['us_estimated']} in the US",
        f"on {s['us_material_selection']} of the {s['us_estimated']} US targets and "
        f"{s['uk_material_selection']} of the {s['uk_estimated']} UK targets with an estimate",
    ]:
        assert phrase in text, phrase
