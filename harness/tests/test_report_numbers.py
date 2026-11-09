"""REPORT.md quotes the public result files exactly."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import report_rows as R  # noqa: E402

REPORT = Path(__file__).resolve().parents[2] / "REPORT.md"


@pytest.mark.parametrize("table", ["selection", "closer", "remedies", "lcfs", "coverage",
                                   "known_answer", "draws", "distribution"])
def test_every_generated_row_appears_in_the_report(table):
    lines = {line.strip() for line in REPORT.read_text().splitlines()}
    missing = [row for row in R.all_rows()[table] if row not in lines]
    assert not missing, "\n".join(missing)


def test_report_states_the_lcfs_coverage_ranges():
    text = " ".join(REPORT.read_text().split())
    r = R.lcfs_coverage_ranges(R.load()["coverage_by_tau"])
    phrase = (f"Across the {r['n']} LCFS targets the remedies cover, {r['a10lo']}-{r['a10hi']}% of "
              f"held-out donors fall at or below the forest's 10th percentile and "
              f"{r['a90lo']}-{r['a90hi']}% at or below its 90th; honest fills give "
              f"{r['h10lo']}-{r['h10hi']}% and {r['h90lo']}-{r['h90hi']}%.")
    assert phrase in text, phrase


def test_report_states_the_draw_numbers():
    text = " ".join(REPORT.read_text().split())
    d = R.RT.draw_summary_row(R.load()["draws"])
    assert d["targets"] == 36 and d["same sign"] == 36 and d["fits"] == 13
    for phrase in [f"same sign on all {d['targets']} targets",
                   f"within {d['largest shift']:.2f} points on the largest shift",
                   f"Draws run a median {-d['median draw gap']:.2f}% below the pooled mean",
                   f"{-d['smallest draw gap']:.2f}% on LCFS health",
                   f"by a median {d['median mass effect']:.2f} points in absolute terms",
                   f"({d['median selection at survey-weight mass']:.2f} at survey-weight mass, "
                   f"{d['median selection in means']:.2f} at integer mass)"]:
        assert phrase in text, phrase


def test_report_states_the_counts_in_the_files():
    text = " ".join(REPORT.read_text().split())
    h = R.headline(R.load()["targets"])
    phrases = [
        f"{h['us_material']} of the {h['us_reported']} US targets and "
        f"{h['uk_material']} of the {h['uk_reported']} UK targets",
        f"{h['us_material'] + h['uk_material']} carry material held-out selection: "
        f"{h['us_material']} of {h['us_reported']} in the US ({h['outside_material']} of "
        f"{h['outside_reported']} outside the PUF chain, {h['puf_material']} of its "
        f"{h['puf_reported']}) and {h['uk_material']} of {h['uk_reported']} in the UK",
        f"same sign on {h['outside_same_sign']} of {h['outside_prod']} targets",
        f"a Spearman rank correlation of {h['outside_spearman']:.2f}",
        f"{h['outside_median_prod']:.1f}% of the drawn total against "
        f"{h['outside_median_held']:.1f}% of the held-out total",
        f"{h['puf_material']} show material held-out selection, {h['chain_material_drawn']} show "
        f"material production selection as a percent of the drawn total and "
        f"{h['chain_material_ratio']} on the ratio",
    ]
    missing = [p for p in phrases if p not in text]
    assert not missing, "\n".join(missing)
    assert h["zero_selection"] == 10 and "Ten targets whose nonzero values all match" in text
