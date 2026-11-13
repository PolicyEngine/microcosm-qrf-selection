"""Invariants of the public result files (``results/public/``).

They need no survey data, so CI runs them on every push. For every row of
every public file:

- no fit whose donors are the HMRC SPI tape appears (``export_public.py``
  holds those results until HMRC has had its ten working days' notice);
- no UK row carries a production-recipient number, a release total or an
  absolute held-out total (the UK build frame holds the SPI-channel
  households);
- every target whose sign forests grew on fewer than 50 nonzero training
  donors carries no result, and says so;
- a UK target appears in the remedy files only when its held-out contrast
  is material;
- no suppressed target appears in a supplemental file, whose rows exist only
  for targets whose results passed a test;
- every target the coverage table does not report gives a reason.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "results/public"
SPI_DONOR_KEYS = {"uk_spi_stage1", "uk_property_finance"}
RESULT_FILES = ("all_targets.csv", "remedies.csv", "remedies_compact.csv", "scaling.csv",
                "full_size.csv", "smoothing.csv")
SUPPLEMENTS = ("remedies.csv", "remedies_compact.csv", "scaling.csv", "full_size.csv",
               "draw_means.csv", "coverage_by_tau.csv")
RESULT_PREFIXES = ("sel_", "prod_", "own_", "adaptive_", "honest_", "effect",
                   "residual_", "vs_reported", "pinball", "coverage", "material",
                   "selection", "ratio", "honest fills", "min_samples_leaf",
                   "weighted_leaves", "smoothing", "contrast", "n_fit")
SUPPRESS_BELOW = 50


def load(name: str) -> pd.DataFrame:
    path = PUBLIC / name
    if not path.exists():
        pytest.skip(f"{name} not built")
    return pd.read_csv(path)


@pytest.mark.parametrize("name", RESULT_FILES)
def test_no_spi_donor_fit(name):
    df = load(name)
    assert not df["key"].isin(SPI_DONOR_KEYS).any()
    assert not df["key"].str.split("/").str[0].isin(SPI_DONOR_KEYS).any()


@pytest.mark.parametrize("name", RESULT_FILES)
def test_uk_rows_carry_no_production_or_absolute_numbers(name):
    df = load(name)
    uk = df["key"].str.startswith("uk_")
    cols = [c for c in df.columns if "prod_" in c or c == "material_production"
            or c in ("reported_total_mean", "reported_abs_mean")]
    assert uk.any()
    assert df.loc[uk, cols].isna().all().all(), cols


@pytest.mark.parametrize("name", RESULT_FILES)
def test_rare_targets_carry_no_results(name):
    df = load(name)
    flagged = df["suppressed"].fillna("").str.startswith("fewer than")
    rare = df["n_train_nonzero"].lt(SUPPRESS_BELOW)
    assert not (rare & ~flagged).any()
    cols = [c for c in df.columns if c.startswith(RESULT_PREFIXES)
            or c in ("reported_total_mean", "reported_abs_mean")]
    assert df.loc[flagged, cols].isna().all().all()


@pytest.mark.parametrize("name", RESULT_FILES)
def test_rare_targets_show_no_exact_count(name):
    df = load(name)
    flagged = df["suppressed"].fillna("").str.startswith("fewer than")
    assert df.loc[flagged, "n_train_nonzero"].isna().all()


@pytest.mark.parametrize("name", ("remedies.csv", "remedies_compact.csv", "full_size.csv"))
def test_uk_remedies_only_on_held_out_material_targets(name):
    """A UK target that entered through production alone would reveal that result."""
    df = load(name)
    uk = df.loc[df["key"].str.startswith("uk_"), ["key", "target"]]
    targets = load("all_targets.csv")
    material = targets["material"].astype(str).str.lower().eq("true")
    allowed = set(map(tuple, targets.loc[material, ["key", "target"]].to_numpy()))
    assert set(map(tuple, uk.to_numpy())) <= allowed


@pytest.mark.parametrize("name", SUPPLEMENTS)
def test_no_suppressed_target_appears_in_a_supplement(name):
    """Membership in a supplement says a result passed a test; a suppressed one must not show."""
    df = load(name)
    targets = load("all_targets.csv")
    suppressed = set(map(tuple, targets.loc[targets["suppressed"].fillna("").ne(""),
                                            ["key", "target"]].to_numpy()))
    assert suppressed, "expected some suppressed targets"
    assert not set(map(tuple, df[["key", "target"]].to_numpy())) & suppressed


def test_reported_targets_carry_results():
    df = load("all_targets.csv")
    shown = df["suppressed"].fillna("").eq("")
    assert df.loc[shown, "sel_pooled"].notna().all()
    assert df.loc[shown, "sel_pooled_se"].notna().all()


def test_coverage_gives_a_reason_for_every_unreported_target():
    cov = load("coverage.csv")
    unreported = cov["status"].ne("measured")
    assert cov.loc[unreported, "reason"].fillna("").str.len().gt(0).all()
    spi = cov["key"].fillna("").str.split("/").str[0].isin(SPI_DONOR_KEYS)
    assert cov.loc[spi & cov["status"].eq("measured"), "reason"].str.startswith(
        "SPI-derived").all()
    assert cov.loc[cov["country"].eq("uk"), "weighted_total"].isna().all()


def test_inventory_carries_no_uk_totals():
    inv = load("inventory_totals.csv")
    cols = [c for c in inv.columns if c.startswith(("weighted_", "total_")) or c == "n_unique"]
    assert inv.loc[inv["country"].eq("uk"), cols].isna().all().all()
