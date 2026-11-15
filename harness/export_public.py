"""Write the public result files (``results/public/``) and the private ones.

The public repository carries aggregates only, and none derived from the
HMRC Survey of Personal Incomes (SPI) public use tape, whose licence sends
any analysis to HMRC ten working days before it is distributed or published.
So the public files:

- drop the two fits whose donors are the SPI tape (``uk_spi_stage1``, the
  SPI stage-1 income fill; ``uk_property_finance``, finance costs from
  ``put2223uk.tab``);
- keep the held-out results of the four FRS-trained fits whose production
  recipients are SPI-channel rows (``uk_spi_stage2``, ``uk_spi_housing_shell``,
  ``uk_uc_reporter_redraw``, ``uk_salary_sacrifice``: their donors are FRS
  records) but drop every UK production-side column, since each UK
  production recipient set is rows of the build frame, which holds the
  SPI-channel households;
- carry no UK release totals or other release statistics, since every UK
  release column includes the SPI-channel households; UK rows keep their
  names and statuses;
- carry no absolute held-out totals for UK targets (the percentages carry the
  result), and suppress every result of a target whose sign forests grew on
  fewer than ``SUPPRESS_BELOW`` nonzero training donors (median over folds),
  since a held-out fold then holds about a quarter as many: a statistic on a
  handful of survey records is a disclosure risk and, at that size, noise;
- list a UK target's remedies only when its held-out contrast is material,
  because a target that entered the remedy runs through its production-side
  contrast alone would reveal that result;
- leave every suppressed target out of the supplemental files (remedies,
  production-size forests, scaling, expected draws, coverage by tau): those
  files cover a target only when a result of it passed a test, so a
  suppressed target's presence there would reveal that its result did. Only
  ``all_targets.csv`` and ``smoothing.csv``, which list every measured target
  whatever its result, keep suppressed rows, blank;
- leave every suppressed target out of the files with their own schema too:
  the PUF twins' coverage (``puf_twin_coverage.csv``) and the targets whose
  chained features fix them (``determined.csv``).

US results are all public: CPS, ACS, SCF and SIPP are public files, and the
IRS PUF licence allows aggregates.

The private files (``--private-dir``, outside the repository) carry
everything.

Usage: python export_public.py [--private-dir ~/reviews/microcosm-selection-2026-11-07-private]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PUBLIC = ROOT / "results/public"

SPI_DONOR_KEYS = {"uk_spi_stage1", "uk_property_finance"}
SPI_RECIPIENT_KEYS = {"uk_spi_stage2", "uk_spi_housing_shell", "uk_uc_reporter_redraw",
                      "uk_salary_sacrifice"}
SUPPRESS_BELOW = 50
UK_ABSOLUTE = ("reported_total_mean", "reported_abs_mean")
RESULT_PREFIXES = ("sel_", "prod_", "own_", "adaptive_", "honest_", "effect",
                   "residual_", "vs_reported", "pinball", "coverage", "material",
                   "selection", "ratio", "honest fills", "min_samples_leaf",
                   "weighted_leaves", "mass_effect", "smoothing", "contrast", "n_fit",
                   "estimated")
#: Donor-count columns, blanked on rows kept without a training donor count.
COUNT_PREFIXES = ("n_fit",)
#: Columns that hold a target's selection measure, one per result file. A
#: target with no donor count keeps its row only when that measure is zero.
SELECTION_COLUMNS = ("sel_pooled", "selection", "residual_selection", "selection_est")


def public_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the SPI-donor fits; blank every UK production-recipient column.

    Every UK production recipient set is rows of the build frame, which holds
    the SPI-channel households, so no UK production-side number is public.
    Reported totals of held-out UK folds are survey-donor totals (FRS, LCFS,
    WAS, ETB, NTS), not release totals: they stay.
    """
    out = df.loc[~df["key"].isin(SPI_DONOR_KEYS)].copy()
    prod = [c for c in out.columns if "prod_" in c or c == "material_production"]
    out[prod] = out[prod].astype(object)
    uk = out["key"].str.startswith("uk_")
    out.loc[uk, prod] = None
    absolute = [c for c in UK_ABSOLUTE if c in out.columns]
    out.loc[uk, absolute] = None
    return out


def suppress_rare(out: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Blank every result of a target with too few nonzero training donors."""
    out = out.merge(counts, on=["key", "target"], how="left", suffixes=("", "_c"))
    if "n_train_nonzero_c" in out.columns:
        out["n_train_nonzero"] = out["n_train_nonzero_c"]
        out = out.drop(columns="n_train_nonzero_c")
    # A target with no count is a constant forest (selection exactly zero,
    # kept: ``selection.analyse_fold`` builds its parts as zeros) or one whose
    # signs were skipped (suppressed).
    n = out["n_train_nonzero"]
    sel = pd.Series(float("nan"), index=out.index)
    for name in SELECTION_COLUMNS:  # a file without one keeps no uncounted target
        if name in out.columns:
            sel = pd.to_numeric(out[name], errors="coerce")
            break
    rare = n.lt(SUPPRESS_BELOW) | (n.isna() & ~sel.fillna(1.0).eq(0.0))
    cols = [c for c in out.columns if c.startswith(RESULT_PREFIXES)
            or c in ("reported_total_mean", "reported_abs_mean")]
    out[cols] = out[cols].astype(object)
    out.loc[rare, cols] = None
    out.loc[rare, "n_train_nonzero"] = None  # the reason says "fewer than"; no exact small count
    # A row kept without a donor count (a constant forest) shows no count at
    # all: its held-out donor counts can run to a handful, which would say how
    # few donors it has.
    uncounted = n.isna() & ~rare
    count_cols = [c for c in out.columns if c.startswith(COUNT_PREFIXES)]
    out[count_cols] = out[count_cols].astype(object)
    out.loc[uncounted, count_cols] = None
    out["suppressed"] = rare.map({True: f"fewer than {SUPPRESS_BELOW} nonzero training donors",
                                  False: ""})
    return out


def public_coverage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    spi = (out["key"].fillna("").str.split("/").str[0].isin(SPI_DONOR_KEYS)
           & out["status"].eq("measured"))
    out.loc[spi, "reason"] = ("SPI-derived: result held until HMRC clears it "
                              "(licence: ten working days' notice)")
    out.loc[out["country"].eq("uk"), "weighted_total"] = None
    return out


def public_inventory(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    stats = [c for c in out.columns if c.startswith(("weighted_", "total_")) or c == "n_unique"]
    out.loc[out["country"].eq("uk"), stats] = None
    return out


def keep_held_out_material(df: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Drop UK targets whose held-out contrast is not material.

    Remedy runs and the checks built on them cover targets whose contrast is
    material on held-out donors or on production recipients. On UK rows the
    second route would reveal a production-recipient result, so a UK target
    appears only when its held-out contrast is material.
    """
    material = targets.loc[targets["material"].fillna(False).astype(bool), ["key", "target"]]
    held_out = df.merge(material.assign(_held_out=True), on=["key", "target"], how="left")
    keep = ~held_out["key"].str.startswith("uk_") | held_out["_held_out"].eq(True)
    return df.loc[keep.to_numpy()].reset_index(drop=True)


def drop_suppressed(df: pd.DataFrame) -> pd.DataFrame:
    """A supplemental file without its suppressed rows (their presence is a result)."""
    return df.loc[df["suppressed"].fillna("").eq("")].reset_index(drop=True)


def public_remedies(path: Path, targets: pd.DataFrame) -> pd.DataFrame | None:
    """Remedy rows, without UK targets that entered through production alone."""
    if not path.exists():
        return None
    return keep_held_out_material(public_targets(pd.read_csv(path)), targets)


def public_held_out(df: pd.DataFrame, counts: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """A held-out check (percentages only): no SPI-donor fit and no suppressed target."""
    out = keep_held_out_material(df.loc[~df["key"].isin(SPI_DONOR_KEYS)], targets)
    return drop_suppressed(suppress_rare(out, counts))


TWIN_KEY = "puf_chain_twin"


def suppressed_targets(public_all: pd.DataFrame, key: str) -> set[str]:
    """Targets of one fit that ``all_targets.csv`` suppresses."""
    rows = public_all.loc[public_all["key"].eq(key)]
    return set(rows.loc[rows["suppressed"].fillna("").ne(""), "target"])


def puf_twin_coverage(path: Path, suppressed: set[str]) -> pd.DataFrame | None:
    """How much of the PUF chain's production recipients the ASEC-twin subset covers.

    US aggregates from ``puf_chain_twin.py``'s check: the subset's share of the
    PUF tax units and of their weight, each target's share of the PUF units'
    released weighted total, and each predictor's weighted mean on the twins
    (CPS values) over its mean at the released values. A suppressed target
    carries no result anywhere, so its row is left out.
    """
    if not path.exists():
        return None
    check = json.loads(path.read_text())
    check = next(_find(check, "released_total_share"))
    rows = [{"kind": "units", "name": "share of PUF tax units", "value": check["unit_share"]},
            {"kind": "units", "name": "share of PUF tax units' weight", "value": check["weight_share"]}]
    rows += [{"kind": "target: share of released total", "name": t, "value": v}
             for t, v in check["released_total_share"].items() if t not in suppressed]
    rows += [{"kind": "predictor: twin mean over released mean", "name": k, "value": v}
             for k, v in check["predictor_mean_twin_over_released"].items()]
    return pd.DataFrame(rows)


def public_determined(path: Path, public_all: pd.DataFrame) -> pd.DataFrame | None:
    """Target-signs whose chained features fix every donor's value (``determined.py``).

    Only the identities: no SPI-donor fit, no suppressed target, no donor
    counts and no near-misses (an exact fit on fewer donors than five per
    column describes a handful of records, not the target).
    """
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # A target's selection is zero only when every sign with a donor is fixed
    # (or constant, whose forest returns its one value).
    fixed = df["status"].isin(["determined", "constant"]) | df["n"].eq(0)
    whole = fixed.groupby([df["key"], df["target"]]).transform("all")
    df = df.assign(whole_target=whole)
    df = df.loc[df["status"].eq("determined") & ~df["key"].isin(SPI_DONOR_KEYS)]
    shown = public_all.loc[public_all["suppressed"].fillna("").eq(""), ["key", "target"]]
    df = df.merge(shown, on=["key", "target"])
    return df[["key", "target", "sign", "whole_target", "identity",
               "max_rel_resid"]].reset_index(drop=True)


def _find(obj, key):
    """Every dict under ``obj`` that holds ``key``."""
    if isinstance(obj, dict):
        if key in obj:
            yield obj
        for v in obj.values():
            yield from _find(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _find(v, key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--private-dir", type=Path,
                    default=Path.home() / "reviews/microcosm-selection-2026-11-07-private")
    args = ap.parse_args()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(ROOT / "results/all_targets.csv")
    coverage = pd.read_csv(ROOT / "results/coverage.csv")
    inventory = pd.read_csv(ROOT / "results/inventory_totals.csv")
    counts = targets[["key", "target", "n_train_nonzero"]]
    public_all = suppress_rare(public_targets(targets), counts)
    public_all.to_csv(PUBLIC / "all_targets.csv", index=False)
    public_coverage(coverage).to_csv(PUBLIC / "coverage.csv", index=False)
    public_inventory(inventory).to_csv(PUBLIC / "inventory_totals.csv", index=False)
    smoothing = ROOT / "results/smoothing.csv"
    if smoothing.exists():
        # Every measured target, whatever its result: suppressed rows stay, blank.
        smooth = public_targets(pd.read_csv(smoothing))
        suppress_rare(smooth, counts).to_csv(PUBLIC / "smoothing.csv", index=False)
    for name in ("remedies.csv", "remedies_compact.csv", "full_size.csv"):
        rem = public_remedies(ROOT / "results" / name, targets)
        if rem is not None:
            drop_suppressed(suppress_rare(rem, counts)).to_csv(PUBLIC / name, index=False)
    twin = puf_twin_coverage(ROOT / "results/capture_us_puf_chain_twin.json",
                             suppressed_targets(public_all, TWIN_KEY))
    if twin is not None:
        twin.to_csv(PUBLIC / "puf_twin_coverage.csv", index=False)
    determined = public_determined(ROOT / "results/determined.csv", public_all)
    if determined is not None:
        determined.to_csv(PUBLIC / "determined.csv", index=False)
    for name in ("scaling.csv", "draw_means.csv", "coverage_by_tau.csv"):
        check = ROOT / "results" / name
        if check.exists():
            public_held_out(pd.read_csv(check), counts, targets).to_csv(PUBLIC / name, index=False)
    for name in ("all_targets.csv", "coverage.csv", "inventory_totals.csv", "remedies.csv",
                 "remedies_compact.csv", "reproduction.csv", "scaling.csv", "draw_means.csv",
                 "coverage_by_tau.csv", "full_size.csv", "smoothing.csv", "determined.csv"):
        src = ROOT / "results" / name
        if src.exists():
            shutil.copy2(src, args.private_dir / name)
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(PUBLIC.glob("*.csv"))}
    (PUBLIC / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
