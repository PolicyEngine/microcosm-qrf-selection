"""Markdown tables for REPORT.md, built from the result CSVs.

The public tables read ``results/public/`` (written by ``export_public.py``,
which removes SPI-derived results); the private ones read ``results/``.

Usage:
  python report_tables.py --source ../results/public --out ../results/public/TABLES.md
  python report_tables.py --source ../results --out <private dir>/TABLES.md --private
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

#: key prefix -> (country, donor survey, production stage)
KEY_INFO = {
    "acs_rent": ("US", "ACS 2022 (household heads)", "pre-subsidy rent"),
    "org_wages": ("US", "CPS ORG", "hourly wage"),
    "puf_chain_twin": ("US", "IRS PUF 2015 (uprated), tax units", "PUF tax-detail chain"),
    "puf_chain": ("US", "IRS PUF 2015 (uprated), tax units", "PUF tax-detail chain"),
    "puf_support": ("US", "CPS ASEC persons (reconstruction)", "PUF-support fills"),
    "scf": ("US", "SCF 2022", "wealth"),
    "sipp": ("US", "SIPP", "SIPP fills"),
    "voluntary_filing": ("US", "CPS ASEC", "voluntary filing"),
    "uk_lcfs_road_fuel": ("UK", "LCFS 2024-25", "road-fuel redraw"),
    "uk_lcfs": ("UK", "LCFS 2024-25", "consumption"),
    "uk_etb": ("UK", "ETB 1977-2024", "indirect taxes and services"),
    "uk_was_lisa": ("UK", "WAS round 8 (LISA holders)", "LISA balance"),
    "uk_was": ("UK", "WAS round 8", "wealth"),
    "uk_nts": ("UK", "NTS", "bus use band"),
    "uk_spi_stage1": ("UK", "SPI public use tape", "SPI income spine, stage 1"),
    "uk_property_finance": ("UK", "SPI public use tape", "property finance costs"),
    "uk_spi_stage2": ("UK", "FRS 2024-25 (release rebuild)", "SPI channel, stage 2"),
    "uk_spi_housing_shell": ("UK", "FRS 2024-25 (release rebuild)", "SPI housing shell"),
    "uk_uc_reporter_redraw": ("UK", "FRS 2024-25 (release rebuild)", "UC reporter redraw"),
    "uk_salary_sacrifice": ("UK", "FRS 2024-25 (release rebuild)", "salary sacrifice"),
}


def info(key: str) -> tuple[str, str, str]:
    for prefix in sorted(KEY_INFO, key=len, reverse=True):
        if key.startswith(prefix):
            return KEY_INFO[prefix]
    return ("?", "?", "?")


def pm(x, se, digits=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if se is None or (isinstance(se, float) and not np.isfinite(se)):
        return f"{x:+.{digits}f}"
    return f"{x:+.{digits}f} ± {se:.{digits}f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def coverage_counts(cov: pd.DataFrame) -> str:
    t = cov.pivot_table(index="status", columns="country", values="output", aggfunc="count",
                        fill_value=0)
    t = t.reindex([s for s in ("measured", "measured_as", "derived", "binary_zero", "not_qrf",
                               "not_produced", "not_rebuilt", "pending", "unclassified")
                   if s in t.index])
    t.loc["total"] = t.sum()
    t = t.reset_index()
    return md(t)


def material_table(targets: pd.DataFrame) -> str:
    df = targets.copy()
    mp = df.get("material_production", pd.Series(False, index=df.index))
    mp = mp.fillna(False).astype(bool)
    df = df.loc[df["material"].fillna(False).astype(bool) | mp]
    rows = []
    for _, r in df.iterrows():
        country, survey, stage = info(r["key"])
        rows.append({
            "country": country, "donors": survey, "fit": r["key"], "target": r["target"],
            "folds": int(r["folds"]) if pd.notna(r.get("folds")) else "",
            "held-out selection, % of total": pm(r["sel_pooled"], r["sel_pooled_se"]),
            "production recipients, % of drawn total": pm(r.get("prod_sel_pooled"),
                                                         r.get("prod_sel_pooled_se")),
            "_abs": abs(r["sel_pooled"]) if pd.notna(r["sel_pooled"]) else 0.0,
        })
    out = pd.DataFrame(rows).sort_values(["country", "fit", "_abs"],
                                         ascending=[False, True, False])
    return md(out.drop(columns="_abs"))


def all_measured(targets: pd.DataFrame) -> str:
    rows = []
    for _, r in targets.iterrows():
        rows.append({
            "fit": r["key"], "target": r["target"], "regime": r["regime"],
            "folds": int(r["folds"]) if pd.notna(r.get("folds")) else "",
            "held-out, %": pm(r["sel_pooled"], r["sel_pooled_se"]),
            "adaptive / honest − 1, %": pm(r.get("sel_vs_honest"), r.get("sel_vs_honest_se")),
            "production, %": pm(r.get("prod_sel_pooled"), r.get("prod_sel_pooled_se")),
            "material": ("held-out" if r.get("material") is True else "")
            + (" production" if r.get("material_production") is True else ""),
        })
    return md(pd.DataFrame(rows))


REMEDY_NAMES = ("honest fills", "min_samples_leaf=5", "min_samples_leaf=20",
                "weighted_leaves=True")


def remedies_table(compact: pd.DataFrame) -> str:
    rows = []
    for _, r in compact.iterrows():
        row = {"fit": r["key"], "target": r["target"],
               "selection (leaf 1, adaptive, pooled)": pm(r["selection"], r["selection_se"])}
        for n in REMEDY_NAMES:
            row[f"{n}: effect on total"] = pm(r[f"{n}: effect"], r[f"{n}: effect_se"])
            row[f"{n}: residual selection"] = (f"{r[f'{n}: residual']:+.2f}"
                                               if pd.notna(r[f"{n}: residual"]) else "—")
            row[f"{n}: pinball change"] = pm(r[f"{n}: pinball"], r[f"{n}: pinball_se"], 1)
        rows.append(row)
    return md(pd.DataFrame(rows))


def remedies_short(compact: pd.DataFrame) -> str:
    """Effect on the total (%) and pinball change (%) per remedy, one line each."""
    rows = []
    for _, r in compact.iterrows():
        row = {"fit": r["key"], "target": r["target"],
               "selection": pm(r["selection"], r["selection_se"])}
        for n, short in zip(REMEDY_NAMES, ("honest", "leaf 5", "leaf 20", "tree weights")):
            row[f"{short}: total / residual / pinball"] = (
                f"{r[f'{n}: effect']:+.1f} / {r[f'{n}: residual']:+.1f} / "
                f"{r[f'{n}: pinball']:+.1f}")
        rows.append(row)
    return md(pd.DataFrame(rows))


def reproduction_table() -> str:
    """What each capture checked against production, from the capture records."""
    res = ROOT / "results"
    rows = []
    us = json.loads((res / "capture_us.json").read_text())
    for stage, rec in us.items():
        for c in rec.get("checks", []):
            rows.append({"fit": stage, "check": c["label"], "kind": "replay of released values",
                         "result": f"share exact {c['share_exact']:.4f}; weighted totals differ "
                                   f"by {100 * c['weighted_total_rel_diff']:.4f}%",
                         "rows": c["n"]})
    acs = json.loads((res / "capture_us_acs_rent.json").read_text())["check"]["rent"]
    rows.append({"fit": "acs_rent", "check": "rent (household heads)",
                 "kind": "redraw on rebuilt recipients",
                 "result": f"share exact {acs['share_exact']:.2f}; reconstructed total "
                           f"{100 * (acs['reconstructed_total'] / acs['production_total'] - 1):+.1f}% "
                           "vs released", "rows": ""})
    sup = json.loads((res / "capture_us_puf_support.json").read_text())
    for stage, rec in sup.items():
        for c in rec.get("checks", []):
            for t, v in c.items():
                if not v.get("production_total"):
                    continue
                rows.append({"fit": f"puf_support_{stage}", "check": t,
                             "kind": "reconstruction (different training pool)",
                             "result": "reconstructed total "
                                       f"{100 * (v['reconstructed_total'] / v['production_total'] - 1):+.0f}% "
                                       "vs released", "rows": ""})
    twin = json.loads((res / "capture_us_puf_chain_twin.json").read_text())
    rows.append({"fit": "puf_chain_twin", "check": "production recipients with an ASEC twin",
                 "kind": "recipient coverage",
                 "result": f"{twin['with_asec_twin']:,} of {twin['puf_units']:,} PUF units "
                           f"({100 * twin['weight_share']:.0f}% of their weight)", "rows": ""})
    return md(pd.DataFrame(rows))


def uk_rebuild_checks() -> str:
    """The UK release rebuilds' checks against the build's receipts and draws.

    Private: the recipient side counts SPI-channel rows.
    """
    rec = json.loads((ROOT / "results/capture_uk_release.json").read_text())
    rows = []
    ss = rec["uk_salary_sacrifice"]["check"]
    rows.append({"fit": "uk_salary_sacrifice", "check": "training rows; prediction rows; "
                 "post-conversion headcount; recipient rows reproduced",
                 "receipt": f"{ss['receipt']['training_rows']:,}; "
                            f"{ss['receipt']['prediction_rows']:,}; "
                            f"{ss['receipt']['post_headcount']:,.0f}",
                 "rebuild": f"{ss['rebuild']['training_rows']:,}; "
                            f"{ss['rebuild']['prediction_rows']:,}; "
                            f"{ss['rebuild']['post_headcount']:,.0f}; "
                            f"{100 * ss['recipients']['rows_reproduced']:.1f}%"})
    hs = rec["uk_spi_housing_shell"]["check"]
    share = hs["redraw_equals_release_share"]
    rows.append({"fit": "uk_spi_housing_shell", "check": "training and recipient households; "
                 "redraw equal to the release",
                 "receipt": f"{hs['receipt_training_households']:,}; "
                            f"{hs['receipt_recipient_households']:,}",
                 "rebuild": f"{hs['training_households']:,}; {hs['recipient_households']:,}; "
                            f"{100 * min(share.values()):.1f}-{100 * max(share.values()):.1f}% "
                            "of recipients per column"})
    s2 = rec["uk_spi_stage2"]["check"]
    share = {k: v for k, v in s2["redraw_equals_release_share"].items() if v is not None}
    low = sorted(share.items(), key=lambda kv: kv[1])[:3]
    rows.append({"fit": "uk_spi_stage2", "check": "recipient rows; positive training and "
                 "recipient rows; redraw equal to the release",
                 "receipt": f"{s2['receipt']['recipient_rows']:,}; "
                            f"{s2['receipt']['training_positive_receipt_rows']:,}; "
                            f"{s2['receipt']['recipient_positive_receipt_rows']:,}",
                 "rebuild": f"{s2['rebuild']['recipient_rows']:,}; "
                            f"{s2['rebuild']['training_positive_receipt_rows']:,}; "
                            f"{s2['rebuild']['recipient_positive_receipt_rows']:,}; lowest: "
                            + ", ".join(f"{k} {100 * v:.0f}%" for k, v in low)})
    uc = rec["uk_uc_reporter_redraw"]["check"]
    rows.append({"fit": "uk_uc_reporter_redraw", "check": "training benefit units; screened "
                 "SPI benefit units; SPI persons whose UC equals the release's",
                 "receipt": f"{uc['receipt']['training_benunits']:,}; "
                            f"{uc['receipt']['screened_spi_benunits']:,}",
                 "rebuild": f"{uc['rebuild']['training_benunits']:,}; "
                            f"{uc['rebuild']['screened_spi_benunits']:,}; "
                            f"{100 * uc['spi_person_uc_equals_release_share']:.1f}%"})
    return md(pd.DataFrame(rows))


LCFS_BUILD_RECORD = (ROOT / "prereqs/uk_staged_a6299504/staged/"
                     "uk-frs-calibration-attempt-20261025T091414Z-a6299504/build_record.json")


def lcfs_levelling(targets: pd.DataFrame, build_record: Path = LCFS_BUILD_RECORD) -> str:
    """The build's division over-draw ratios beside each division's selection.

    Private: the ratios are weighted frame spend, and the frame holds the
    SPI-channel households.
    """
    from scipy.stats import spearmanr

    rec = json.loads(build_record.read_text())
    lev = (rec["spine_provenance"]["stage_evidence"]["lcfs_consumption"]["recomposed_totals"]
           ["division_levelling"])
    lcfs = targets.loc[targets["key"].eq("uk_lcfs")].set_index("target")
    rows = []
    for division, v in lev["divisions"].items():
        r = lcfs.loc[division]
        rows.append({"division": division.removesuffix("_consumption"),
                     "selection, % (held-out)": r["sel_pooled"],
                     "se": r["sel_pooled_se"], "over-draw ratio": v["ratio"],
                     "levelling factor": v["factor"]})
    df = pd.DataFrame(rows).sort_values("selection, % (held-out)", ascending=False)
    rho = spearmanr(df["selection, % (held-out)"], df["over-draw ratio"])
    out = df.assign(**{
        "selection, % (held-out)": [pm(a, b) for a, b in zip(df["selection, % (held-out)"],
                                                            df["se"])],
        "over-draw ratio": df["over-draw ratio"].map(lambda x: f"{x:.3f}"),
        "levelling factor": df["levelling factor"].map(
            lambda x: "—" if x == 1.0 else f"{x:.3f}")}).drop(columns="se")
    return (md(out) + f"\n\nShared ratio {lev['shared_ratio']:.4f} (rule "
            f"`{lev['rule']}`, basis `{lev['basis']}`). Spearman rank correlation of "
            f"selection with the over-draw ratio over the {len(df)} divisions: "
            f"{rho.statistic:.3f} (p = {rho.pvalue:.4f}).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=ROOT / "results/public")
    ap.add_argument("--out", type=Path, default=ROOT / "results/public/TABLES.md")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    targets = pd.read_csv(args.source / "all_targets.csv")
    for col in ("material", "material_production"):
        if col in targets:
            targets[col] = targets[col].map({True: True, False: False, "True": True,
                                             "False": False}).astype(object)
    cov = pd.read_csv(args.source / "coverage.csv")
    parts = ["# Tables\n", "Generated by `harness/report_tables.py` from "
             f"`{args.source.relative_to(ROOT) if args.source.is_relative_to(ROOT) else args.source}`."
             " Selection is adaptive minus honest fills of the same trees, as % of the "
             "held-out fold's reported total (held-out) or of the production recipients' "
             "drawn total (production); mean over 5 folds ± its standard error.\n",
             "## Coverage of production QRF outputs\n", coverage_counts(cov), "",
             "## Material targets\n", "Selection exceeds its fold standard error and 0.5% of "
             "the total on the held-out donors or on the production recipients.\n",
             material_table(targets), ""]
    comp = args.source / "remedies_compact.csv"
    if comp.exists():
        c = pd.read_csv(comp)
        parts += ["## Remedies\n", "Each cell: effect on the total / residual selection / "
                  "change in mean pinball loss, all %; effect and residual as % of the held-out "
                  "reported total, pinball relative to production's configuration.\n",
                  remedies_short(c), ""]
    parts += ["## Reproduction checks (US)\n", reproduction_table(), ""]
    if args.private:
        parts += ["## UK release rebuilds against the build\n", uk_rebuild_checks(), ""]
        parts += ["## LCFS division levelling against selection\n",
                  lcfs_levelling(pd.read_csv(args.source / "all_targets.csv")), ""]
    parts += ["## Every measured target\n", all_measured(targets), ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts))
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
