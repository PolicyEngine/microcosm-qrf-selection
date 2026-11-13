"""Every table row REPORT.md quotes, generated from the public result files.

``python harness/report_rows.py`` prints the rows.
``harness/tests/test_report_numbers.py`` checks that each one appears in
REPORT.md as written, and that the report's counts match the files, so the
report cannot drift from the numbers it quotes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import report_tables as RT  # noqa: E402

PUBLIC = HERE.parent / "results/public"

# (key, target, label, donors)
US_ROWS = [
    ("scf_net_worth", "net_worth", "net worth", "SCF 2022"),
    ("scf_financial_assets", "bank_account_assets", "bank account assets", "SCF 2022"),
    ("scf_auto_loans", "auto_loan_balance", "auto loan balance", "SCF 2022"),
    ("scf_auto_loans", "auto_loan_interest", "auto loan interest", "SCF 2022"),
    ("sipp_financial_assets__0", "bank_account_assets", "bank account assets", "SIPP"),
    ("sipp_financial_assets__1", "stock_assets", "stock assets", "SIPP"),
    ("sipp_vehicles", "household_vehicles_value", "household vehicles value", "SIPP"),
    ("acs_rent", "rent", "rent", "ACS 2022"),
    ("org_wages", "hourly_wage", "hourly wage", "CPS ORG"),
    ("puf_support_childcare", "spm_unit_pre_subsidy_childcare_expenses", "childcare expenses",
     "CPS ASEC (PUF-support reconstruction)"),
    ("puf_support_weeks_unemployed", "weeks_unemployed", "weeks unemployed",
     "CPS ASEC (PUF-support reconstruction)"),
]
PUF_ROWS = [
    ("taxable_interest_income", "taxable interest income"),
    ("qualified_dividend_income", "qualified dividend income"),
    ("non_qualified_dividend_income", "non-qualified dividend income"),
    ("long_term_capital_gains_before_response", "long-term capital gains"),
    ("tax_exempt_interest_income", "tax-exempt interest income"),
    ("taxable_private_pension_income", "taxable private pension income"),
    ("taxable_ira_distributions", "taxable IRA distributions"),
    ("charitable_cash_donations", "charitable cash donations"),
    ("charitable_non_cash_donations", "charitable non-cash donations"),
    ("real_estate_taxes", "real estate taxes"),
    ("student_loan_interest", "student loan interest"),
    ("qualified_tuition_expenses", "qualified tuition expenses"),
    ("partnership_income", "partnership income"),
    ("sstb_self_employment_income_before_lsr", "SSTB self-employment income"),
    ("health_savings_account_ald", "health savings account deduction"),
]
_LCFS = "LCFS 2024-25"
_WAS = "WAS round 8"
_ETB = "ETB 1977-2024"
_SHELL = "FRS (SPI housing shell)"
_STAGE2 = "FRS (SPI-channel stage 2)"
UK_ROWS = [
    ("uk_lcfs", "household_furnishings_consumption", "household furnishings", _LCFS),
    ("uk_lcfs", "health_consumption", "health", _LCFS),
    ("uk_lcfs", "recreation_consumption", "recreation", _LCFS),
    ("uk_lcfs", "transport_consumption", "transport", _LCFS),
    ("uk_lcfs", "education_consumption", "education", _LCFS),
    ("uk_lcfs", "restaurants_and_hotels_consumption", "restaurants and hotels", _LCFS),
    ("uk_lcfs", "alcohol_and_tobacco_consumption", "alcohol and tobacco", _LCFS),
    ("uk_lcfs", "food_and_non_alcoholic_beverages_consumption", "food", _LCFS),
    ("uk_lcfs_road_fuel", "petrol_share_of_road_fuel", "petrol share of road fuel (redraw)", _LCFS),
    ("uk_was__3", "corporate_wealth_excl_isa", "corporate wealth excluding ISAs", _WAS),
    ("uk_was__2", "other_property_value", "other property value", _WAS),
    ("uk_was__2", "non_residential_property_value", "non-residential property value", _WAS),
    ("uk_was__4", "num_vehicles", "number of vehicles", _WAS),
    ("uk_was__9", "other_residential_property_secured_debt",
     "other residential property secured debt", _WAS),
    ("uk_was__6", "other_mortgage", "other mortgage", _WAS),
    ("uk_was__4", "cash_isa", "cash ISA", _WAS),
    ("uk_was__1", "main_residence_value", "main residence value", _WAS),
    ("uk_was__5", "main_residence_mortgage", "main residence mortgage", _WAS),
    ("uk_was_lisa", "lifetime_isa_balance", "lifetime ISA balance", "WAS round 8 (LISA holders)"),
    ("uk_etb_services", "rail_subsidy_spending", "rail subsidy", _ETB),
    ("uk_etb_services", "dfe_education_spending", "DfE education spending", _ETB),
    ("uk_nts_bus_band", "local_bus_use_band", "local bus use band", "NTS"),
    ("uk_spi_housing_shell", "mortgage_capital_repayment", "mortgage capital repayment", _SHELL),
    ("uk_spi_housing_shell", "structural_insurance_payments", "structural insurance", _SHELL),
    ("uk_spi_housing_shell", "mortgage_interest_repayment", "mortgage interest repayment", _SHELL),
    ("uk_spi_stage2", "pension_contributions_via_salary_sacrifice",
     "pension contributions via salary sacrifice", _STAGE2),
    ("uk_spi_stage2", "child_benefit_reported", "child benefit", _STAGE2),
    ("uk_spi_stage2", "pip_m_reported", "PIP mobility", _STAGE2),
    ("uk_spi_stage2", "tax_free_savings_income", "tax-free savings income", _STAGE2),
    ("uk_spi_stage2", "attendance_allowance_reported", "attendance allowance", _STAGE2),
    ("uk_uc_reporter_redraw", "universal_credit_reported_amount", "universal credit",
     "FRS (UC reporter redraw)"),
    ("uk_salary_sacrifice", "pension_contributions_via_salary_sacrifice",
     "pension contributions via salary sacrifice", "FRS (salary sacrifice)"),
]
LCFS_DIVISIONS = [
    ("household_furnishings_consumption", "furnishings"),
    ("health_consumption", "health"),
    ("recreation_consumption", "recreation"),
    ("transport_consumption", "transport"),
    ("education_consumption", "education"),
    ("restaurants_and_hotels_consumption", "restaurants and hotels"),
    ("alcohol_and_tobacco_consumption", "alcohol and tobacco"),
    ("housing_water_and_electricity_consumption", "housing, water and electricity"),
]
COVERAGE_LABELS = [
    ("measured", "measured"),
    ("measured_as", "measured under the fit's own name"),
    ("derived", "derived from measured draws"),
    ("binary_zero", "binary target (a gate, no amount forest)"),
    ("not_qrf", "another method (carry, rule, classifier, rates)"),
    ("not_produced", "declared but absent from the release"),
    ("not_rebuilt", "QRF target this analysis cannot rebuild"),
]
_PRODUCTION = ("production", "full-size forest, leaf 1 (production)")
_HALF_ADAPTIVE = ("half-size adaptive", "half-size forest, adaptive fill")
_HONEST = ("honest fills", "half-size forest, honest fill")
_LEAF_20 = ("min_samples_leaf=20", "full-size forest, leaf 20")
DISTRIBUTION_ROWS = [
    ("household_furnishings_consumption", "household furnishings",
     [_PRODUCTION, _HALF_ADAPTIVE, _HONEST, _LEAF_20]),
    ("health_consumption", "health", [_PRODUCTION, _HALF_ADAPTIVE, _HONEST]),
    ("housing_water_and_electricity_consumption", "housing, water and electricity",
     [_PRODUCTION, _HALF_ADAPTIVE, _HONEST]),
]
REMEDY_GROUPS = ("US outside the PUF chain", "PUF chain", "UK", "All")


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


def plus_minus(x: float, se: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "—"
    return signed(x, digits) + (f" ± {se:.{digits}f}" if pd.notna(se) else "")


def _split_cells(smoothing: pd.DataFrame, key: str, target: str) -> str:
    """'estimated smoothing | estimated selection' for one target, or dashes."""
    s = smoothing.loc[smoothing["key"].eq(key) & smoothing["target"].eq(target)]
    if s.empty or not _true(s["estimated"]).iloc[0]:
        return "— | —"
    r = s.iloc[0]
    return (f"{plus_minus(r.smoothing, r.smoothing_se)} | "
            f"{plus_minus(r.selection_est, r.selection_est_se)}")


def selection_rows(targets: pd.DataFrame, smoothing: pd.DataFrame) -> list[str]:
    """Section 4's tables: the contrast, its estimated split and the production column."""
    a = targets.set_index(["key", "target"])
    out = []
    for key, target, label, donors in US_ROWS:
        r = a.loc[(key, target)]
        out.append(f"| {donors} | {label} | {plus_minus(r.sel_pooled, r.sel_pooled_se)} | "
                   f"{_split_cells(smoothing, key, target)} | "
                   f"{plus_minus(r.prod_sel_pooled, r.prod_sel_pooled_se)} |")
    for target, label in PUF_ROWS:
        r = a.loc[("puf_chain_twin", target)]
        out.append(f"| {label} | {plus_minus(r.sel_pooled, r.sel_pooled_se)} | "
                   f"{_split_cells(smoothing, 'puf_chain_twin', target)} | "
                   f"{plus_minus(r.prod_sel_vs_honest, r.prod_sel_vs_honest_se)} |")
    for key, target, label, donors in UK_ROWS:
        r = a.loc[(key, target)]
        out.append(f"| {donors} | {label} | {plus_minus(r.sel_pooled, r.sel_pooled_se)} | "
                   f"{_split_cells(smoothing, key, target)} |")
    return out


CHECK_LABELS = {"deterministic": "y = exp(x1), no noise", "lognormal": "lognormal",
                "lognormal, heavy tail": "lognormal, heavy tail",
                "lognormal, discrete predictors": "lognormal, discrete predictors",
                "no signal": "no signal (the known-answer process)"}
CHECK_GROWN = {"target": "the target", "noise": "separate noise"}


def smoothing_check_rows(check: pd.DataFrame) -> list[str]:
    """Section 2's synthetic split: pooled averaging, trees grown on the target or on noise."""
    out = []
    df = check.loc[check["mode"].eq("pooled") & check["trees grown on"].isin(list(CHECK_GROWN))]
    for process, label in CHECK_LABELS.items():
        for grown in CHECK_GROWN:
            for _, r in df.loc[df["process"].eq(process) & df["trees grown on"].eq(grown)].iterrows():
                out.append(f"| {label} | {CHECK_GROWN[grown]} | {int(r['donors']):,} | "
                           f"{plus_minus(r['contrast'], r['contrast_se'])} | "
                           f"{plus_minus(r['selection'], r['selection_se'])} | "
                           f"{plus_minus(r['smoothing'], r['smoothing_se'])} | "
                           f"{plus_minus(r['smoothing_estimated'], r['smoothing_estimated_se'])} |")
    return out


def smoothing_numbers(targets: pd.DataFrame, smoothing: pd.DataFrame) -> dict[str, int | float]:
    """The split's counts and medians the report states, over reported targets with an estimate."""
    rep = targets.loc[targets["suppressed"].fillna("").eq(""), ["key", "target", "material"]]
    df = smoothing.merge(rep, on=["key", "target"])
    est = df.loc[_true(df["estimated"])]
    uk, puf = est["key"].str.startswith("uk_"), est["key"].str.startswith("puf_chain")
    con, smo, sel = (pd.to_numeric(est[c], errors="coerce") for c in
                     ("contrast", "smoothing", "selection_est"))
    mat_c, mat_s = _true(est["material"]), _true(est["material_selection_est"])
    return {"reported": len(df), "estimated": len(est), "missing": len(df) - len(est),
            "us_estimated": int((~uk).sum()), "uk_estimated": int(uk.sum()),
            "material_contrast": int(mat_c.sum()),
            "material_selection": int(mat_s.sum()),
            "us_material_selection": int((mat_s & ~uk).sum()),
            "uk_material_selection": int((mat_s & uk).sum()),
            "outside_estimated": int((~uk & ~puf).sum()),
            "outside_material_selection": int((mat_s & ~uk & ~puf).sum()),
            "puf_estimated": int(puf.sum()), "puf_material_selection": int((mat_s & puf).sum()),
            "material_both": int((mat_c & mat_s).sum()),
            "same_sign": int((mat_c & mat_s & (np.sign(con) == np.sign(sel))).sum()),
            "median_contrast": float(con.abs().median()),
            "median_smoothing": float(smo.abs().median()),
            "median_selection": float(sel.abs().median()),
            "smoothing_over_half": int((smo.abs() > 0.5 * con.abs()).sum())}


def closer_rows(targets: pd.DataFrame) -> list[str]:
    out = []
    for _, r in RT.closer_summary(targets).iterrows():
        out.append(f"| {r['targets']} | {r['n']} | {r['forest as fitted']:.2f} | "
                   f"{r['adaptive fill']:.2f} | {r['honest fill']:.2f} | "
                   f"{r['honest nearer than adaptive']} | "
                   f"{r['honest nearer than forest as fitted']} |")
    return out


def remedy_rows(compact: pd.DataFrame, targets: pd.DataFrame) -> list[str]:
    material = targets[["key", "target", "material"]]
    out = []
    s = RT.remedy_summary(compact, material)
    for _, r in s.loc[s["group"].isin(REMEDY_GROUPS)].iterrows():
        out.append(" ".join([
            f"| {r['group']} | {r['remedy']} | {r['targets']} |",
            f"{r['median contrast (absolute)']:.2f} |",
            f"{r['median residual (absolute)']:.2f} |",
            f"{r['targets with a smaller residual']} |",
            f"{r[f'targets with residual below {RT.REMOVED_BELOW}%']} |",
            f"{signed(r['median pinball change, %'])} |",
            f"{r['targets with lower pinball']} |",
            f"{r[f'targets with pinball up more than {RT.HAZARD_ABOVE}%']} |",
            f"{signed(r['median change in coverage gap'], 3)} |",
            f"{signed(r['median change in gap to reported total'])} |",
        ]))
    return out


def full_size_rows(full: pd.DataFrame, targets: pd.DataFrame) -> list[str]:
    """Each configuration against the fold's production-size forest, per group."""
    out = []
    s = RT.full_size_summary(full, targets[["key", "target", "material"]])
    for _, r in s.loc[s["group"].isin(REMEDY_GROUPS)].iterrows():
        prod = r["median effect on production recipients (absolute)"]
        out.append(" ".join([
            f"| {r['group']} | {r['configuration']} | {r['targets']} |",
            f"{r['median effect on the total (absolute)']:.2f} |",
            "— |" if pd.isna(prod) else f"{prod:.2f} |",
            f"{signed(r['median pinball change, %'])} |",
            f"{r['targets with lower pinball']} |",
            f"{r[f'targets with pinball up more than {RT.HAZARD_ABOVE}%']} |",
            f"{r[f'targets with pinball up {RT.LARGE_HAZARD}% or more']} |",
            f"{signed(r['median change in coverage gap'], 3)} |",
            f"{signed(r['median change in gap to reported total'])} |",
        ]))
    return out


def lcfs_rows(compact: pd.DataFrame) -> list[str]:
    c = compact.loc[compact["key"].eq("uk_lcfs")].set_index("target")
    out = []
    for target, label in LCFS_DIVISIONS:
        r = c.loc[target]
        cells = [f"{signed(r['honest fills: effect'], 1)} / {signed(r['honest fills: pinball'], 1)}"]
        for name in ("min_samples_leaf=5", "min_samples_leaf=20", "weighted_leaves=True"):
            cells.append(f"{signed(r[f'{name}: effect'], 1)} / {signed(r[f'{name}: residual'], 1)}"
                         f" / {signed(r[f'{name}: pinball'], 1)}")
        out.append(f"| {label} | {signed(r['selection'], 1)} | " + " | ".join(cells) + " |")
    return out


def coverage_rows(coverage: pd.DataFrame) -> list[str]:
    n = coverage.groupby(["status", "country"]).size()
    out = []
    for status, label in COVERAGE_LABELS:
        out.append(f"| {label} | {n.get((status, 'uk'), 0)} | {n.get((status, 'us'), 0)} |")
    return out


def known_answer_rows(known: pd.DataFrame) -> list[str]:
    labels = {"noise": "separate noise", "target": "the target"}
    return [f"| {r['donors']:,} | {labels[r['trees grown on']]} | "
            f"{plus_minus(r['adaptive_over_honest_pct'], r['se'])} |"
            for _, r in known.iterrows()]


def distribution_rows(full: pd.DataFrame) -> list[str]:
    """Section 5's table: coverage at the 10th, 50th and 90th percentiles, LCFS divisions."""
    c = full.loc[full["key"].eq("uk_lcfs")].set_index(["target", "config"])
    out = []
    for target, label, fills in DISTRIBUTION_ROWS:
        for i, (config, name) in enumerate(fills):
            r = c.loc[(target, config)]
            out.append(f"| {label if i == 0 else ''} | {name} | {100 * r['coverage_0.1']:.1f}% | "
                       f"{100 * r['coverage_0.5']:.1f}% | {100 * r['coverage_0.9']:.1f}% |")
    return out


def lcfs_coverage_ranges(full: pd.DataFrame) -> dict[str, int]:
    """Whole-percent ranges over the LCFS remedy targets: the production-size forest and honest fills."""
    c = full.loc[full["key"].eq("uk_lcfs")]
    out = {}
    for config, name in (("production", "a"), ("honest fills", "h")):
        d = c.loc[c["config"].eq(config)]
        for col, tau in (("coverage_0.1", "10"), ("coverage_0.9", "90")):
            out[f"{name}{tau}lo"] = round(100 * d[col].min())
            out[f"{name}{tau}hi"] = round(100 * d[col].max())
    out["n"] = int(c.loc[c["config"].eq("production")].shape[0])
    return out


def lcfs_gap_ranges(full: pd.DataFrame) -> dict[str, int | float]:
    """The LCFS remedy targets' mean coverage gaps, against the production-size forest."""
    c = full.loc[full["key"].eq("uk_lcfs")].pivot(index="target", columns="config",
                                                  values="coverage_error")
    out = {"n": len(c), "prod_lo": c["production"].min(), "prod_hi": c["production"].max()}
    for config, name in (("honest fills", "honest"), ("min_samples_leaf=20", "leaf20")):
        lower = c[config] < c["production"]
        out[f"{name}_lower"] = int(lower.sum())
        out[f"{name}_lo"], out[f"{name}_hi"] = c[config].min(), c[config].max()
        if not lower.all():
            worst = (c[config] - c["production"]).idxmax()
            out[f"{name}_worst"] = (worst, c.loc[worst, "production"], c.loc[worst, config])
    return out


def full_size_numbers(full: pd.DataFrame, targets: pd.DataFrame) -> dict[str, dict]:
    """Each configuration's numbers at production's forest size, over the material targets."""
    s = RT.full_size_summary(full, targets[["key", "target", "material"]])
    out = {}
    for _, r in s.loc[s["group"].eq("All")].iterrows():
        out[r["configuration"]] = {
            "n": int(r["targets"]),
            "pinball": float(r["median pinball change, %"]),
            "lower": int(r["targets with lower pinball"]),
            "up": int(r[f"targets with pinball up more than {RT.HAZARD_ABOVE}%"]),
            "big": int(r[f"targets with pinball up {RT.LARGE_HAZARD}% or more"]),
            "coverage": float(r["median change in coverage gap"]),
            "gap": float(r["median change in gap to reported total"]),
            "effect": float(r["median effect on the total (absolute)"])}
    for group, name in (("US outside the PUF chain", "outside"), ("PUF chain", "puf")):
        for _, r in s.loc[s["group"].eq(group)].iterrows():
            out[r["configuration"]][f"prod_{name}"] = float(
                r["median effect on production recipients (absolute)"])
    return out


def full_size_hazards(full: pd.DataFrame, targets: pd.DataFrame, config: str) -> pd.DataFrame:
    """Material targets on which ``config`` raises pinball by more than the hazard threshold."""
    material = targets.loc[targets["material"].astype(str).str.lower().eq("true"),
                           ["key", "target", "n_train_nonzero"]]
    df = full.loc[full["config"].eq(config)].merge(material, on=["key", "target"],
                                                   suffixes=("_full", ""))
    return df.loc[df["pinball_change"] > RT.HAZARD_ABOVE].sort_values("pinball_change",
                                                                      ascending=False)


def draw_rows(draws: pd.DataFrame) -> list[str]:
    """Section 2's draw table: expected draws against each honest fill's pooled means."""
    d = RT.draw_summary_row(draws)
    n = d["targets"]
    return [f"| the honest fill at integer masses, as production builds in-bag ones | {d['same sign']} of {n} | "
            f"{d['median contrast in means']:.2f} | {d['median contrast in draws']:.2f} | "
            f"{d['median shift']:.2f} | {d['largest shift']:.2f} |",
            f"| the honest fill at survey-weight masses, as in the rest of this report | "
            f"{d['same sign as survey-weight means']} of {n} | "
            f"{d['median contrast at survey-weight mass']:.2f} | "
            f"{d['median contrast in draws']:.2f} | "
            f"{d['median shift from survey-weight means']:.2f} | "
            f"{d['largest shift from survey-weight means']:.2f} |"]


def _true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().eq("true")


def headline(targets: pd.DataFrame) -> dict[str, int | float]:
    """The counts and correlations the report states in its text."""
    df = targets.loc[targets["suppressed"].fillna("").eq("")]
    uk = df["key"].str.startswith("uk_")
    puf = df["key"].str.startswith("puf_chain")
    material = _true(df["material"])
    sel = pd.to_numeric(df["sel_pooled"], errors="coerce")
    prod = pd.to_numeric(df["prod_sel_pooled"], errors="coerce")
    ratio = pd.to_numeric(df["prod_sel_vs_honest"], errors="coerce")
    outside = ~uk & ~puf & sel.notna() & sel.ne(0) & prod.notna()
    chain = puf & sel.notna() & sel.ne(0) & ratio.notna()
    out = {"us_reported": int((~uk).sum()), "uk_reported": int(uk.sum()),
           "us_material": int((material & ~uk).sum()), "uk_material": int((material & uk).sum()),
           "outside_reported": int((~uk & ~puf).sum()),
           "outside_material": int((material & ~uk & ~puf).sum()),
           "puf_reported": int(puf.sum()), "puf_material": int((material & puf).sum()),
           "zero_selection": int(sel.eq(0).sum()),
           "outside_prod": int(outside.sum()),
           "outside_same_sign": int((np.sign(sel[outside]) == np.sign(prod[outside])).sum()),
           "outside_median_held": float(sel[outside].abs().median()),
           "outside_median_prod": float(prod[outside].abs().median()),
           "outside_spearman": float(spearmanr(sel[outside], prod[outside]).statistic),
           "chain_material_drawn": int(_true(df.loc[puf, "material_production"]).sum()),
           "chain_prod": int(chain.sum()),
           "chain_same_sign": int((np.sign(sel[chain]) == np.sign(ratio[chain])).sum())}
    se = pd.to_numeric(df["prod_sel_vs_honest_se"], errors="coerce")
    out["chain_material_ratio"] = int((puf & (ratio.abs() > se) & (ratio.abs() > 0.5)).sum())
    return out


def load(public: Path = PUBLIC) -> dict[str, pd.DataFrame]:
    return {"targets": pd.read_csv(public / "all_targets.csv"),
            "compact": pd.read_csv(public / "remedies_compact.csv"),
            "coverage": pd.read_csv(public / "coverage.csv"),
            "known": pd.read_csv(public / "known_answer.csv"),
            "draws": pd.read_csv(public / "draw_means.csv"),
            "full_size": pd.read_csv(public / "full_size.csv"),
            "smoothing": _optional(public / "smoothing.csv"),
            "check": _optional(public / "smoothing_check.csv")}


def _optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(
        columns=["key", "target", "estimated", "process", "mode", "trees grown on"])


def all_rows(public: Path = PUBLIC) -> dict[str, list[str]]:
    d = load(public)
    return {"selection": selection_rows(d["targets"], d["smoothing"]),
            "closer": closer_rows(d["targets"]),
            "smoothing_check": smoothing_check_rows(d["check"]),
            "remedies": remedy_rows(d["compact"], d["targets"]),
            "lcfs": lcfs_rows(d["compact"]), "coverage": coverage_rows(d["coverage"]),
            "known_answer": known_answer_rows(d["known"]),
            "draws": draw_rows(d["draws"]),
            "distribution": distribution_rows(d["full_size"]),
            "full_size": full_size_rows(d["full_size"], d["targets"])}


def main() -> None:
    for name, rows in all_rows().items():
        print(f"\n## {name}\n")
        print("\n".join(rows))
    print("\n## headline\n")
    d = load()
    for k, v in headline(d["targets"]).items():
        print(k, v)
    print("\n## smoothing\n")
    for k, v in smoothing_numbers(d["targets"], d["smoothing"]).items():
        print(k, v)


if __name__ == "__main__":
    main()
