"""Coverage: every production QRF output, measured or with a stated reason.

Rows come from two sources:

1. ``results/inventory_totals.csv``: every output of every source stage whose
   manifest declares a QRF operation (``inventory_totals.py``).
2. QRF fits the stage manifests do not list as outputs, which the release
   rebuilds found (the SPI stage-2 FRS-only fill, the SPI housing shell, the
   UC reporter redraw, the LCFS road-fuel redraw, the WAS and LCFS chains'
   internal targets, the SPI stage-1 targets).

Each row gets one status:

- ``measured``: a same-tree result (``results/all_targets.csv``);
- ``measured_as``: measured under the name the QRF fit uses;
- ``derived``: computed from measured draws, so it inherits their selection;
- ``binary_zero``: a QRF target with one positive value, whose positive
  forest is constant, so adaptive and honest fills agree exactly;
- ``not_qrf``: produced by another method (carry, rule, classifier, rates);
- ``not_produced``: declared by the manifest but absent from the release;
- ``not_rebuilt``: a QRF target this analysis could not rebuild, with why;
- ``pending``: a run still in progress when the table was written.

Usage: python coverage.py [--out ../results/coverage.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

US_BASE = "populace 6d06410bb"   # base pool: PUF chain and PUF-support fits
US_REFRESH = "populace cae8640"  # refresh stages of the US release
UK = "microcosm 67ee2ddf3"

#: (country, stage, output) -> (status, measured key/target or None, reason, code reference)
REASONS: dict[tuple[str, str, str], tuple[str, str | None, str, str]] = {
    # ---------------- US ----------------
    ("us", "puf_tax_detail", "investment_interest_expense"): (
        "derived", "puf_chain/home_mortgage_interest",
        "Not a target of the base chain (64 targets at 6d06410bb, none of them this). In the "
        "release it is nonzero on exactly the rows where home_mortgage_interest is (14,107 of "
        "14,108 persons) and its share of the two clusters at a few values (1.5%, 4.0%, 6.0%, "
        "6.8%), as a per-AGI-band split of one interest draw gives. It inherits that draw's "
        "selection. The split step was not traced end to end.",
        f"{US_REFRESH} us_runtime/puf_interest_components.py:164-185 "
        "(split_us_puf_e19200_by_agi_band)"),
    ("us", "scf_wealth", "qualified_passenger_vehicle_loan_interest"): (
        "derived", "scf_auto_loans/auto_loan_interest",
        "auto_loan_interest times one expected-share factor (6 million loans over weighted "
        "households with positive interest, capped at one).",
        f"{US_REFRESH} us_runtime/scf_auto_loans.py:15-24, 520-525"),
    ("us", "ssi_disability_criteria", "meets_ssi_disability_criteria"): (
        "binary_zero", None,
        "Boolean QRF target: the positive forest holds only the value 1, so every fill "
        "returns 1 and adaptive equals honest.",
        "harness/tests/test_selection.py::test_binary_target_has_no_selection"),
    ("us", "sipp_tips", "treasury_tipped_occupation_code"): (
        "not_qrf", None, "Carried from the ASEC occupation code PEIOOCC.",
        f"{US_REFRESH} us_runtime/sipp_tips.py:22-23, 374"),
    ("us", "org_wages", "is_union_member_or_covered"): (
        "not_qrf", None, "Assigned at random from published 2024 BLS state rates.",
        f"{US_REFRESH} us_runtime/org_wages.py:252, 682"),
    ("us", "org_wages", "fsla_overtime_premium"): (
        "derived", "org_wages/hourly_wage",
        "Computed from the imputed hourly wage and the hours signals.",
        f"{US_REFRESH} us_runtime/org_wages.py:14-31, 943"),
    ("us", "prior_year_income", "employment_income_last_year"): (
        "not_rebuilt", None,
        "First target of the stage's joint QRF; formula-owned in PolicyEngine-US, so the "
        "stage drops it and the release lacks it. Its donor is the adjacent-year ASEC join "
        "on the base pool, which the release does not carry.",
        f"{US_BASE} us_runtime/prior_year_income.py:10-18, 118"),
    ("us", "prior_year_income", "self_employment_income_last_year"): (
        "not_rebuilt", None,
        "Second target of the same joint QRF; it conditions on the dropped first target, "
        "so the fit cannot be rebuilt from the release.",
        f"{US_BASE} us_runtime/prior_year_income.py:10-18, 118"),
    ("us", "prior_year_income", "previous_year_income_available"): (
        "not_qrf", None, "Flag recording whether both prior-year amounts were observed.",
        f"{US_BASE} us_runtime/prior_year_income.py:3-8"),
    ("us", "retirement_distributions", "tax_exempt_ira_distributions"): (
        "not_qrf", None, "Roth IRA distributions: the ASEC value copied to the PUF half.",
        f"{US_BASE} us_runtime/retirement_distributions.py:338-347"),
    ("us", "acs_rent", "pre_subsidy_rent"): (
        "measured_as", "acs_rent/rent",
        "The ACS rent QRF's draw, placed on the household head.",
        f"{US_REFRESH} us_runtime/housing_inputs.py:10-14"),
    ("us", "acs_rent", "receives_housing_assistance"): (
        "binary_zero", None,
        "ASEC SPM mapping on the ASEC half; on the PUF half a boolean QRF target, whose "
        "positive forest is constant.",
        f"{US_REFRESH} us_runtime/housing_inputs.py:5-9, 519, 1095-1105"),
    ("us", "acs_rent", "takes_up_housing_assistance_if_eligible"): (
        "not_qrf", None, "Set equal to receipt of housing assistance.",
        f"{US_REFRESH} us_runtime/housing_inputs.py:520, 1129"),
    ("us", "acs_rent", "spm_unit_tenure_type"): (
        "not_qrf", None, "ASEC SPM_TENMORTSTATUS mapped directly.",
        f"{US_REFRESH} us_runtime/housing_inputs.py:5-9"),
    ("us", "acs_rent", "tenure_type"): (
        "not_qrf", None, "ASEC H_TENURE mapped directly.",
        f"{US_REFRESH} us_runtime/housing_inputs.py:5-6"),
    ("us", "vehicle_assets", "household_vehicles_owned"): (
        "not_qrf", None,
        "scikit-learn RandomForestClassifier (majority vote), not RegimeGatedQRF.",
        f"{US_REFRESH} us_runtime/sipp_vehicles.py:804-818"),
    ("us", "other_health_insurance_premiums", "health_insurance_premiums"): (
        "derived", "puf_support_other_health_insurance/"
        "health_insurance_premiums_without_medicare_part_b",
        "Self-employed premium attribution of the imputed premiums.",
        f"{US_REFRESH} us_runtime/other_health_insurance.py:111-122"),
    ("us", "other_health_insurance_premiums", "is_self_employed"): (
        "not_qrf", None, "Flag set by the self-employed premium attribution.",
        f"{US_REFRESH} us_runtime/other_health_insurance.py:111-122"),
    # ---------------- UK ----------------
    ("uk", "hmrc_spi_income_spine", "hmrc_spi_employed_income"): (
        "derived", "uk_spi_stage1", "Sum of drawn SPI employment components.",
        f"{UK} uk_runtime/spi_income.py:2085-2124"),
    ("uk", "hmrc_spi_income_spine", "hmrc_spi_total_earned_income"): (
        "derived", "uk_spi_stage1", "Employed income plus drawn earned components.",
        f"{UK} uk_runtime/spi_income.py:2104-2110"),
    ("uk", "hmrc_spi_income_spine", "hmrc_spi_total_investment_income"): (
        "derived", "uk_spi_stage1", "Sum of drawn investment components.",
        f"{UK} uk_runtime/spi_income.py:2111-2116"),
    ("uk", "hmrc_spi_income_spine", "hmrc_spi_assessable_income"): (
        "derived", "uk_spi_stage1", "Earned plus investment totals.",
        f"{UK} uk_runtime/spi_income.py:2117, 2124"),
    ("uk", "hmrc_spi_income_spine", "person_is_spi_income_recipient"): (
        "not_qrf", None, "Flag marking SPI-channel income recipients.",
        f"{UK} uk_runtime/spi_income.py:128"),
    ("uk", "was_wealth", "property_wealth"): (
        "not_produced", None,
        "Declared derived total (sum of the property values); absent from the release.",
        f"{UK} uk_runtime/was_wealth.py:118-131"),
    ("uk", "was_wealth", "corporate_wealth"): (
        "derived", "uk_was__3",
        "corporate_wealth_excl_isa plus stocks_and_shares_isa.",
        f"{UK} uk_runtime/was_wealth.py:132"),
    ("uk", "was_wealth", "gross_financial_wealth"): (
        "derived", "uk_was__3,uk_was__4",
        "Savings, cash ISA, corporate wealth and other financial assets.",
        f"{UK} uk_runtime/was_wealth.py:133-138"),
    ("uk", "was_wealth", "net_financial_wealth"): (
        "derived", "uk_was__3,uk_was__4,uk_was__7",
        "Gross financial wealth less consumer debt and student loans.",
        f"{UK} uk_runtime/was_wealth.py:145-148"),
    ("uk", "was_wealth", "mortgage_debt"): (
        "derived", "uk_was__5,uk_was__6",
        "Main-residence mortgage plus other mortgages.",
        f"{UK} uk_runtime/was_wealth.py:139-142"),
    ("uk", "was_wealth", "directly_held_shares"): (
        "derived", "uk_was__8,uk_was__3",
        "The drawn directly-held fraction of the non-ISA share total.",
        f"{UK} uk_runtime/was_wealth.py:151-165"),
    ("uk", "was_wealth", "unit_and_investment_trusts"): (
        "derived", "uk_was__8,uk_was__3",
        "The non-ISA share total less directly held shares.",
        f"{UK} uk_runtime/was_wealth.py:151-165"),
    ("uk", "was_lisa", "has_lifetime_isa"): (
        "not_qrf", None, "scikit-learn LogisticRegression ownership model.",
        f"{UK} uk_runtime/was_lisa.py:923-1000"),
    ("uk", "was_lisa", "household_lifetime_isa_balance"): (
        "derived", "uk_was_lisa/lifetime_isa_balance",
        "Person balances summed to the household.",
        f"{UK} uk_runtime/was_lisa.py (aggregate_person_to_household)"),
    ("uk", "property_components", "property_rental_income"): (
        "not_qrf", None,
        "Receipts allocated to people with positive profit by HMRC Table 13 bands.",
        f"{UK} uk_runtime/property_components.py:664-680"),
    ("uk", "nts_bus_travel", "local_bus_use_band"): (
        "pending", "uk_nts_bus_band/local_bus_use_band",
        "Ordinal band (0-6) drawn by RegimeGatedQRF, zero-inflated positive.",
        f"{UK} uk_runtime/nts_bus_travel.py:1185-1240"),
    ("uk", "nts_bus_travel", "bus_in_london_trips"): (
        "derived", "uk_nts_bus_band", "Band means by residence group.",
        f"{UK} uk_runtime/nts_bus_travel.py:1513"),
    ("uk", "nts_bus_travel", "other_local_bus_trips"): (
        "derived", "uk_nts_bus_band", "Band means by residence group.",
        f"{UK} uk_runtime/nts_bus_travel.py:1513"),
    ("uk", "nts_bus_travel", "local_bus_trips"): (
        "derived", "uk_nts_bus_band", "Sum of the two trip series.",
        f"{UK} uk_runtime/nts_bus_travel.py:1513"),
    ("uk", "nts_bus_travel", "bus_pass_eligible"): (
        "not_qrf", None, "Age and region eligibility rules.",
        f"{UK} uk_runtime/nts_bus_travel.py:1611"),
    ("uk", "nts_bus_travel", "local_bus_single_fare_share"): (
        "not_qrf", None, "Donor single-fare shares by group.",
        f"{UK} uk_runtime/nts_bus_travel.py:1302"),
    ("uk", "nts_bus_travel", "household_local_bus_trips"): (
        "derived", "uk_nts_bus_band", "Person trips summed to the household.",
        f"{UK} uk_runtime/nts_bus_travel.py:1513"),
    ("uk", "lcfs_consumption", "has_fuel_consumption"): (
        "not_qrf", None, "Assigned from a rate (assign_recipient_has_fuel).",
        f"{UK} uk_runtime/lcfs_consumption.py:608"),
    ("uk", "lcfs_consumption", "uses_energy_prepayment_meter"): (
        "binary_zero", None,
        "Boolean QRF target: the positive forest is constant.",
        f"{UK} uk_runtime/lcfs_consumption.py:1483-1497, 1550-1570"),
    ("uk", "etb_services", "rail_usage"): (
        "derived", "uk_etb_services/rail_subsidy_spending",
        "Rail subsidy spending over a fare index.",
        f"{UK} uk_runtime/etb_services.py:211"),
    **{("uk", "etb_services", o): (
        "not_qrf", None, "NHS visits and spending allocated by age-gender cells.",
        f"{UK} uk_runtime/etb_services.py:656-677")
       for o in ("a_and_e_visits", "admitted_patient_visits", "outpatient_visits",
                 "nhs_a_and_e_spending", "nhs_admitted_patient_spending",
                 "nhs_outpatient_spending")},
    ("uk", "salary_sacrifice", "pension_contributions_via_bonus_sacrifice"): (
        "not_qrf", None, "Bonus conversion at an anchored rate.",
        f"{UK} uk_runtime/salary_sacrifice.py:109-184"),
    ("uk", "salary_sacrifice", "employment_income"): (
        "not_qrf", None, "Pay reduced by the amount converted to salary sacrifice.",
        f"{UK} uk_runtime/salary_sacrifice.py:654-689"),
    ("uk", "salary_sacrifice", "employee_pension_contributions"): (
        "not_qrf", None,
        "Contributions of the converted records move to salary sacrifice.",
        f"{UK} uk_runtime/salary_sacrifice.py:654-689"),
    ("uk", "salary_sacrifice", "salary_sacrifice_pre_conversion_pay"): (
        "not_produced", None, "Audit column of the conversion; absent from the release.",
        f"{UK} uk_runtime/salary_sacrifice.py:654-689"),
}

#: Carried ASEC fields of the ORG stage.
for _o in ("cps_race", "is_hispanic", "detailed_occupation_recode", "has_never_worked",
           "is_military", "is_computer_scientist",
           "is_executive_administrative_professional", "is_farmer_fisher"):
    REASONS[("us", "org_wages", _o)] = (
        "not_qrf", None, "ASEC demographic or occupation field carried through.",
        f"{US_REFRESH} us_runtime/org_wages.py:4-10, 539-548")

#: SCF components the manifest declares but neither the runtime nor the
#: release has.
for _o in ("scf_primary_residence_value", "scf_retirement_assets", "scf_business_equity",
           "scf_mortgage_debt", "scf_other_residential_real_estate",
           "scf_nonresidential_real_estate_equity", "scf_other_residential_debt",
           "scf_other_financial_assets", "scf_other_nonfinancial_assets",
           "scf_other_managed_assets", "scf_cash_value_life_insurance",
           "scf_certificates_of_deposit", "scf_savings_bonds", "scf_credit_card_debt",
           "scf_student_loan_debt", "scf_vehicle_installment_debt",
           "scf_other_installment_debt", "scf_other_lines_of_credit", "scf_other_debt"):
    REASONS[("us", "scf_wealth", _o)] = (
        "not_produced", None,
        "Declared in the stage manifest only: no runtime file at cae8640 names it and the "
        "release lacks it.",
        f"{US_REFRESH} populace-build/src/populace/build/us/source_stages.json")

#: Inventory output -> the key holding its same-tree result, where the key's
#: name does not follow from the stage.
STAGE_KEYS = {
    "puf_tax_detail": ["puf_chain"],
    "scf_wealth": ["scf_net_worth", "scf_financial_assets", "scf_auto_loans",
                   "sipp_financial_assets__0", "sipp_financial_assets__1",
                   "sipp_financial_assets__2"],
    "org_wages": ["org_wages"], "acs_rent": ["acs_rent"], "sipp_tips": ["sipp_tips"],
    "vehicle_assets": ["sipp_vehicles"], "sipp_head_start": ["sipp_head_start"],
    "voluntary_filing_input": ["voluntary_filing"],
    "child_support_inputs": ["puf_support_child_support"],
    "childcare_inputs": ["puf_support_childcare"],
    "disability_benefits_input": ["puf_support_disability_benefits"],
    "energy_subsidy": ["puf_support_energy_subsidy"],
    "other_health_insurance_premiums": ["puf_support_other_health_insurance"],
    "retirement_contributions": ["puf_support_retirement_contributions"],
    "retirement_distributions": ["puf_support_retirement_distributions", "puf_chain"],
    "weeks_unemployed_input": ["puf_support_weeks_unemployed"],
    "workers_compensation_input": ["puf_support_workers_compensation"],
    "hmrc_spi_income_spine": ["uk_spi_stage1"],
    "was_wealth": [f"uk_was__{i}" for i in range(11)],
    "was_lisa": ["uk_was_lisa"], "lcfs_consumption": ["uk_lcfs", "uk_lcfs_road_fuel"],
    "etb_services": ["uk_etb_services"], "etb_vat": ["uk_etb_vat"],
    "property_components": ["uk_property_finance"],
    "salary_sacrifice": ["uk_salary_sacrifice"],
}

#: QRF fits beyond the manifests' declared outputs, by key.
EXTRA_FITS = {
    "uk_spi_stage2": ("uk", "hmrc_spi_income_spine", "FRS-only fill (stage 2)"),
    "uk_spi_housing_shell": ("uk", "spi_housing_shell", "SPI housing shell"),
    "uk_uc_reporter_redraw": ("uk", "uc_reporter_redraw", "UC reporter redraw"),
    "uk_lcfs_road_fuel": ("uk", "lcfs_consumption", "zero road-fuel redraw"),
    "uk_spi_stage1": ("uk", "hmrc_spi_income_spine", "stage-1 targets"),
    "uk_property_finance": ("uk", "property_components", "FRS property finance costs"),
    "uk_was_lisa": ("uk", "was_lisa", "balance"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/coverage.csv")
    args = ap.parse_args()
    inv = pd.read_csv(ROOT / "results/inventory_totals.csv")
    meas = pd.read_csv(ROOT / "results/all_targets.csv")
    by_target = {(r.key, r.target): r for r in meas.itertuples()}
    rows, used = [], set()
    for r in inv.itertuples():
        reason = REASONS.get((r.country, r.stage, r.output))
        keys = STAGE_KEYS.get(r.stage, [])
        hit = next(((k, r.output) for k in keys if (k, r.output) in by_target), None)
        row = {"country": r.country, "stage": r.stage, "output": r.output,
               "weighted_total": r.weighted_total}
        if hit is not None and reason is None:
            m = by_target[hit]
            used.add(hit)
            status = "measured" if m.regime != "not_run" else "pending"
            row.update(status=status, key=hit[0], target=hit[1],
                       reason="" if status == "measured" else "fold runs in progress",
                       code="")
        elif reason is not None:
            status, ref, why, code = reason
            row.update(status=status, key=ref, target=None, reason=why, code=code)
            if status == "measured_as":
                k, t = ref.split("/")
                used.add((k, t))
        else:
            row.update(status="unclassified", key=None, target=None,
                       reason="no measurement and no recorded reason", code="")
        rows.append(row)
    # Measured targets the inventory does not list (internal and extra fits).
    for m in meas.itertuples():
        if (m.key, m.target) in used:
            continue
        country, stage, label = EXTRA_FITS.get(
            m.key, ("uk" if m.key.startswith("uk_") else "us", m.key, "internal target"))
        rows.append({"country": country, "stage": stage, "output": m.target,
                     "weighted_total": None,
                     "status": "measured" if m.regime != "not_run" else "pending",
                     "key": m.key, "target": m.target,
                     "reason": f"{label}; not a declared stage output", "code": ""})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out["status"].value_counts().to_string())
    bad = out[out["status"] == "unclassified"]
    if len(bad):
        print(bad[["country", "stage", "output"]].to_string(index=False))


if __name__ == "__main__":
    main()
