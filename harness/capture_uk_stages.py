"""Capture UK QRF stage fits on their pinned donor files with the code at 67ee2ddf3.

Each stage function rebuilds the donor exactly as the stage's transform does
(the same reader, cleaning, uprating and seed helpers, fed the stage's own
``SourceStageSpec`` from the packaged UK country spec) and then runs the
stage's own imputation function unchanged under ``capture.recording()``, with
the donor's own predictor rows standing in as recipients (the held-out test
needs no production recipient; a segment or chain fits whatever recipients it
gets). The captured fit is production's: same donor table, predictors,
targets, weights, seed and forest size.

Stages:
  lcfs         ``UKLCFSConsumptionStageTransform.__call__``
               (``lcfs_consumption.py:574-627``): clean_lcfs_consumption_table,
               floor_negative_donor_consumption, impute_lcfs_consumption
               (``lcfs_consumption.py:3353-3412``, one 20-target chain).
  etb_vat      ``etb_vat.py:125-149``: clean_etb_vat_table, donor uprating,
               impute_etb_vat (``etb_vat.py:354-397``).
  etb_services ``etb_services.py:175-198``: clean_etb_services_table,
               impute_etb_services (``etb_services.py:446-489``, a 3-target
               chain).

Usage: python capture_uk_stages.py lcfs etb_vat etb_services spi_stage1 property_finance was_lisa
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
PREREQS = Path.home() / "reviews/uk-data-spi-reported-2026-10-01/prereqs"
LCFS_DIR = ROOT / "prereqs/lcfs_2024_25"
ETB_TAB = PREREQS / "etb_1977_24/householdv2_1977-2024.tab"


def stage_spec(name: str):
    from microcosm.build.country_spec import load_country_spec

    return next(s for s in load_country_spec("uk").sources.stages if s.stage == name)


def _ids(n: int) -> np.ndarray:
    return np.arange(1, n + 1, dtype=np.int64)


def lcfs():
    from microcosm.build.uk_runtime import lcfs_consumption as L
    from microcosm.build.uk_runtime.frs_spine import read_pinned_tab

    stage = stage_spec("lcfs_consumption")
    hh = read_pinned_tab(LCFS_DIR / "dvhh_uk_anon_v3_2024.tab",
                         L._artifact(stage, "lcfs_household_tab"))
    per = read_pinned_tab(LCFS_DIR / "dvper_uk_anon_2024.tab",
                          L._artifact(stage, "lcfs_person_tab"))
    factors, _ = L.lcfs_donor_uprating(stage)
    energy = L.lcfs_energy_pricing(stage)
    donor = L.clean_lcfs_consumption_table(
        per, hh, uprating=factors, energy=energy,
        donor_rake_iterations=L._donor_energy_rake_iterations(stage))
    donor, _ = L.floor_negative_donor_consumption(donor, floor=L.donor_consumption_floor(stage))
    seed = L._operation_seed(stage, "fit_weighted_qrf_chain")
    n_estimators = L._qrf_n_estimators(stage)
    predictors = list(stage_op(stage, "fit_weighted_qrf_chain")["predictors"])
    recipients = donor.loc[:, predictors].copy()
    with capture.recording() as caps:
        L.impute_lcfs_consumption(donor, recipients, seed=seed, n_estimators=n_estimators,
                                  record_ids=_ids(len(recipients)))
    chains = [c for c in caps if c.module.endswith(":chain")]
    assert len(chains) == 1, [c.targets for c in caps]
    return [("uk_lcfs", chains[0], {"stage": "lcfs_consumption", "seed": seed,
                                    "n_estimators": n_estimators,
                                    "donor_files": ["dvhh_uk_anon_v3_2024.tab",
                                                    "dvper_uk_anon_2024.tab"]})]


def stage_op(stage, kind: str) -> dict:
    ops = [op for op in stage.operations if op.kind == kind]
    assert len(ops) == 1, (stage.stage, kind, len(ops))
    return dict(ops[0].parameters)


def etb_vat():
    from microcosm.build.uk_runtime import etb_vat as V
    from microcosm.build.uk_runtime.frs_spine import read_pinned_tab

    stage = stage_spec("etb_vat")
    config = V.etb_vat_configuration(stage)
    raw = read_pinned_tab(ETB_TAB, stage.artifacts[0])
    donor = V.clean_etb_vat_table(raw, **config)
    uprating = V.uprating_operation(stage)
    if uprating is not None:
        factors, _ = V.donor_uprating_factors(uprating)
        donor = V.apply_donor_uprating(donor, factors)
    recipients = donor.loc[:, list(V.UK_ETB_VAT_PREDICTORS)].copy()
    seed = V._qrf_seed(stage)
    with capture.recording() as caps:
        V.impute_etb_vat(donor, recipients, seed=seed, household_ids=_ids(len(recipients)),
                         salt=V._qrf_salt(stage))
    assert len(caps) == 1, len(caps)
    caps[0].recipients = []
    return [("uk_etb_vat", caps[0], {"stage": "etb_vat", "seed": seed,
                                     "donor_files": [ETB_TAB.name]})]


def etb_services():
    from microcosm.build.uk_runtime import etb_services as E
    from microcosm.build.uk_runtime.frs_spine import read_pinned_tab

    stage = stage_spec("etb_services")
    config = E.etb_services_configuration(stage)
    raw = read_pinned_tab(ETB_TAB, stage.artifacts[0])
    factors, _ = E.etb_donor_uprating(stage)
    donor = E.clean_etb_services_table(raw, year=config["year"],
                                       weeks_in_year=config["weeks_in_year"],
                                       uprating=factors)
    recipients = donor.loc[:, list(E.UK_ETB_SERVICES_PREDICTORS)].copy()
    seed = E._qrf_seed(stage)
    with capture.recording() as caps:
        E.impute_etb_services(donor, recipients, seed=seed, record_ids=_ids(len(recipients)))
    chains = [c for c in caps if c.module.endswith(":chain")]
    assert len(chains) == 1, len(caps)
    return [("uk_etb_services", chains[0], {"stage": "etb_services", "seed": seed,
                                            "donor_files": [ETB_TAB.name]})]


SPI_TAB = PREREQS / "spi_2022_23/put2223uk.tab"
WAS_HH_TAB = PREREQS / "was_2006_22/was_round_8_hhold_eul_may_2025_230525.tab"
WAS_PERSON_TAB = PREREQS / "was_2006_22/was_round_8_person_eul_may_2025_230525.tab"
BUILD_PERIOD = 2024  # the release's time_period


def _spi_raw():
    import pandas as pd
    from microcosm.build.uk_runtime import spi_income as SI

    identity = SI.verify_spi_donor_identity(SPI_TAB)  # checks the pinned sha256
    return pd.read_csv(identity.path, delimiter="\t")


def spi_stage1():
    """Stage 1 of ``hmrc_spi_income_spine`` (``spi_income.py:846-1019``).

    The tape is prepared with the stage seed and the donor age model
    (``spi_spine.py:766-770``: ``load_spi_donor_age_model`` at the build
    period), bootstrapped to 100,000 rows by FACT with ``random_state=seed``
    (``spi_income.py:856-866``) and fitted at uniform DESIGN weight. The
    bootstrap repeats tape records, so each row's group is its tape record:
    folds and halves keep a record's copies together.
    """
    import pandas as pd
    from microcosm.build.uk_runtime import spi_income as SI

    stage = stage_spec("hmrc_spi_income_spine")
    op = stage_op(stage, "fit_weighted_qrf_stage1")
    seed, n_estimators, size = int(op["seed"]), int(op["n_estimators"]), int(op["sample_size"])
    assert size == SI.DEFAULT_SPI_DONOR_SAMPLE_SIZE
    assert op["post_sample_fit_weight"] == "uniform"
    donor = SI._prepare_spi_donor(_spi_raw(), seed=seed,
                                  age_model=SI.load_spi_donor_age_model(BUILD_PERIOD))
    sampled = donor.sample(n=size, replace=True, weights="FACT", random_state=seed)
    groups = np.asarray(sampled.index).astype(str)
    sampled = sampled.reset_index(drop=True)
    x = sampled[list(SI.SPI_STAGE1_PREDICTOR_COLUMNS)]
    encoded, _, _ = SI._encode_predictor_frames(
        (x, x.iloc[:50], x.iloc[:50]), categorical=SI.SPI_STAGE1_CATEGORICAL_PREDICTORS)
    outputs = list(SI.SPI_INCOME_QRF_OUTPUT_COLUMNS)
    assert outputs == list(op["outputs"]), (outputs, op["outputs"])
    fit_frame = SI._person_fit_frame(predictors=encoded, targets=sampled[outputs],
                                     weights=np.ones(len(sampled)),
                                     weight_kind=SI.WeightKind.DESIGN)
    with capture.recording(fit_for_real=False) as caps:
        try:
            SI._qrf_class()(n_estimators=n_estimators, seed=seed).fit(
                fit_frame, list(encoded.columns), outputs, weights="design")
        except capture._Stop:
            pass
    assert len(caps) == 1
    return [("uk_spi_stage1", caps[0], {"stage": "hmrc_spi_income_spine", "fit": "stage1",
                                        "seed": seed, "sample_size": size,
                                        "donor_files": [SPI_TAB.name], "_groups": groups})]


def property_finance():
    """``impute_frs_property_finance_costs`` (``property_components.py:527-600``) on
    the analog tape landlords ``spi_tape_finance_cost_donor`` builds
    (``property_components.py:421-520``) from the tape prepared as the stage
    prepares it (``property_components.py:985-1003``)."""
    from microcosm.build.uk_runtime import property_components as PC
    from microcosm.build.uk_runtime import spi_income as SI

    stage = stage_spec("property_components")
    spec = PC.FinanceCostModelSpec.from_parameters(
        PC._operation(stage, PC.IMPUTE_FRS_PROPERTY_FINANCE_COSTS_KIND))
    factors, _ = SI._spi_income_uprating_factors(BUILD_PERIOD)
    prepared = SI.prepare_spi_donor_table(
        _spi_raw(), seed=spec.tape_preparation_seed,
        age_model=SI.load_spi_donor_age_model(BUILD_PERIOD))
    donor = PC.spi_tape_finance_cost_donor(prepared, uprating_factors=factors, spec=spec)
    with capture.recording() as caps:
        PC.impute_frs_property_finance_costs(
            donor, person_ids=np.array([1]), profit=np.array([0.0]), age=np.array([40.0]),
            weights=np.array([1.0]), spec=spec)
    assert len(caps) == 1
    return [("uk_property_finance", caps[0], {"stage": "property_components",
                                              "seed": spec.amount_seed,
                                              "donor_files": [SPI_TAB.name]})]


def was_lisa():
    """``impute_lifetime_isa_balance`` (``was_lisa.py:1049-1085``) on the donor
    ``clean_was_lisa_donor`` builds (``was_lisa.py:1335-1370``). Training rows
    are persons; each row's group is its household (``CASER8``)."""
    from microcosm.build.uk_runtime import was_lisa as WL
    from microcosm.build.uk_runtime.frs_spine import read_pinned_tab

    stage = stage_spec("was_lisa")
    columns = WL.WASLISAColumns.from_parameters(WL._operation(stage, WL.CLEAN_WAS_LISA_DONOR_KIND))
    bspec = WL.BalanceModelSpec.from_parameters(
        WL._operation(stage, WL.IMPUTE_LIFETIME_ISA_BALANCE_KIND))
    person = read_pinned_tab(WAS_PERSON_TAB, WL._artifact(stage, WL.UK_WAS_PERSON_TAB_ROLE),
                             columns=columns.person_raw_columns())
    hh = read_pinned_tab(WAS_HH_TAB, WL._artifact(stage, WL.UK_WAS_HOUSEHOLD_TAB_ROLE))
    donor = WL.clean_was_lisa_donor(person, hh, columns=columns)
    rec = donor.person.loc[:, list(bspec.predictors)].copy()
    rec["person_id"] = np.arange(1, len(rec) + 1)
    with capture.recording() as caps:
        WL.impute_lifetime_isa_balance(donor.person, rec, holds=np.zeros(len(rec), dtype=bool),
                                       spec=bspec)
    assert len(caps) == 1
    value = donor.person[WL.BALANCE_COLUMN].to_numpy(dtype=float)
    credible = (donor.person["holds"].to_numpy() == 1) & np.isfinite(value) & (value > 0)
    key = next(c for c in donor.person.columns if c.lower() in ("caser8", "household_id",
                                                                 "household_key", "case"))
    groups = donor.person.loc[credible, key].astype(str).to_numpy()
    assert len(groups) == len(caps[0].table)
    return [("uk_was_lisa", caps[0], {"stage": "was_lisa", "seed": bspec.seed,
                                      "donor_files": [WAS_PERSON_TAB.name, WAS_HH_TAB.name],
                                      "_groups": groups})]


STAGES = {"lcfs": lcfs, "etb_vat": etb_vat, "etb_services": etb_services,
          "spi_stage1": spi_stage1, "property_finance": property_finance,
          "was_lisa": was_lisa}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=sorted(STAGES))
    ap.add_argument("--out", type=Path, default=ROOT / "results/capture_uk_stages.json")
    args = ap.parse_args()
    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    for name in args.stages:
        t0 = time.time()
        for key, spec, extra in STAGES[name]():
            spec.recipients = []  # donor rows standing in as recipients: not production
            groups = extra.pop("_groups", None)
            capture.save(spec, key, {"country": "uk", "code": "67ee2ddf3", **extra})
            if groups is not None:
                assert len(groups) == len(spec.table)
                # Fixed-width strings: run_selection loads with allow_pickle=False.
                groups = np.asarray(groups).astype(str)
                np.save(capture.DONORS / f"{key}__groups.npy", groups)
                extra["groups"] = int(len(np.unique(groups)))
            w = spec.weights if spec.weights is not None else np.ones(len(spec.table))
            totals = {t: float(np.sum(w * spec.table[t].to_numpy(dtype=np.float64)))
                      for t in spec.targets}
            positives = {t: int((spec.table[t].to_numpy(dtype=np.float64) > spec.zero_atol).sum())
                         for t in spec.targets}
            report[key] = {**spec.meta(), **extra, "donor_weighted_totals": totals,
                           "donor_positive_rows": positives,
                           "seconds": time.time() - t0}
            print(f"{key}: rows={len(spec.table)} seed={spec.seed} trees={spec.n_estimators} "
                  f"targets={len(spec.targets)} weighted={spec.weights is not None}", flush=True)
        args.out.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
