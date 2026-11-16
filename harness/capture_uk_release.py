"""UK QRF stages rebuilt from the release's own rows.

The FRS-trained UK stages fit on rows of the frame being built, not on a
separate survey tape, so their donors exist only inside the build. The UK
release (``data/prod/uk/microcosm_uk_2024_25.h5``, built at 67ee2ddf3) keeps
the columns that pick those rows out (``household_support_channel``,
``household_is_capital_gains_clone``, ``household_is_cgt_support_copy``,
``salary_sacrifice_asked``), so this script rebuilds each stage's training
table and its production recipients from the release and checks the rebuild
against the release's own values before the same-tree test uses it.

What a rebuild from the release cannot match:

* Weights. The stages fit at the household weights of the frame at the stage,
  before calibration (``salary_sacrifice.py:787`` records "household weights at
  the stage, before calibration, per person"). The release carries only the
  calibrated ``household_weight``; the release's calibration artifact
  (``microcosm_uk_2024_25_calibration.npz``, sha256 96756884..., from
  policyengine/populace-uk-private at the release's cut tag) carries
  ``initial_household_weight`` in release row order, the weights calibration
  started from. The rebuild fits at those. They equal a stage's weights when no
  stage after it changes household weights; each stage function checks that
  by reproducing the release's draws.
* Columns a later stage rewrites. Each stage function states which columns
  it reads and which later stages could have written them.

Usage: python capture_uk_release.py salary_sacrifice [--out ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prodpath  # noqa: E402

prodpath.uk()

import capture  # noqa: E402

ROOT = HERE.parent
RELEASE = ROOT / "data/prod/uk/microcosm_uk_2024_25.h5"
CALIBRATION = ROOT / "prereqs/uk_calibration_67ee/microcosm_uk_2024_25_calibration.npz"
CALIBRATION_SHA256 = "96756884d719fa969c2fbdf548f4e56b8311c0c3765c66b1bf9bc8fe862f718f"
# The release renumbers each entity's identifiers from 1 and keeps the build
# identifier, on which "every draw in the build is keyed", beside it
# (``national_frame.py:561-572``, ``uk_release_identifier_tables`` at 615-627).
BUILD_ID = {"person": "person_build_id", "benunit": "benunit_build_id",
            "household": "household_build_id"}


def release(*entities: str) -> dict[str, pd.DataFrame]:
    return {e: pd.read_hdf(RELEASE, key=e) for e in entities}


def initial_household_weights(household: pd.DataFrame) -> pd.Series:
    """Pre-calibration household weights, indexed by ``household_id``."""
    import hashlib

    assert hashlib.sha256(CALIBRATION.read_bytes()).hexdigest() == CALIBRATION_SHA256
    z = np.load(CALIBRATION, allow_pickle=False)
    # The artifact's rows follow the release's household rows: its calibrated
    # weights equal the release's exactly.
    assert np.array_equal(z["household_weight"], household["household_weight"].to_numpy())
    return pd.Series(z["initial_household_weight"], index=household["household_id"].to_numpy())


FRS_JOB = Path.home() / "reviews/uk-data-spi-reported-2026-10-01/prereqs/frs_2024_25/job.tab"
BUILD_RECORD = (ROOT / "prereqs/uk_staged_a6299504/staged/"
                "uk-frs-calibration-attempt-20261025T091414Z-a6299504/build_record.json")


def frs_job_salary_sacrifice() -> pd.Series:
    """The FRS spine's ``pension_contributions_via_salary_sacrifice`` per FRS person.

    ``frs_spine.py:1263-1271``: the sum of the job table's ``spnamt`` over a
    person's jobs, floored at zero, times ``WEEKS_IN_YEAR`` (52, line 82). The
    person identifier is ``sernum * 1000 + person`` (``_normalize_ids``,
    ``frs_spine.py:500-513``). The job table must be the one the build pinned.
    """
    import hashlib

    from microcosm.build.country_spec import load_country_spec

    pin = next(a["sha256"] for s in load_country_spec("uk").sources.stages
               if s.stage == "frs_spine" for a in s.artifacts
               if a.get("role") == "frs_table" and a.get("table") == "job")
    assert hashlib.sha256(FRS_JOB.read_bytes()).hexdigest() == pin
    job = pd.read_csv(FRS_JOB, sep="\t", dtype=str,
                      usecols=lambda c: c.lower() in ("sernum", "person", "spnamt"))
    job.columns = [c.lower() for c in job.columns]
    pid = pd.to_numeric(job["sernum"]) * 1000 + pd.to_numeric(job["person"])
    amount = pd.to_numeric(job["spnamt"], errors="coerce").fillna(0.0)
    return np.maximum(0.0, amount.groupby(pid).sum()) * 52.0


FRS_BENEFITS = FRS_JOB.parent / "benefits.tab"


def frs_benefit_amounts(names: tuple[str, ...]) -> pd.DataFrame:
    """The FRS spine's ``{name}_reported`` per FRS person, for benefit ``names``.

    ``_add_benefits`` (``frs_spine.py:1083-1094``): the sum of ``benamt`` over
    a person's rows of the benefits table whose ``benefit`` code is in
    ``BENEFIT_CODES[name]`` (lines 154-175), times ``WEEKS_IN_YEAR``. The
    person identifier is ``sernum * 1000 + person`` (``_normalize_ids``).
    The release drops five of these (``UK_RELEASE_EXPORT_DROPPED_COLUMNS``,
    ``national_frame.py:530-537``), which the FRS-only fill trains on.
    """
    import hashlib

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.frs_spine import BENEFIT_CODES, WEEKS_IN_YEAR

    pin = next(a["sha256"] for s in load_country_spec("uk").sources.stages
               if s.stage == "frs_spine" for a in s.artifacts
               if a.get("role") == "frs_table" and a.get("table") == "benefits")
    assert hashlib.sha256(FRS_BENEFITS.read_bytes()).hexdigest() == pin
    tab = pd.read_csv(FRS_BENEFITS, sep="\t", dtype=str,
                      usecols=lambda c: c.lower() in ("sernum", "person", "benefit", "benamt"))
    tab.columns = [c.lower() for c in tab.columns]
    tab = tab.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    pid = (tab["sernum"] * 1000 + tab["person"]).astype("int64")
    out = {}
    for name in names:
        hit = tab["benamt"] * tab["benefit"].isin(BENEFIT_CODES[name])
        out[f"{name}_reported"] = hit.groupby(pid).sum() * WEEKS_IN_YEAR
    return pd.DataFrame(out).fillna(0.0)


def stage_receipt(stage: str) -> dict:
    """The stage's evidence in the release's build record (``spine_provenance``)."""
    record = json.loads(BUILD_RECORD.read_text())
    return record["spine_provenance"]["stage_evidence"][stage]


_SALSAC_STATE: dict = {}


def salary_sacrifice():
    """``impute_salary_sacrifice`` (``salary_sacrifice.py:525-640`` at 67ee2ddf3).

    Training rows (``salary_sacrifice.py:577-584``): FRS-channel people outside
    capital-gains clones and CGT support copies who were asked the
    salary-sacrifice question and have employment income. Target:
    ``pension_contributions_via_salary_sacrifice``; predictors ``age`` and
    ``employment_income``; weight: the household weight at the stage; QRF with
    100 trees and seed 42 (``salary_sacrifice.py:71-72``). Recipients
    (``salary_sacrifice.py:592-626``): everyone not asked who has employment
    income, frame-wide, drawn with uniforms keyed on the build ``person_id``.

    After the fit the stage limits amounts to pay and converts the employee
    contributions of a random share of people with contributions and no salary
    sacrifice (``salary_sacrifice.py:654-689``; the rebuild reads that share
    from the stage's receipt in the build record). A converted record's pay falls by the amount moved,
    so the release shows converted rows with salary sacrifice and lower pay.
    The rebuild undoes the conversion:

    * Training rows take their target from the FRS job table (``spnamt``, as
      the spine computes it). On every unconverted training row the release
      equals that amount capped at pay.
    * A converted row is one whose conversion draw falls below the rate, that
      held no salary sacrifice before the stage, and that shows some in the
      release. Its pay before the stage is the release pay plus the amount
      moved (``salary_sacrifice.py:676-683``).
    * For recipients, "held none before the stage" means the QRF drew zero: a
      candidate row is converted when the redrawn QRF output at its
      pre-conversion pay is zero and the release amount is not the QRF draw.

    ``check`` compares the rebuild with the stage's receipt (training rows,
    prediction rows, converted rows and mass, moved amount) and the redrawn
    recipients with the release, row by row.
    """
    import microcosm.build.uk_runtime.salary_sacrifice as SS
    from microcosm.build.stochastic_assignment import (
        integer_record_keys,
        stable_identity_uniforms,
    )
    from microcosm.build.uk_runtime.cgt_structure import HOUSEHOLD_IS_CGT_CLONE
    from microcosm.build.uk_runtime.cgt_support import HOUSEHOLD_IS_CGT_SUPPORT_COPY
    from microcosm.build.uk_runtime.spi_support import support_channel_column

    receipt = stage_receipt("salary_sacrifice")
    rate = float(receipt["headcount_receipt"]["rate"])
    t = release("person", "household")
    person, household = t["person"], t["household"]
    h = household.set_index("household_id")
    ph = person["person_household_id"]
    channels = ph.map(h[support_channel_column("household")]).astype(str).to_numpy()
    clones = ph.map(h[HOUSEHOLD_IS_CGT_CLONE]).fillna(False).astype(bool).to_numpy()
    copies = ph.map(h[HOUSEHOLD_IS_CGT_SUPPORT_COPY]).fillna(False).astype(bool).to_numpy()
    asked = pd.to_numeric(person["salary_sacrifice_asked"]).to_numpy() == 1
    age = pd.to_numeric(person["age"]).to_numpy(dtype=float)
    pay = pd.to_numeric(person["employment_income"]).to_numpy(dtype=float)
    ss = pd.to_numeric(person[SS.SALSAC_OUTPUT]).to_numpy(dtype=float)
    weights = ph.map(initial_household_weights(household)).to_numpy(dtype=float)
    build_id = person[BUILD_ID["person"]].to_numpy()
    low = stable_identity_uniforms(build_id, seed=SS.SALSAC_CONVERSION_SEED,
                                   salt=SS.SALSAC_CONVERSION_SALT) < rate
    frs = person["person_source_id"].map(frs_job_salary_sacrifice()).fillna(0.0).to_numpy()

    # Conversion keeps pay positive (pay falls by at most half), so pay > 0 in
    # the release picks the same rows as pay > 0 at the stage.
    base = (channels == "frs") & ~clones & ~copies
    training_mask = base & asked & (pay > 0.0)
    predict_mask = ~asked & (pay > 0.0)
    unconverted_train = training_mask & ~low
    conv_train = training_mask & low & (frs == 0.0) & (ss > 0.0)
    pay_pre = pay.copy()
    pay_pre[conv_train] += ss[conv_train]

    training = pd.DataFrame({"age": age[training_mask],
                             "employment_income": pay_pre[training_mask],
                             SS.SALSAC_OUTPUT: frs[training_mask],
                             "_fit_weight": weights[training_mask]})
    rows = np.flatnonzero(predict_mask)
    keys = integer_record_keys(build_id[rows], label="salary sacrifice person_id")
    salt = f"salary_sacrifice:{SS.SALSAC_OUTPUT}"
    quantiles = stable_identity_uniforms(keys, seed=SS.SALSAC_QRF_SEED, salt=salt)
    signs = stable_identity_uniforms(keys, seed=SS.SALSAC_QRF_SEED, salt=f"{salt}:sign")

    def frame(pay_values):
        return pd.DataFrame({"age": age[rows], "employment_income": pay_values})

    with capture.recording(fit_for_real=True) as caps:
        model = SS._qrf_class()(n_estimators=SS.SALSAC_QRF_ESTIMATORS, seed=SS.SALSAC_QRF_SEED)
        fitted = model.fit(training, list(SS.SALSAC_PREDICTORS), [SS.SALSAC_OUTPUT],
                           weights="_fit_weight")
    spec = caps[0]

    def draw(pay_values):
        out = fitted.predict_from_uniforms(frame(pay_values),
                                           quantiles={SS.SALSAC_OUTPUT: quantiles},
                                           sign_uniforms={SS.SALSAC_OUTPUT: signs})
        return np.maximum(0.0, out[SS.SALSAC_OUTPUT].to_numpy(dtype=float))

    at_release_pay = np.minimum(draw(pay[rows]), pay[rows])
    at_pre_pay = draw(pay[rows] + ss[rows])  # the pay a converted row had
    candidate = low[rows] & (ss[rows] > 0.0)
    is_draw = np.isclose(at_release_pay, ss[rows], rtol=1e-9, atol=1e-6)
    conv_pred = candidate & ~is_draw & (at_pre_pay == 0.0)
    pay_pre[rows[conv_pred]] += ss[rows[conv_pred]]
    stage_draw = np.where(conv_pred, 0.0, at_release_pay)  # QRF output, capped at pay
    reproduced = np.isclose(stage_draw, np.where(conv_pred, 0.0, ss[rows]), rtol=1e-9, atol=1e-6)

    others = ~training_mask & ~predict_mask  # asked rows outside training, unpaid rows
    conv_other = others & low & (ss > 0.0) & (pay > 0.0)  # an upper bound: see check
    hr = receipt["headcount_receipt"]
    converted = conv_train.copy()
    converted[rows[conv_pred]] = True
    check = {
        "receipt": {"training_rows": receipt["qrf"]["training_rows"],
                    "prediction_rows": receipt["qrf"]["prediction_rows"],
                    "converted_rows": hr["converted_rows"], "converted_mass": hr["converted_mass"],
                    "moved_amount": hr["moved_amount"], "rate": rate,
                    "post_headcount": hr["post_headcount"]},
        "rebuild": {"training_rows": int(training_mask.sum()),
                    "prediction_rows": int(predict_mask.sum()),
                    "post_headcount": float(weights[ss > 0.0].sum()),
                    "converted_training_rows": int(conv_train.sum()),
                    "converted_recipient_rows": int(conv_pred.sum()),
                    "converted_rows_training_plus_recipients": int(converted.sum()),
                    "converted_mass_training_plus_recipients": float(weights[converted].sum()),
                    "moved_amount_training_plus_recipients": float(ss[converted].sum()),
                    "other_rows_low_draw_with_amount_upper_bound": int(conv_other.sum()),
                    "other_mass_upper_bound": float(weights[conv_other].sum()),
                    "other_amount_upper_bound": float(ss[conv_other].sum())},
        "training_unconverted_release_equals_frs_capped_at_pay": float(np.mean(np.isclose(
            ss[unconverted_train], np.minimum(frs[unconverted_train], pay[unconverted_train]),
            atol=1e-6))),
        "training_rows_frs_above_pre_pay": int((frs[training_mask] > pay_pre[training_mask]).sum()),
        "recipients": {"rows_reproduced": float(np.mean(reproduced)),
                       "rows_not_reproduced": int((~reproduced).sum()),
                       "candidates_both_readings_fit": int((candidate & is_draw & (at_pre_pay == 0.0)).sum()),
                       "candidates_neither_reading_fits": int((candidate & ~is_draw & (at_pre_pay != 0.0)).sum()),
                       "stage_draw_total": float((weights[rows] * stage_draw).sum())},
    }
    spec.recipients = [{
        "table": pd.DataFrame({"age": age[rows], "employment_income": pay_pre[rows],
                               SS.SALSAC_OUTPUT: stage_draw}),
        "index": rows, "entity": "person",
        "method": "predict_from_uniforms (release rebuild, conversion undone)",
        "weight": weights[rows], "in_production": np.ones(len(rows), dtype=bool),
        "weight_source": "initial_household_weight (pre-calibration) of the person's household",
        "check": check["recipients"]}]
    groups = ph.to_numpy()[training_mask].astype(str)
    # Pay before the conversion, for stages before this one: exact on the
    # training rows, the recipients and every unpaid row; asked rows outside
    # training (SPI channel, clones, copies) keep their release pay. The
    # amount moved out of an employee contribution equals the release's salary
    # sacrifice on a converted row (salary_sacrifice.py:681-689), keyed here by
    # build person id.
    moved = np.where(converted, ss, 0.0)
    _SALSAC_STATE.update(pay_pre=pay_pre.copy(),
                         exact=training_mask | predict_mask | (pay <= 0.0),
                         moved=pd.Series(moved, index=build_id))
    return [("uk_salary_sacrifice", spec, {
        "stage": "salary_sacrifice", "fit": "impute_salary_sacrifice",
        "rebuilt_from_release": True,
        "weights": "pre-calibration household weights from the calibration artifact",
        "target_source": "FRS 2024-25 job.tab spnamt x 52 (frs_spine.py:1263-1271)",
        "check": check, "_groups": groups})]


def stage18_frame():
    """The frame the ``spi_housing_shell`` stage (stage 18 of 49) saw, rebuilt.

    Households: the release's households less those later stages created
    (high-value support copies, stage 22; CGT support copies, clones and
    residential clones, stages 39-43). Each such household descends from one
    stage-18 household with the same ``household_source_id``,
    ``household_support_clone_index`` and ``spi_income_band_donor_lower_bound``;
    a stage-18 household's weight is the sum of its descendants' initial
    weights. Identifiers are the build identifiers the stage keyed its draws on.
    Pay is the pay before the salary-sacrifice conversion (stage 45) wherever
    that rebuild recovers it.
    """
    if not _SALSAC_STATE:
        salary_sacrifice()
    t = release("person", "household")
    person, household = t["person"].copy(), t["household"].copy()
    iw = initial_household_weights(household).to_numpy()
    later = (household["household_is_high_value_support_copy"]
             | household["household_is_cgt_support_copy"]
             | household["household_is_capital_gains_clone"]
             | household["household_is_cgt_residential_clone"]).to_numpy(dtype=bool)
    key = pd.MultiIndex.from_arrays([household["household_source_id"],
                                     household["household_support_clone_index"],
                                     household["spi_income_band_donor_lower_bound"]])
    assert key[~later].is_unique
    w18 = pd.Series(iw, index=key).groupby(level=[0, 1, 2]).sum().reindex(key[~later])
    person["employment_income"] = _SALSAC_STATE["pay_pre"]
    # property_components (stage 24) adds the finance costs it draws to the
    # property_income of every row outside the SPI tape concept, whose costs
    # start at zero (property_components.py:968-1030); tape rows keep theirs.
    tape = person["person_is_spi_income_recipient"].to_numpy(dtype=bool)
    costs = pd.to_numeric(person["property_finance_costs"]).to_numpy(dtype=float)
    person["property_income"] = (pd.to_numeric(person["property_income"]).to_numpy(dtype=float)
                                 - np.where(tape, 0.0, costs))
    hh_build = household.set_index("household_id")[BUILD_ID["household"]]
    keep_ids = household.loc[~later, "household_id"]
    person = person.loc[person["person_household_id"].isin(keep_ids)].copy()
    household = household.loc[~later].copy()
    bu = release("benunit")["benunit"].set_index("benunit_id")[BUILD_ID["benunit"]]
    person["person_household_id"] = person["person_household_id"].map(hh_build)
    person["person_benunit_id"] = person["person_benunit_id"].map(bu)
    person["person_id"] = person[BUILD_ID["person"]]
    household["household_id"] = household[BUILD_ID["household"]]
    weights = pd.Series(w18.to_numpy(), index=household["household_id"].to_numpy())
    return person.reset_index(drop=True), household.reset_index(drop=True), weights


def spi_housing_shell():
    """``impute_spi_housing_shell`` (``spi_housing_shell.py:505-580`` at 67ee2ddf3).

    Trains on the FRS-channel households with positive weight and draws every
    SPI-channel household's housing: categories by weighted gradient-boosted
    classifiers, then 12 amounts (10 housing amounts and the two housing-linked
    benefit reports) by one ``RegimeGatedQRF(n_estimators=100, seed=0)`` fit
    on the one-hot predictors and categories (``fit_housing_model``,
    ``spi_housing_shell.py:404-470``). The rebuild runs those functions on the
    stage-18 frame from ``stage18_frame`` and checks it against the stage's
    receipt (household counts, weighted category shares) and the redrawn SPI
    households against the release.
    """
    from microcosm.build.uk_runtime import spi_housing_shell as H

    person, household, weights = stage18_frame()
    channel = household["household_support_channel"].astype(str).to_numpy()
    train_ids = household.loc[(channel == H.FRS_CHANNEL)
                              & (weights.to_numpy() > 0.0), "household_id"]
    rec_ids = household.loc[channel == H.SPI_CHANNEL, "household_id"]
    predictors = H.household_housing_predictors(person, household)
    targets = H.household_housing_targets(person, household)
    with capture.recording(fit_for_real=True) as caps:
        model = H.fit_housing_model(predictors.loc[train_ids], targets.loc[train_ids],
                                    weights.loc[train_ids].to_numpy(), seed=0,
                                    n_estimators=100)
        raw = model.draw(predictors.loc[rec_ids])
    assert len(caps) == 1 and len(caps[0].recipients) == 1
    spec = caps[0]
    draws, _ = H.apply_structural_rules(raw, predictors["region"], model.structural_zeros)
    receipt = stage_receipt("spi_housing_shell")
    shares = {}
    for column, rec_shares in receipt["category_shares"].items():
        mine = H._weighted_shares(targets.loc[train_ids, column], weights.loc[train_ids])
        shares[column] = max(abs(mine.get(k, 0.0) - v) for k, v in rec_shares["training"].items())
    released = targets.loc[rec_ids]
    agree = {}
    for column in (*H.HOUSING_CATEGORICAL_COLUMNS, *model.amount_targets):
        a, b = draws[column].to_numpy(), released[column].to_numpy()
        if column in H.HOUSING_CATEGORICAL_COLUMNS:
            agree[column] = float(np.mean(a.astype(str) == b.astype(str)))
        else:
            agree[column] = float(np.mean(np.isclose(a.astype(float), b.astype(float),
                                                     rtol=1e-9, atol=1e-6)))
    check = {"receipt_training_households": receipt.get("training_households"),
             "training_households": int(len(train_ids)),
             "receipt_recipient_households": receipt["recipient_households"],
             "recipient_households": int(len(rec_ids)),
             "max_abs_training_share_gap_vs_receipt": shares,
             "redraw_equals_release_share": agree}
    rec = spec.recipients[0]
    rw = weights.loc[rec_ids].to_numpy()
    # Production recipients: the features as drawn; target columns hold the
    # release's values (equal to the redraw where the check says so).
    rtab = rec["table"].loc[:, list(spec.predictors)].copy()
    for target in spec.targets:
        rtab[target] = released[target].to_numpy(dtype=float)
    spec.recipients = [{
        "table": rtab, "index": rec_ids.to_numpy(), "entity": "household",
        "method": "predict_from_uniforms (stage-18 rebuild)", "weight": rw,
        "in_production": np.ones(len(rtab), dtype=bool),
        "weight_source": "stage-18 weight: initial weights summed over descendants",
        "check": agree}]
    return [("uk_spi_housing_shell", spec, {
        "stage": "spi_housing_shell", "fit": "fit_housing_model amounts",
        "rebuilt_from_release": True, "check": check,
        "_groups": train_ids.astype(str).to_numpy()})]


def spi_stage2():
    """The FRS-only fill, stage 2 of ``impute_uk_spi_income_support``.

    ``spi_income.py:1154-1262`` at 67ee2ddf3 (stage 17 of 49,
    ``hmrc_spi_income_spine``). Training rows (``spi_income.py:954-972``):
    FRS-channel people in FRS-channel households. Predictors
    (``_person_predictors``, 2172-2195): age, gender, region and the six FRS
    income columns (``spi_support.py:142-149``), plus the State Pension receipt
    bridge, which the stage turns on because its operation lists
    ``state_pension_receipt`` (``spi_spine.py:801-806``; the receipt's
    ``pension_receipt_bridge`` records it). Gender and region are one-hot
    encoded jointly over training and query rows, columns sorted
    (``_encode_predictor_frames``, 2420-2442). Targets: the FRS-only columns
    present (``_stage2_outputs``, 2479-2504), owner-manager flags as 0/1.
    Weight: the household weight at the stage, importance kind. One
    ``RegimeGatedQRF(n_estimators=100, seed=43)`` (seed + 1, checked at
    ``spi_spine.py:1180``). Query rows: SPI-channel people aged 16 and over,
    drawn with uniforms keyed on the build ``person_id`` and salted per target;
    the rows of the recipients (``person_is_spi_income_recipient``) keep their
    draws.

    The rebuild works on ``stage18_frame`` (stage 18 changes no person income
    or weight, so its frame equals stage 17's output) and undoes, on top of
    what that frame undoes, the one later rewrite of a training row's stage-2
    columns: the salary-sacrifice stage (45) moved converted employee
    contributions into salary sacrifice and drew salary sacrifice for people
    not asked (``salary_sacrifice.py:592-689``). Training rows take
    ``pension_contributions_via_salary_sacrifice`` from the FRS job table and
    get the moved amount back on ``employee_pension_contributions``. No other
    stage after 17 writes a stage-2 column on an FRS-channel row: the take-up
    stages read the reported columns as anchors, ``spi_benefit_coherence``
    rewrites SPI rows only and checks base rows unchanged
    (``spi_benefit_coherence.py:340-395``), and the housing shell writes SPI
    households. Query rows' savings interest at the time of the draw is the
    taxable draw, the release's value less the tax-free part
    (``spi_income.py:1152-1153, 1279-1282``).

    ``check`` compares the training and recipient counts with the stage's
    receipt and the redrawn recipients with the release, column by column; a
    column a later step rewrites on SPI rows (pension coherence, owner-manager
    rules and the State Pension age guard inside stage 17; the housing shell;
    the UC redraw; benefit coherence; salary sacrifice) can disagree.
    """
    import microcosm.build.uk_runtime.spi_income as SI
    from microcosm.build.stochastic_assignment import (
        integer_record_keys,
        stable_identity_uniforms,
    )
    from microcosm.build.uk_runtime.spi_support import (
        BASE_FRS_SUPPORT_CHANNEL,
        FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS,
        SPI_SYNTHETIC_SUPPORT_CHANNEL,
        support_channel_column,
    )
    from microcosm.frame import WeightKind

    person, household, weights = stage18_frame()
    household = household.copy()
    household["household_weight"] = weights.reindex(household["household_id"]).to_numpy()
    pchan = person[support_channel_column("person")].astype(str)
    hchan = household[support_channel_column("household")].astype(str)
    base_people = pchan == BASE_FRS_SUPPORT_CHANNEL
    base_household = hchan == BASE_FRS_SUPPORT_CHANNEL
    assert "clone_index" not in household.columns  # spi_income.py:966-967 then adds nothing
    training_people = base_people & person["person_household_id"].isin(
        set(household.loc[base_household, "household_id"]))

    # Undo the salary-sacrifice stage on training rows.
    ss_col = "pension_contributions_via_salary_sacrifice"
    moved = person["person_id"].map(_SALSAC_STATE["moved"]).fillna(0.0).to_numpy(dtype=float)
    released_employee = pd.to_numeric(person["employee_pension_contributions"]).to_numpy(dtype=float)
    released_ss = pd.to_numeric(person[ss_col]).to_numpy(dtype=float)
    frs_ss = person["person_source_id"].map(frs_job_salary_sacrifice()).fillna(0.0).to_numpy()
    tr = training_people.to_numpy(dtype=bool)
    person["employee_pension_contributions"] = released_employee + np.where(tr, moved, 0.0)
    person[ss_col] = np.where(tr, frs_ss, released_ss)
    # The five disability amounts the release drops: the FRS person's own
    # amounts (no stage between the spine and stage 17 writes them; a stage-2
    # draw replaces them on recipient rows, so their values there never enter).
    dropped = ("attendance_allowance", "dla_sc", "dla_m", "pip_m", "pip_dl")
    amounts = frs_benefit_amounts(dropped)
    not_in_release = [f"{n}_reported" for n in dropped]
    assert not set(not_in_release) & set(person.columns)
    for column in not_in_release:
        person[column] = person["person_source_id"].map(amounts[column]).fillna(0.0).to_numpy()
    category = {"attendance_allowance_reported": "aa_category"}
    category.update({f"{n}_reported": f"{n}_category" for n in dropped[1:]})
    disability_check = {}
    for column, cat in category.items():
        amt = person.loc[training_people, column].to_numpy(dtype=float)
        has_cat = person.loc[training_people, cat].astype(str).to_numpy() != "NONE"
        disability_check[column] = {
            "category_implies_amount": float(np.mean(amt[has_cat] > 0.0)) if has_cat.any() else None,
            "amount_rows_with_category": float(np.mean(has_cat[amt > 0.0])) if (amt > 0).any() else None,
            "rows_with_amount": int((amt > 0).sum()), "rows_with_category": int(has_cat.sum())}

    stage2_outputs, reviewed_absent = SI._stage2_outputs(person)
    person = SI._owner_manager_flags_as_numbers(person)
    spi_channel = pchan == SPI_SYNTHETIC_SUPPORT_CHANNEL
    spi_age_domain = spi_channel & person["age"].ge(SI.SPI_MINIMUM_RECIPIENT_AGE)
    spi_people = person[SI.SPI_INCOME_RECIPIENT_COLUMN].astype(bool)
    assert not (spi_people & ~spi_age_domain).any()
    released_sav = pd.to_numeric(person["savings_interest_income"]).to_numpy(dtype=float)
    tax_free = pd.to_numeric(person["tax_free_savings_income"]).to_numpy(dtype=float)
    sp = spi_people.to_numpy(dtype=bool)
    person["savings_interest_income"] = np.where(sp, released_sav - tax_free, released_sav)

    income = FRS_ONLY_SPI_FILL_INCOME_PREDICTOR_COLUMNS
    train_predictors = SI._person_predictors(person.loc[training_people], household,
                                             income_predictors=income)
    target_predictors = SI._person_predictors(person.loc[spi_age_domain], household,
                                              income_predictors=income)
    positions = np.flatnonzero(spi_people.loc[spi_age_domain].to_numpy(dtype=bool))
    train_receipt = person.loc[training_people, "state_pension_reported"].gt(0)
    target_receipt = person.loc[spi_age_domain, SI.SPI_HMRC_STATE_PENSION_INCOME_COLUMN].gt(0)
    train_predictors["state_pension_receipt"] = train_receipt.astype(float)
    target_predictors["state_pension_receipt"] = target_receipt.astype(float)
    encoded_train, encoded_target = SI._encode_predictor_pair(train_predictors,
                                                              target_predictors)
    training_targets = person.loc[training_people, list(stage2_outputs)].copy()
    assert np.isfinite(training_targets.to_numpy(dtype=np.float64)).all()
    assert (training_targets.to_numpy(dtype=np.float64) >= 0.0).all()
    person_weights = SI._person_household_weights(person, household)
    fit_frame = SI._person_fit_frame(
        predictors=encoded_train, targets=training_targets,
        weights=person_weights.loc[training_people].to_numpy(dtype=np.float64),
        weight_kind=WeightKind.IMPORTANCE)
    with capture.recording(fit_for_real=True) as caps:
        stage2 = SI._qrf_class()(n_estimators=100, seed=43).fit(
            fit_frame, list(encoded_train.columns), list(stage2_outputs), weights="importance")
    assert len(caps) == 1
    spec = caps[0]
    keys = integer_record_keys(person.loc[spi_age_domain, "person_id"].to_numpy(),
                               label="SPI stage-2 person_id")
    draws = stage2.predict_from_uniforms(
        encoded_target,
        quantiles={t: stable_identity_uniforms(keys, seed=43, salt=f"{SI.FRS_ONLY_FIT_NAME}:{t}")
                   for t in stage2_outputs},
        sign_uniforms={t: stable_identity_uniforms(
            keys, seed=43, salt=f"{SI.FRS_ONLY_FIT_NAME}:{t}:sign") for t in stage2_outputs})
    draws = draws.iloc[positions].reset_index(drop=True)
    released = person.loc[spi_people, list(stage2_outputs)].reset_index(drop=True)
    agree = {t: (None if t in not_in_release else
                 float(np.mean(np.isclose(draws[t].to_numpy(dtype=float),
                                          released[t].to_numpy(dtype=float),
                                          rtol=1e-9, atol=1e-6))))
             for t in stage2_outputs}
    receipt = stage_receipt("hmrc_spi_income_spine")
    bridge = receipt["pension_receipt_bridge"]
    unconverted_base = tr & (moved == 0.0)
    check = {
        "receipt": {"recipient_rows": receipt["recipient_domain"]["recipient_rows"],
                    "training_positive_receipt_rows": bridge["training_positive_rows"],
                    "recipient_positive_receipt_rows": bridge["recipient_positive_rows"]},
        "rebuild": {"recipient_rows": int(sp.sum()), "training_rows": int(tr.sum()),
                    "training_positive_receipt_rows": int(train_receipt.sum()),
                    "recipient_positive_receipt_rows": int(
                        target_receipt.iloc[positions].sum())},
        "training_ss_release_equals_frs_capped_at_pay_unconverted": float(np.mean(np.isclose(
            released_ss[unconverted_base],
            np.minimum(frs_ss[unconverted_base],
                       np.maximum(_pay_release(person)[unconverted_base], 0.0)),
            atol=1e-6))),
        "training_rows_converted": int((tr & (moved > 0.0)).sum()),
        "reviewed_absent_outputs": sorted(reviewed_absent),
        "dropped_disability_amounts_from_frs": disability_check,
        "redraw_equals_release_share": agree,
    }
    rtab = encoded_target.iloc[positions].reset_index(drop=True).copy()
    for t in stage2_outputs:
        rtab[t] = draws[t].to_numpy(dtype=float)
    rw = person_weights.loc[spi_people].to_numpy(dtype=np.float64)
    spec.recipients = [{
        "table": rtab.loc[:, list(dict.fromkeys([*spec.predictors, *spec.targets]))],
        "index": np.flatnonzero(sp), "entity": "person",
        "method": "predict_from_uniforms (stage-18 rebuild); targets hold the redraws",
        "weight": rw, "in_production": np.ones(len(rtab), dtype=bool),
        "weight_source": "stage-18 household weight of the person's household",
        "check": agree}]
    groups = person.loc[training_people, "person_household_id"].astype(str).to_numpy()
    return [("uk_spi_stage2", spec, {
        "stage": "hmrc_spi_income_spine", "fit": "FRS-only fill (stage 2)",
        "rebuilt_from_release": True, "check": check, "_groups": groups})]


#: Columns stages 31-48 create (``produced`` in the build record's stage
#: records): absent from the frame stage 30 saw. The salary-sacrifice stage
#: also lists three columns it rewrites, which ``stage30_frame`` restores.
_CREATED_AFTER_30 = {
    "benunit": ("uc_reported_capital", "pension_credit_reported_capital",
                "would_claim_uc_at_legacy_closure",
                "has_mixed_age_couple_pension_credit_saving", "uc_deduction_random_draw",
                "uc_deduction_type_random_draw", "uc_latent_deduction_rate",
                "uc_deduction_combination"),
    "household": ("household_is_cgt_support_copy", "cgt_support_copies",
                  "cgt_support_copy_index", "household_is_capital_gains_clone",
                  "household_is_cgt_residential_clone", "cgt_residential_clone_index",
                  "cgt_residential_probability"),
    "person": ("capital_gains", "capital_gains_residential_property",
               "capital_gains_asset_type", "capital_gains_badr",
               "pension_contributions_via_bonus_sacrifice",
               "salary_sacrifice_pre_conversion_pay", "student_loan_plan",
               "attends_private_school"),
}


def stage30_frame():
    """The frame ``uc_reporter_redraw`` (stage 30 of 49) saw, rebuilt as a Frame.

    Households: the release's less the CGT support copies, incidence clones
    and residential arms (stages 39-43); each descends from one stage-30
    household with the same source id, support clone index, band bound and
    high-value copy index, whose weight is the sum of its descendants' initial
    weights (the CGT anchor moves mass only within a clone pair). Columns:
    those stages 31-48 create are dropped; the salary-sacrifice stage's
    rewrites (pay, employee contributions, salary sacrifice) are undone as in
    ``salary_sacrifice`` and ``spi_stage2``; the copies' childcare take-up,
    which ``childcare_copy_take_up`` (stage 47) re-drew, goes back to the host
    unit's draws every copy inherited (``childcare_copy_take_up.py:1-30``).
    Not undone: ``spi_benefit_coherence`` (stage 31), which rewrote the SPI
    rows' benefit reports and disability categories after this stage; the
    redraw's SPI screen can differ through it, and ``check`` shows by how much.
    Identifiers are build identifiers, groups sorted ascending as the Frame
    requires.
    """
    from microcosm.build.uk_runtime.childcare_copy_take_up import copy_benunits
    from microcosm.build.uk_runtime.frs_take_up import (
        UK_CHILDCARE_HOURS_OUTPUT,
        UK_CHILDCARE_TAKE_UP_FLAGS,
        UK_UNDER_TWO_USAGE_OUTPUT,
    )
    from microcosm.build.uk_runtime.national_frame import uk_national_frame
    from microcosm.frame import WeightKind

    if not _SALSAC_STATE:
        salary_sacrifice()
    t = release("person", "benunit", "household")
    person, benunit, household = (t["person"].copy(), t["benunit"].copy(),
                                  t["household"].copy())
    iw = initial_household_weights(household).to_numpy()
    later = (household["household_is_cgt_support_copy"]
             | household["household_is_capital_gains_clone"]
             | household["household_is_cgt_residential_clone"]).to_numpy(dtype=bool)
    key = pd.MultiIndex.from_arrays([
        household["household_source_id"], household["household_support_clone_index"],
        household["spi_income_band_donor_lower_bound"].fillna(-1.0),
        household["high_value_support_copy_index"].fillna(-1)])
    assert key[~later].is_unique
    w30 = pd.Series(iw, index=key).groupby(level=[0, 1, 2, 3]).sum().reindex(key[~later])
    assert np.isclose(w30.sum(), iw.sum()) and not w30.isna().any()

    # Salary sacrifice (stage 45) undone.
    ss_col = "pension_contributions_via_salary_sacrifice"
    moved = _SALSAC_STATE["moved"].to_numpy()
    person["employment_income"] = _SALSAC_STATE["pay_pre"]
    person["employee_pension_contributions"] = (
        pd.to_numeric(person["employee_pension_contributions"]).to_numpy(dtype=float) + moved)
    frs_ss = person["person_source_id"].map(frs_job_salary_sacrifice()).fillna(0.0).to_numpy()
    base_person = person["person_support_channel"].astype(str).to_numpy() == "frs"
    person[ss_col] = np.where(base_person, frs_ss,
                              pd.to_numeric(person[ss_col]).to_numpy(dtype=float))

    keep_ids = set(household.loc[~later, "household_id"])
    person = person.loc[person["person_household_id"].isin(keep_ids)].copy()
    benunit = benunit.loc[benunit["benunit_id"].isin(set(person["person_benunit_id"]))].copy()
    household = household.loc[~later].copy()
    # Childcare copy take-up (stage 47) undone: copies take their host's draws.
    # ``copy_benunits`` reads the CGT lineage columns (stages 39-43 create
    # them), so it runs before those columns are dropped; on the kept rows the
    # CGT flags are all False, so they add no copy path.
    copies = copy_benunits(person, benunit, household)
    for entity, frame in (("person", person), ("benunit", benunit), ("household", household)):
        frame.drop(columns=[c for c in _CREATED_AFTER_30[entity] if c in frame.columns],
                   inplace=True)

    native = ~copies
    host = benunit.loc[native].set_index("benunit_source_id")
    assert host.index.is_unique
    columns = [o for o, _ in UK_CHILDCARE_TAKE_UP_FLAGS] + [UK_CHILDCARE_HOURS_OUTPUT]
    src = benunit.loc[copies, "benunit_source_id"]
    has_host = src.isin(host.index).to_numpy()
    for column in columns:
        values = benunit[column].to_numpy(copy=True)
        rows = np.flatnonzero(copies)[has_host]
        values[rows] = host.loc[src[has_host], column].to_numpy()
        benunit[column] = values
    copy_unit_ids = set(benunit.loc[copies, "benunit_id"])
    copy_people = person["person_benunit_id"].isin(copy_unit_ids).to_numpy()
    native_people = person.loc[~copy_people].set_index("person_source_id")
    assert native_people.index.is_unique
    psrc = person.loc[copy_people, "person_source_id"]
    p_has = psrc.isin(native_people.index).to_numpy()
    usage = person[UK_UNDER_TWO_USAGE_OUTPUT].to_numpy(dtype=float, copy=True)
    usage[np.flatnonzero(copy_people)[p_has]] = native_people.loc[
        psrc[p_has], UK_UNDER_TWO_USAGE_OUTPUT].to_numpy(dtype=float)
    person[UK_UNDER_TWO_USAGE_OUTPUT] = usage
    childcare = {"copy_benunits": int(copies.sum()), "copies_with_host": int(has_host.sum()),
                 "copy_people": int(copy_people.sum()), "copy_people_with_host": int(p_has.sum())}

    # Build identifiers, sorted.
    hh_build = household.set_index("household_id")[BUILD_ID["household"]]
    bu_build = benunit.set_index("benunit_id")[BUILD_ID["benunit"]]
    person["person_household_id"] = person["person_household_id"].map(hh_build)
    person["person_benunit_id"] = person["person_benunit_id"].map(bu_build)
    person["person_id"] = person[BUILD_ID["person"]]
    benunit["benunit_id"] = benunit[BUILD_ID["benunit"]]
    household["household_id"] = household[BUILD_ID["household"]]
    weights = pd.Series(w30.to_numpy(), index=household["household_id"].to_numpy())
    person = person.sort_values("person_id").reset_index(drop=True)
    benunit = benunit.sort_values("benunit_id").reset_index(drop=True)
    household = household.sort_values("household_id").reset_index(drop=True)
    weights = weights.reindex(household["household_id"].to_numpy())
    household = household.drop(columns=["household_weight"])
    frame = uk_national_frame(person=person, benunit=benunit, household=household,
                              time_period="2024", weight_kind=WeightKind.IMPORTANCE,
                              household_weights=weights.to_numpy())
    return frame, {"childcare_copy_revert": childcare,
                   "households": int(len(household)), "weight_total": float(weights.sum())}


def uc_reporter_redraw():
    """``redraw_spi_reported_uc`` (``uc_reporter_redraw.py:188-400`` at 67ee2ddf3).

    Runs production's own stage function on ``stage30_frame`` with
    production's engine adapter (``PolicyEngineUKEngine``, ``spine_build.py:
    712-725``; policyengine-uk 2.151.0, the build's pin) and records the QRF
    fit. Training: screened canonical FRS benefit units (positive pre-take-up
    award and a non-child member, ``uc_reporter_redraw.py:248-300``); target:
    the unit's reported UC; predictors: the nine of ``UC_REPORTER_PREDICTORS``
    (region one-hot); weight: household weight per benefit unit; one
    ``RegimeGatedQRF(seed=44)`` with its default 100 trees. Recipients: the
    screened SPI benefit units. ``check`` compares the rebuild's counts and
    reporter transitions with the stage's receipt and its SPI draws with the
    release.
    """
    from microcosm.build.uk_runtime import uc_reporter_redraw as U
    from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

    frame, info = stage30_frame()
    with capture.recording(fit_for_real=True) as caps:
        result = U.redraw_spi_reported_uc(frame, engine=PolicyEngineUKEngine())
    assert len(caps) == 1
    spec = caps[0]
    receipt = stage_receipt("uc_reporter_redraw")
    mine = result.evidence()
    cells = {}
    for channel, fams in receipt["reporter_transitions"].items():
        for fam, counts in fams.items():
            for k, v in counts.items():
                got = mine["reporter_transitions"].get(channel, {}).get(fam, {}).get(k)
                if got != v:
                    cells[f"{channel}/{fam}/{k}"] = {"receipt": v, "rebuild": got}
    out_person = result.frame.table("person")
    rel = release("person")["person"].set_index(BUILD_ID["person"])
    uc_rebuild = out_person.set_index("person_id")[U.UC_REPORTER_REDRAW_OUTPUT]
    spi_rows = out_person["person_support_channel"].astype(str).eq("spi").to_numpy()
    ids = out_person.loc[spi_rows, "person_id"]
    a = uc_rebuild.loc[ids].to_numpy(dtype=float)
    b = pd.to_numeric(rel.loc[ids, U.UC_REPORTER_REDRAW_OUTPUT]).to_numpy(dtype=float)
    check = {
        "receipt": {k: receipt[k] for k in ("training_benunits", "screened_spi_benunits",
                                            "screen_failed_spi_benunits")},
        "rebuild": {k: mine[k] for k in ("training_benunits", "screened_spi_benunits",
                                         "screen_failed_spi_benunits")},
        "transition_cells_differing": cells,
        "spi_person_uc_equals_release_share": float(np.mean(np.isclose(a, b, atol=1e-6))),
        "spi_uc_total_rebuild_vs_release_initial_weights": None,
        **info,
    }
    if len(spec.recipients) != 1:
        raise RuntimeError("the recorder should keep exactly one recipient set")
    rec = spec.recipients[0]
    bweights = np.asarray(U._household_to_benunit_weights(
        frame.table("benunit"), person=frame.table("person"),
        household=frame.table("household"),
        household_weights=frame.weights_for("household").values), dtype=np.float64)
    rec.update({"weight": bweights[np.asarray(rec["index"])], "entity": "benunit",
                "in_production": np.ones(len(rec["table"]), dtype=bool),
                "weight_source": "stage-30 household weight per benefit unit",
                "check": {k: check[k] for k in ("receipt", "rebuild")}})
    check["spi_uc_drawn_total_stage30_weights"] = float(
        (rec["weight"] * rec["table"][U.UC_REPORTER_TARGET].to_numpy(dtype=float)).sum())
    return [("uk_uc_reporter_redraw", spec, {
        "stage": "uc_reporter_redraw", "fit": "redraw_spi_reported_uc",
        "rebuilt_from_release": True, "check": check})]


def lcfs_road_fuel():
    """The LCFS stage's zero-road-fuel redraw (``lcfs_consumption.py:1331-1480``).

    For a household the fuel flag marks whose chain draw of petrol plus
    diesel is zero, the stage redraws the total from a second
    ``RegimeGatedQRF`` fit on the donors with positive road fuel, on the
    chain's encoded predictors at the donor household weight: targets the
    road-fuel total (positive by construction) and the petrol share (lines
    1392-1407). Its training rows are the chain's own donors (captured as
    ``uk_lcfs`` from the stage) with positive petrol plus diesel; seed and
    trees come from the stage's receipt (``road_fuel_incidence``), which also
    records the training rows (rounded) and the two regimes the rebuild must
    match. Production recipients: none here; the flagged households' chain
    predictors include the engine's ``hbai_household_net_income`` at stage 27,
    as for the chain itself.
    """
    from microcosm.build.uk_runtime import lcfs_consumption as L
    from microcosm.fit.qrf import RegimeGatedQRF

    table, meta = capture.load("uk_lcfs")
    receipt = stage_receipt("lcfs_consumption")["road_fuel_incidence"]
    predictors = list(meta["predictors"])
    petrol = table["petrol_spending"].to_numpy(dtype=float)
    total = petrol + table["diesel_spending"].to_numpy(dtype=float)
    pos = total > 0
    train = table.loc[pos, predictors].reset_index(drop=True)
    train[L.UK_LCFS_ROAD_FUEL_TOTAL] = total[pos]
    train[L.UK_LCFS_PETROL_SHARE] = petrol[pos] / total[pos]
    train["household_weight"] = table.loc[pos, "__weight__"].to_numpy(dtype=float)
    targets = [L.UK_LCFS_ROAD_FUEL_TOTAL, L.UK_LCFS_PETROL_SHARE]
    with capture.recording(fit_for_real=True) as caps:
        fitted = RegimeGatedQRF(n_estimators=int(receipt["n_estimators"]),
                                seed=int(receipt["seed"])).fit(
            train, predictors, targets, weights="household_weight")
    spec = caps[0]
    regimes = {t: str(r) for t, r in fitted.regimes().items()}
    check = {"receipt_training_rows_rounded": receipt["training_rows"],
             "training_rows": int(pos.sum()),
             "receipt_regimes": receipt["regimes"], "regimes": regimes,
             "receipt_redrawn_total_gbp": receipt["weighted_total_after"]
             - receipt["weighted_total_before"]}
    assert regimes == receipt["regimes"], check
    return [("uk_lcfs_road_fuel", spec, {
        "stage": "lcfs_consumption", "fit": "redraw_zero_road_fuel",
        "rebuilt_from_release": False, "check": check})]


def _pay_release(person: pd.DataFrame) -> np.ndarray:
    """Release pay of each row of a ``stage18_frame`` person table."""
    rel = release("person")["person"]
    pay = pd.Series(pd.to_numeric(rel["employment_income"]).to_numpy(),
                    index=rel[BUILD_ID["person"]].to_numpy())
    return person["person_id"].map(pay).to_numpy(dtype=float)


NTS_DIR = Path.home() / "microcosm-uk-work/inputs/nts_2002_2024"
NTS_TABS = {"nts_household_tab": "household_eul_2002-2024.tab",
            "nts_individual_tab": "individual_eul_2002-2024.tab",
            "nts_trip_tab": "trip_eul_2002-2024.tab",
            "nts_stage_tab": "stage_eul_2002-2024.tab",
            "nts_ticket_tab": "ticket_eul_2002-2024.tab"}


def nts_bus_band():
    """``impute_bus_use_band`` (``nts_bus_travel.py:1185-1240`` at 67ee2ddf3).

    The NTS stage (26 of 49) draws each English FRS person's interview bus-use
    band, an ordinal 0-6 (0 = non-user), from one ``RegimeGatedQRF`` fit on
    NTS persons (zero-inflated positive regime), then reads trips off band
    means (``assign_trips_from_band_means``, line 1513). Donor: production's
    own cleaner (``clean_nts_travel_tables``) on the five UKDS SN 5340 tabs,
    each verified against the stage's pinned digest by ``read_pinned_tab``.
    Recipients: ``recipient_predictors`` (lines 1068-1150) on the stage-30
    frame with production's engine adapter. Stages 27-30 add no households,
    so the households and weights equal stage 26's; the income band reads the
    engine's HBAI gross income, which the UC redraw (stage 30) can move for SPI
    rows, so ``check`` compares the rebuilt draws with the release's bands.
    Groups: NTS households.
    """
    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime import nts_bus_travel as N
    from microcosm.build.uk_runtime.frs_spine import read_pinned_tab
    from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

    stage = next(s for s in load_country_spec("uk").sources.stages
                 if s.stage == "nts_bus_travel")
    derive = N._operation(stage, N.CLEAN_NTS_TRAVEL_TABLES_KIND)
    columns = N.NTSColumns.from_parameters(derive)
    fare_cap = N.FareCapRegime.from_parameters(derive, survey_years=columns.survey_years)
    cars_clip = tuple(int(v) for v in derive.get("cars_clip", (0, 5)))
    tabs = {role: read_pinned_tab(NTS_DIR / name, N._artifact(stage, role))
            for role, name in NTS_TABS.items()}
    donor = N.clean_nts_travel_tables(
        tabs["nts_household_tab"], tabs["nts_individual_tab"], tabs["nts_trip_tab"],
        tabs["nts_stage_tab"], tabs["nts_ticket_tab"], columns=columns,
        cars_clip=(cars_clip[0], cars_clip[1]), fare_cap=fare_cap)
    del tabs
    if donor.frequency_source != N.FREQUENCY_SOURCE_INTERVIEW:
        raise RuntimeError("production drew the band from the QRF; this donor would not")
    band = N._operation(stage, N.IMPUTE_BUS_USE_BAND_KIND)
    frame, info = stage30_frame()
    engine = PolicyEngineUKEngine()
    recipient = N.recipient_predictors(
        frame, engine, columns=columns,
        income_band_edges=[float(v) for v in band["income_band_edges"]],
        region_remap={str(k): str(v) for k, v in
                      N._mapping(band.get("region_remap", {}), "region_remap").items()},
        cars_clip=(cars_clip[0], cars_clip[1]))
    seed, n_estimators = int(band.get("seed", 0)), int(band.get("n_estimators", 100))
    with capture.recording(fit_for_real=True) as caps:
        imputation = N.impute_bus_use_band(donor.person, recipient, seed=seed,
                                           n_estimators=n_estimators)
    assert len(caps) == 1
    spec = caps[0]
    receipt = stage_receipt("nts_bus_travel")["band"]
    rel = release("person")["person"].set_index(BUILD_ID["person"])
    released = pd.to_numeric(rel.loc[recipient["person_id"].to_numpy(), N.BAND_COLUMN]
                             ).to_numpy(dtype=float)
    mine = imputation.receipt
    check = {
        "receipt": {k: receipt[k] for k in ("donor_persons", "recipient_persons", "regime",
                                            "seed", "n_estimators")},
        "rebuild": {k: mine[k] for k in ("donor_persons", "recipient_persons", "regime",
                                         "seed", "n_estimators")},
        "donor_band_shares_max_abs_diff": float(max(
            abs(mine["donor_band_shares"][b] - receipt["donor_band_shares"][b])
            for b in receipt["donor_band_shares"])),
        "predictors_equal": list(mine["predictors"]) == list(receipt["predictors"]),
        "band_equals_release_share": float(np.mean(imputation.band == released)),
        **info,
    }
    if len(spec.recipients) != 1:
        raise RuntimeError("the recorder should keep exactly one recipient set")
    rec = spec.recipients[0]
    hh_w = pd.Series(frame.weights_for("household").values,
                     index=frame.table("household")["household_id"].to_numpy())
    pw = hh_w.reindex(recipient["person_household_id"].to_numpy()).to_numpy(dtype=np.float64)
    rec.update({"weight": pw[np.asarray(rec["index"])], "entity": "person",
                "in_production": np.ones(len(rec["table"]), dtype=bool),
                "weight_source": "stage-30 (= stage-26) household weight of the person",
                "check": {k: check[k] for k in ("receipt", "rebuild")}})
    groups = donor.person["household_id"].astype(str).to_numpy()
    if len(groups) != len(spec.table):
        raise RuntimeError("the fit's donor rows should be the cleaned NTS persons")
    return [("uk_nts_bus_band", spec, {
        "stage": "nts_bus_travel", "fit": "impute_bus_use_band",
        "rebuilt_from_release": True, "check": check, "_groups": groups})]


STAGES = {"salary_sacrifice": salary_sacrifice, "spi_housing_shell": spi_housing_shell,
          "spi_stage2": spi_stage2, "uc_reporter_redraw": uc_reporter_redraw,
          "lcfs_road_fuel": lcfs_road_fuel, "nts_bus_band": nts_bus_band}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=sorted(STAGES))
    ap.add_argument("--out", type=Path, default=ROOT / "results/capture_uk_release.json")
    args = ap.parse_args()
    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    for name in args.stages:
        t0 = time.time()
        for key, spec, extra in STAGES[name]():
            groups = extra.pop("_groups", None)
            capture.save(spec, key, {"country": "uk", "code": "67ee2ddf3", **extra})
            if groups is not None:
                assert len(groups) == len(spec.table)
                np.save(capture.DONORS / f"{key}__groups.npy", np.asarray(groups).astype(str))
                extra["groups"] = int(len(np.unique(groups)))
            w = spec.weights
            totals = {t: float(np.sum(w * spec.table[t].to_numpy(dtype=np.float64)))
                      for t in spec.targets}
            report[key] = {**spec.meta(), **extra, "donor_weighted_totals": totals,
                           "seconds": time.time() - t0}
            print(json.dumps({key: report[key]}, indent=1, default=str), flush=True)
        args.out.write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
