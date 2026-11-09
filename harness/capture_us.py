"""Capture the US refresh stages' QRF fits and check them against production.

Runs each stage's production donor loader and imputation function (populace
at cae8640, the commit that built the US national release's parent
``populace-us-2024-buildp-sparse-rmloss100-cae8640``) with every
``RegimeGatedQRF.fit`` recorded (``capture.recording``). Recipients are the
production H5's own tables, loaded the way the refresh tool's ``_load_frame``
loads its base (``tools/build_us_fiscal_refresh_release.py:2012-2029``).

For each stage it saves the captured donor table(s) under ``donors/`` and
compares the stage's output with the production H5 column: the share of
records equal to the last bit, the largest absolute difference and both
weighted totals. An exact match confirms the donor table, predictors,
targets, weights, seed and forest size all reproduce production's fit.

Seeds: the refresh tool passes ``--seed`` (default 0, line 1063) to every
stage except SIPP vehicles and SSI disability criteria, which pin 42.

Usage:
  python capture_us.py --stages scf_net_worth scf_financial_assets ... --out ../results/capture_us.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("POPULACE_FIT_N_JOBS", "4")
os.environ.setdefault("POPULACE_FIT_PREDICT_WORKERS", "4")

import prodpath  # noqa: E402

prodpath.us()

import capture  # noqa: E402

ROOT = HERE.parent
H5 = ROOT / "data/prod/us/populace_us_2024.h5"
CACHE = Path.home() / ".cache/microcosm"
SCF_SUMMARY = CACHE / "scf/rscfp2022.dta"
SCF_FULL = CACHE / "scf/p22i6.dta"
SIPP = Path.home() / (".cache/huggingface/hub/models--PolicyEngine--policyengine-us-data/"
                      "blobs/5c30439e365fc26483318ef61d1d8f4bb2f0e9d6bb47c22c06756a7698733ee2")
ORG = CACHE / "org/census_cps_org_2024_wages.csv.gz"
SIPP_SLIM = Path.home() / ".cache/populace/sipp/pu2023_slim.csv"  # the pinned tip donor
PERIOD = 2024
SEED = 0


def load_tables():
    tables = {k: pd.read_hdf(H5, key=k) for k in
              ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")}
    return tables


def load_frame(tables):
    from populace.frame import Frame, Weights, WeightKind
    from populace.frame.units import US_SCHEMA

    t = {k: v.copy() for k, v in tables.items()}
    weights = t["household"].pop("household_weight").to_numpy(dtype=np.float64)
    return Frame(t, US_SCHEMA, {"household": Weights(weights, WeightKind.CALIBRATED)})


def household_weight_of(table, tables, link):
    hh = tables["household"]
    w = pd.Series(hh["household_weight"].to_numpy(), index=hh["household_id"].to_numpy())
    return w.reindex(table[link].to_numpy()).to_numpy(dtype=np.float64)


def compare(mine, prod, weight, label, mask=None):
    mine = np.asarray(mine, dtype=np.float64)
    prod = np.asarray(prod, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if mask is not None:
        mine, prod, weight = mine[mask], prod[mask], weight[mask]
    equal = mine == prod
    out = {
        "label": label, "n": int(len(mine)), "share_exact": float(equal.mean()),
        "max_abs_diff": float(np.max(np.abs(mine - prod))) if len(mine) else 0.0,
        "weighted_total_mine": float((weight * mine).sum()),
        "weighted_total_prod": float((weight * prod).sum()),
    }
    out["weighted_total_rel_diff"] = (out["weighted_total_mine"] / out["weighted_total_prod"] - 1
                                      if out["weighted_total_prod"] else float("nan"))
    print(f"  {label}: exact {out['share_exact']:.4f} of {out['n']}, max |diff| "
          f"{out['max_abs_diff']:.3g}, weighted total rel diff {out['weighted_total_rel_diff']:+.2e}",
          flush=True)
    return out


def run_capture(fn):
    with capture.recording() as captured:
        result = fn()
    return result, captured


# ---------------------------------------------------------------- stages


def stage_scf_net_worth(tables, frame):
    from populace.build.us_runtime import scf_wealth as M

    donor = M.load_scf_2022_financial_asset_donor(SCF_SUMMARY)
    result, caps = run_capture(lambda: M.impute_us_scf_net_worth(
        tables["person"], tables["household"], donor, seed=SEED))
    hh = tables["household"]
    checks = [compare(result.to_numpy(), hh["net_worth"], hh["household_weight"], "net_worth")]
    return caps, checks


def _heads(tables):
    from populace.build.us_runtime import scf_wealth as M

    person = tables["person"]
    head = M._household_head_mask(person)
    hh_ids = person.loc[head, "person_household_id"]
    is_scf = M.financial_asset_source_is_scf(hh_ids, seed=SEED, time_period=PERIOD)
    return head, is_scf


def stage_scf_financial_assets(tables, frame):
    from populace.build.us_runtime import scf_wealth as M

    donor = M.load_scf_2022_financial_asset_donor(SCF_SUMMARY)
    result, caps = run_capture(lambda: M.impute_us_scf_financial_assets(
        tables["person"], donor, seed=SEED))
    person = tables["person"]
    head, is_scf = _heads(tables)
    w = household_weight_of(person, tables, "person_household_id")
    mask = np.zeros(len(person), dtype=bool)
    mask[np.flatnonzero(head)[is_scf]] = True
    checks = [compare(result[c], person[c], w, f"{c} (SCF-source heads)", mask)
              for c in M.US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS]
    return caps, checks


def stage_sipp_financial_assets(tables, frame):
    from populace.build.us_runtime import sipp_financial_assets as M
    from populace.build.us_runtime import scf_wealth as W

    donor = M.load_sipp_2023_financial_asset_donor(
        SIPP, expected_sha256=M.SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256,
        expected_size_bytes=M.SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES)
    result, caps = run_capture(lambda: M.impute_us_sipp_financial_assets(
        tables["person"], donor, seed=SEED))
    person = tables["person"]
    head, is_scf = _heads(tables)
    w = household_weight_of(person, tables, "person_household_id")
    mask = np.zeros(len(person), dtype=bool)
    mask[np.flatnonzero(head)[~is_scf]] = True
    checks = [compare(result[c], person[c], w, f"{c} (SIPP-source heads)", mask)
              for c in W.US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS]
    return caps, checks


def stage_scf_auto_loans(tables, frame):
    from populace.build.us_runtime import scf_auto_loans as M

    donor = M.load_scf_2022_auto_loan_donor(SCF_SUMMARY, SCF_FULL)
    result, caps = run_capture(lambda: M.impute_us_scf_auto_loans(frame, donor, seed=SEED))
    checks = []
    for c in result.columns:
        for entity, table in tables.items():
            if c in table.columns and len(table) == len(result):
                link = "household_id" if entity == "household" else f"person_household_id"
                w = (table["household_weight"] if entity == "household"
                     else household_weight_of(table, tables, link))
                checks.append(compare(result[c], table[c], w, f"{c} ({entity})"))
                break
    return caps, checks


def stage_sipp_vehicles(tables, frame):
    from populace.build.us_runtime import sipp_vehicles as M

    donor = M.load_sipp_2023_vehicle_donor(SIPP, expected_sha256=M.SIPP_2023_VEHICLE_DONOR_SHA256,
                                           expected_size_bytes=M.SIPP_2023_VEHICLE_DONOR_SIZE_BYTES)
    result, caps = run_capture(lambda: M.impute_us_sipp_vehicles(frame, donor, seed=42))
    hh = tables["household"]
    checks = [compare(result[c], hh[c], hh["household_weight"], c)
              for c in result.columns if c in hh.columns and len(result) == len(hh)]
    return caps, checks


def stage_sipp_tips(tables, frame):
    from populace.build.us_runtime import sipp_tips as M

    donor = M.load_sipp_2023_tip_donor(SIPP_SLIM, expected_sha256=M.SIPP_2023_TIP_DONOR_SHA256)
    result, caps = run_capture(lambda: M.impute_us_sipp_tips(tables["person"], donor, seed=SEED))
    person = tables["person"]
    w = household_weight_of(person, tables, "person_household_id")
    checks = [compare(result[c], person[c], w, c) for c in result.columns if c in person.columns]
    return caps, checks


def stage_org_wages(tables, frame):
    from populace.build.us_runtime import org_wages as M

    donor = M.load_org_2024_donor(ORG)
    result, caps = run_capture(lambda: M.impute_us_org_wages(frame, donor, seed=SEED))
    person = tables["person"]
    w = household_weight_of(person, tables, "person_household_id")
    checks = []
    parts = result if isinstance(result, tuple) else (result,)
    for part in parts:
        for c in part.columns:
            if c in person.columns and len(part) == len(person):
                checks.append(compare(part[c], person[c], w, c))
    return caps, checks


def _binary_stage(module, loader, imputer, seed, column_entity):
    def stage(tables, frame):
        import importlib

        M = importlib.import_module(f"populace.build.us_runtime.{module}")
        donor = getattr(M, loader)(SIPP)
        result, caps = run_capture(lambda: getattr(M, imputer)(frame, donor, seed=seed))
        entity, column = column_entity
        table = tables[entity]
        link = "household_id" if entity == "household" else (
            "person_household_id" if entity == "person" else None)
        checks = []
        if column in table.columns and len(result) == len(table):
            if entity == "household":
                w = table["household_weight"]
            elif entity == "person":
                w = household_weight_of(table, tables, link)
            else:
                person = tables["person"]
                key = f"person_{entity}_id"
                first = person.drop_duplicates(key).set_index(key)
                w = household_weight_of(first.reset_index(), tables, "person_household_id")
                w = pd.Series(w, index=first.index).reindex(table[f"{entity}_id"]).to_numpy()
            checks.append(compare(np.asarray(result, dtype=float), table[column], w, column))
        return caps, checks

    return stage


# ---------------------------------------------------------------- recipients

# How each stage's predict input lines up with the production tables:
# "person" inputs carry person-table labels; "household" inputs list every
# household in household-table order (scf_auto_loans indexes them by id).
RECIPIENT_KIND = {
    "scf_net_worth": "person", "scf_financial_assets": "person",
    "sipp_financial_assets": "person", "scf_auto_loans": "household",
    "sipp_vehicles": "household", "sipp_tips": "person", "org_wages": "person",
}


def _production_value(name, target, drawn):
    """The stage's own post-processing of a draw (code lines in the docstring above)."""
    if name == "sipp_vehicles":
        return np.clip(drawn, 0, None).astype(np.float32).astype(np.float64)
    if name in ("sipp_tips", "org_wages") and target != "is_paid_hourly":
        return np.maximum(drawn, 0.0)
    if target == "is_paid_hourly":
        return (drawn >= 0.5).astype(np.float64)
    return drawn


def annotate_recipients(name, caps, tables):
    """Attach production weights, the production mask and a draw check to each recipient set.

    Weights are the production H5's ``household_weight`` (persons take their
    household's). The production mask keeps the rows whose draw reaches the
    released column: SCF-source heads for the SCF financial-asset model,
    SIPP-source heads for the SIPP one (``scf_wealth.py``
    ``impute_us_sipp_scf_financial_assets``), and workers with positive
    earnings and hours for ORG wages (``org_wages.py:717-731`` zeroes the rest).
    The check compares each draw, after the stage's own clipping, with the
    production column on the masked rows.
    """
    kind = RECIPIENT_KIND.get(name)
    if kind is None:
        return
    person, hh = tables["person"], tables["household"]
    assert person.index.is_unique and hh.index.is_unique
    for spec in caps:
        for rec in spec.recipients:
            table, index = rec["table"], rec["index"]
            n = len(table)
            if kind == "person":
                rows = person.index.get_indexer(index)
                assert (rows >= 0).all(), "recipient labels outside the person table"
                weight = household_weight_of(person.iloc[rows], tables, "person_household_id")
                source = "H5 household_weight of the person's household"
            else:
                assert n == len(hh), (name, n, len(hh))
                if name == "scf_auto_loans":
                    assert (np.asarray(index) == hh["household_id"].to_numpy()).all()
                rows = np.arange(n)
                weight = hh["household_weight"].to_numpy(dtype=np.float64)
                source = "H5 household_weight"
            in_prod = np.ones(n, dtype=bool)
            if name in ("scf_financial_assets", "sipp_financial_assets"):
                head, is_scf = _heads(tables)
                assert (rows == np.flatnonzero(head)).all()
                in_prod = is_scf if name == "scf_financial_assets" else ~is_scf
            if name == "org_wages":
                in_prod = ((table["employment_income"].to_numpy() > 0)
                           & (table["weekly_hours_worked"].to_numpy() > 0))
            checks = {}
            for target in spec.targets:
                drawn = table[target].to_numpy(dtype=np.float64)
                mine = _production_value(name, target, drawn)
                if kind == "person" and target in person.columns:
                    prod = person[target].to_numpy(dtype=np.float64)[rows]
                elif kind == "person" and target in hh.columns:
                    link = person["person_household_id"].to_numpy()[rows]
                    prod = (hh.set_index("household_id")[target].reindex(link)
                            .to_numpy(dtype=np.float64))
                elif target in hh.columns:
                    prod = hh[target].to_numpy(dtype=np.float64)[rows]
                else:
                    checks[target] = "no production column"
                    continue
                eq = mine[in_prod] == prod[in_prod]
                checks[target] = {"n": int(in_prod.sum()), "share_exact": float(eq.mean()),
                                  "drawn_total": float((weight * drawn)[in_prod].sum()),
                                  "production_total": float((weight * prod)[in_prod].sum())}
                print(f"  recipients {target}: {in_prod.sum()} in production, exact "
                      f"{eq.mean():.4f}", flush=True)
            rec.update({"weight": weight, "in_production": in_prod, "weight_source": source,
                        "entity": kind, "check": checks})


STAGES = {
    "scf_net_worth": stage_scf_net_worth,
    "scf_financial_assets": stage_scf_financial_assets,
    "sipp_financial_assets": stage_sipp_financial_assets,
    "scf_auto_loans": stage_scf_auto_loans,
    "sipp_vehicles": stage_sipp_vehicles,
    "sipp_tips": stage_sipp_tips,
    "org_wages": stage_org_wages,
    "ssi_disability_criteria": _binary_stage(
        "ssi_disability_criteria", "load_sipp_2023_ssi_disability_donor",
        "impute_us_ssi_disability_criteria", 42, ("person", "meets_ssi_disability_criteria")),
    "sipp_head_start": _binary_stage(
        "sipp_head_start", "load_sipp_2023_head_start_donor", "impute_us_sipp_head_start",
        SEED, ("person", "takes_up_head_start_if_eligible")),
    "voluntary_filing": _binary_stage(
        "voluntary_filing", "load_sipp_2023_voluntary_filing_donor",
        "impute_us_voluntary_filing", SEED, ("tax_unit", "would_file_taxes_voluntarily")),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", nargs="+", default=list(STAGES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    tables = load_tables()
    frame = load_frame(tables)
    for name in args.stages:
        t0 = time.time()
        print(f"{name}:", flush=True)
        try:
            caps, checks = STAGES[name](tables, frame)
        except Exception as exc:  # noqa: BLE001 - record and move on
            import traceback

            traceback.print_exc()
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
            args.out.write_text(json.dumps(report, indent=1, default=str))
            continue
        annotate_recipients(name, caps, tables)
        saved = []
        for i, spec in enumerate(caps):
            key = name if len(caps) == 1 else f"{name}__{i}"
            capture.save(spec, key, {"stage": name, "fit_index": i})
            saved.append({"key": key, **spec.meta()})
        report[name] = {"fits": saved, "checks": checks, "seconds": time.time() - t0}
        args.out.write_text(json.dumps(report, indent=1, default=str))
        print(f"  {len(caps)} fit(s) captured in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
