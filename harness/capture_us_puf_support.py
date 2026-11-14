"""Reconstruct the US PUF-support QRF fits that train on ASEC persons, and capture them.

Ten stages (retirement distributions and contributions, other health insurance
premiums, prior-year income, childcare, child support, disability benefits,
workers' compensation, energy subsidy, weeks unemployed) fit a ``RegimeGatedQRF``
on the ASEC half of the PUF-support spine and draw onto its PUF-tax-detail half.
Each runs in the base builder after support cloning
(``tools/build_us_puf_support_base.py:1859-2060`` at 6d06410bb), on the base
pool of 337,704 households (``parent_build_manifest.json``
``calibration.selection_source.n_base_candidates``), before calibration
selected the 57,240 households the release ships. That pool (base sha
``5d8a3162``, built at 6d06410bb) is not published, so production's
5,000-person training sample cannot be redrawn.

This script reruns each stage's own imputation handler
(``impute_us_*_to_puf_support_from_manifest``) at 6d06410bb on the production
H5 instead: the same predictor builder (``_person_*_predictors``), prefix
columns, operation parameters from the stage spec (5,000 training rows, 100
trees, seed 0) and fit code, with the H5's ASEC-channel persons as the
training pool and the H5's calibrated household weights as person weights.
Childcare and energy subsidy train on an SPM-unit value carried on every
member person, so the script copies the released unit value to each member,
as the derive step does from the ASEC's replicated SPM fields. The training
pool and weights differ from production's, so these are reconstructions, not
reproductions: the selection they measure is that of the same method on the
same kind of sample. Each reconstruction's draws are compared with the H5's
PUF-half values as a check on totals. Prior-year income cannot be rebuilt
(``UNRECONSTRUCTABLE``).

Usage: python capture_us_puf_support.py --out ../results/capture_us_puf_support.json
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prodpath  # noqa: E402

# The base pool these stages ran in was built at 6d06410bb (BUILDM_STATE.json
# build_commit); capture_us then reuses that path instead of cae8640.
prodpath.us("6d06410bb")

import capture_us as C  # noqa: E402  (H5 loaders; keeps the 6d06410bb path)
import capture  # noqa: E402

MODULES = (
    "retirement_distributions", "other_health_insurance", "prior_year_income",
    "retirement_contributions", "childcare", "child_support", "disability_benefits",
    "workers_compensation", "energy_subsidy", "weeks_unemployed",
)

#: Stages whose fit cannot be rebuilt from the release, with the reason.
UNRECONSTRUCTABLE = {
    "prior_year_income": (
        "The chain's first target, employment_income_last_year, is formula-owned and "
        "dropped from the stage output (prior_year_income.py:118, 690-691 at 6d06410bb), "
        "so the release lacks the training column; the two later targets condition on it."),
}

#: Stages whose output is an SPM-unit value carried on every member person in
#: training (the derive step replicates the ASEC SPM value, childcare.py:173-199,
#: energy_subsidy.py:207-215) and reduced to the unit by the first person's draw
#: (childcare.py:504-517).
SPM_OUTPUT_STAGES = ("childcare", "energy_subsidy")


def _one(M, pattern_fn, label):
    names = [n for n, obj in vars(M).items() if callable(obj) and pattern_fn(n)]
    if len(names) != 1:
        raise LookupError(f"{M.__name__}: expected one {label}, found {names}")
    return names[0]


def reconstruct(module: str, tables, frame):
    M = importlib.import_module(f"populace.build.us_runtime.{module}")
    helper = _one(M, lambda n: n.startswith("_person_") and n.endswith("predictors"),
                  "predictor builder")
    handler = _one(M, lambda n: n.startswith("impute_us_")
                   and n.endswith("_to_puf_support_from_manifest"), "PUF-support handler")
    spec_fn = _one(M, lambda n: n.startswith("us_") and n.endswith("_stage_spec"), "stage spec")
    prefix = getattr(M, "_PUF_PREDICTOR_PREFIX", None) or getattr(M, "_PREDICTOR_PREFIX")
    names = (getattr(M, "_PUF_PREDICTORS", None) or getattr(M, "_PREDICTORS", None)
             or getattr(M, "WEEKS_UNEMPLOYED_PUF_PREDICTORS"))
    spec = getattr(M, spec_fn)()
    ops = [op for op in spec.operations if op.kind.endswith("_to_puf_support")]
    if len(ops) != 1:
        raise LookupError(f"{module}: expected one PUF-support operation, found {ops}")
    op = ops[0]
    person = frame.table("person")
    stage_person = person.copy(deep=True)
    stage_person[M._PERSON_WEIGHT_COLUMN] = frame.resolve_weights("person").values
    predictors = getattr(M, helper)(frame)
    for column in names:
        if column in predictors.columns:  # weeks_unemployed's UC predictor is optional
            stage_person[prefix + column] = predictors[column].to_numpy()
    if module in SPM_OUTPUT_STAGES:
        spm = frame.table("spm_unit")
        unit_value = pd.Series(spm[M._OUTPUT].to_numpy(dtype=np.float64),
                               index=spm["spm_unit_id"].to_numpy())
        stage_person[M._OUTPUT] = unit_value.reindex(
            person["person_spm_unit_id"].to_numpy()).to_numpy()
    context = SimpleNamespace(config=SimpleNamespace(seed=C.SEED, target_year=C.PERIOD))
    with capture.recording() as caps:
        result = getattr(M, handler)(stage_person, op, context)
    return M, op, stage_person, result, caps, {"helper": helper, "handler": handler,
                                               "spec": spec_fn, "operation": op.kind,
                                               "parameters": dict(op.parameters)}


def annotate(caps, tables, stage_person, M, module):
    """Production recipients: the H5's PUF-half persons, weighted by their household.

    For an SPM-unit output only each unit's first person's draw reaches the
    release (``childcare.py:504-517``), so only those persons count as
    production recipients, compared with the released unit value.
    """
    person = tables["person"]
    channel = stage_person[M._PERSON_SUPPORT_CHANNEL_COLUMN].astype(str)
    checks = []
    for spec in caps:
        for rec in spec.recipients:
            rows = person.index.get_indexer(rec["index"])
            assert (rows >= 0).all()
            assert (channel.iloc[rows] != channel.iloc[rows].iloc[0]).sum() == 0
            weight = C.household_weight_of(person.iloc[rows], tables, "person_household_id")
            in_production = np.ones(len(rows), dtype=bool)
            if module in SPM_OUTPUT_STAGES:
                first = ~person["person_spm_unit_id"].duplicated(keep="first").to_numpy()
                in_production = first[rows]
            check = {}
            w_prod = weight[in_production]
            for target in spec.targets:
                drawn = np.maximum(rec["table"][target].to_numpy(dtype=np.float64), 0.0)
                prod = stage_person[target].to_numpy(dtype=np.float64)[rows]
                drawn, prod = drawn[in_production], prod[in_production]
                check[target] = {
                    "reconstructed_total": float((w_prod * drawn).sum()),
                    "production_total": float((w_prod * prod).sum()),
                    "positive_share_reconstructed": float(np.average(drawn > 0, weights=w_prod)),
                    "positive_share_production": float(np.average(prod > 0, weights=w_prod)),
                }
                c = check[target]
                ratio = (c["reconstructed_total"] / c["production_total"]
                         if c["production_total"] else float("nan"))
                print(f"  {target}: reconstructed/production total {ratio:.3f}; positive share "
                      f"{c['positive_share_reconstructed']:.3f} vs "
                      f"{c['positive_share_production']:.3f}", flush=True)
            rec.update({"weight": weight, "in_production": in_production,
                        "weight_source": "H5 household_weight of the person's household",
                        "entity": "person", "check": check})
            checks.append(check)
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", nargs="+", default=list(MODULES))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = json.loads(args.out.read_text()) if args.out.exists() else {}
    tables = C.load_tables()
    frame = C.load_frame(tables)
    for module in args.modules:
        t0 = time.time()
        print(f"{module}:", flush=True)
        if module in UNRECONSTRUCTABLE:
            report[module] = {"not_reconstructed": UNRECONSTRUCTABLE[module]}
            args.out.write_text(json.dumps(report, indent=1, default=str))
            print(f"  not reconstructed: {UNRECONSTRUCTABLE[module]}", flush=True)
            continue
        try:
            M, op, stage_person, result, caps, info = reconstruct(module, tables, frame)
            checks = annotate(caps, tables, stage_person, M, module)
        except Exception as exc:  # noqa: BLE001 - record and move on
            import traceback

            traceback.print_exc()
            report[module] = {"error": f"{type(exc).__name__}: {exc}"}
            args.out.write_text(json.dumps(report, indent=1, default=str))
            continue
        saved = []
        for i, spec in enumerate(caps):
            key = f"puf_support_{module}" + ("" if len(caps) == 1 else f"__{i}")
            capture.save(spec, key, {"stage": module, "fit_index": i, "reconstructed": True,
                                     **{k: v for k, v in info.items() if k != "parameters"}})
            # Training rows are ASEC-half persons (row labels of the person
            # table); folds and halves keep each household whole, which also
            # keeps together the members that carry one SPM unit's value.
            person = tables["person"]
            rows = person.index.get_indexer(spec.source_index)
            assert (rows >= 0).all(), f"{key}: training rows outside the person table"
            groups = person["person_household_id"].to_numpy()[rows].astype(str)
            np.save(capture.DONORS / f"{key}__groups.npy", groups)
            saved.append({"key": key, **spec.meta(), "households": int(len(np.unique(groups)))})
        report[module] = {"fits": saved, "checks": checks, "info": info,
                          "seconds": time.time() - t0}
        args.out.write_text(json.dumps(report, indent=1, default=str))
        print(f"  {len(caps)} fit(s) captured in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
