"""Record every RegimeGatedQRF fit a production stage function makes.

``recording()`` wraps ``RegimeGatedQRF.fit`` in both ``populace.fit.qrf``
(US code at cae8640) and ``microcosm.fit.qrf`` (UK code at 67ee2ddf3, and the
venv's fit engine) so a production imputation function runs unchanged and each
fit's inputs are kept: the donor table, predictors, targets, the resolved
weight vector, ``n_estimators``, ``seed``, ``zero_atol`` and
``max_samples_leaf``. The real fit then proceeds, so the stage's own output can
be compared with the production dataset (the reproduction check).

It also wraps ``FittedRegimeGatedQRF.predict`` (and, where present, the UK
code's ``predict_from_uniforms`` and ``predict_positive_from_uniforms``) so
each fit's production recipients
are kept: the predictor columns the draw read, the drawn targets (the values a
chained target conditions on), and the input's row index, which the stage
scripts map to production weights.

``save``/``load`` keep a captured fit as a parquet donor table plus a JSON
spec under ``donors/`` (record-level survey data: local only, never shared),
and ``load_recipients`` reads a saved recipient table.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DONORS = ROOT / "donors"


@dataclass
class CapturedSpec:
    table: pd.DataFrame
    predictors: list[str]
    targets: list[str]
    weights: np.ndarray | None
    n_estimators: int
    seed: int
    zero_atol: float
    max_samples_leaf: object
    module: str
    # One entry per predict call on this fit: the recipients' predictor
    # columns plus the drawn targets (the chain's conditioning values), the
    # input's index, and the Frame entity when the input was a Frame.
    recipients: list = field(default_factory=list)
    # The fit input's row labels (``_resolve_qrf_fit_input`` keeps the caller's
    # index), so a stage script can map donor rows back to their households.
    source_index: np.ndarray | None = None

    def meta(self) -> dict:
        return {"predictors": self.predictors, "targets": self.targets,
                "n_estimators": self.n_estimators, "seed": self.seed,
                "zero_atol": self.zero_atol, "max_samples_leaf": self.max_samples_leaf,
                "module": self.module, "n_rows": int(len(self.table)),
                "weighted": self.weights is not None}


def _modules():
    mods = []
    try:
        import populace.fit.qrf as P

        mods.append(("populace.fit.qrf", P))
    except Exception:  # noqa: BLE001 - the US snapshot may not be on the path
        pass
    import microcosm.fit.qrf as M

    mods.append(("microcosm.fit.qrf", M))
    return mods


@contextlib.contextmanager
def recording(*, fit_for_real: bool = True):
    """Yield a list that fills with a ``CapturedSpec`` per fit call."""
    captured: list[CapturedSpec] = []
    originals = []
    for name, mod in _modules():
        cls = mod.RegimeGatedQRF
        real = cls.fit

        def wrapper(self, frame_or_df, predictors, targets, *, weights=mod.DESIGN_WEIGHTS
                    if hasattr(mod, "DESIGN_WEIGHTS") else "design", _real=real, _mod=mod,
                    _name=name):
            resolved = _mod._resolve_qrf_fit_input(frame_or_df, list(predictors),
                                                   list(targets), weights)
            table = resolved.table
            cols = list(dict.fromkeys([*predictors, *targets]))
            captured.append(CapturedSpec(
                table=table.loc[:, cols].reset_index(drop=True).copy(),
                predictors=list(predictors), targets=list(targets),
                weights=None if resolved.weights is None else np.asarray(
                    resolved.weights, dtype=np.float64).copy(),
                n_estimators=int(self.n_estimators), seed=int(self.seed),
                zero_atol=float(self.zero_atol), max_samples_leaf=self.max_samples_leaf,
                module=_name, source_index=np.asarray(table.index).copy()))
            if not fit_for_real:
                raise _Stop()
            self._capture_index = len(captured) - 1
            fitted = _real(self, frame_or_df, predictors, targets, weights=weights)
            fitted._capture_index = len(captured) - 1
            return fitted

        originals.append((cls, "fit", real))
        cls.fit = wrapper

        if hasattr(cls, "start_chain"):
            real_start, real_step = cls.start_chain, cls.fit_draw_next

            def start_wrapper(self, frame_or_df, predictors, targets, *, weights=mod.DESIGN_WEIGHTS
                              if hasattr(mod, "DESIGN_WEIGHTS") else "design", _real=real_start,
                              _mod=mod, _name=name):
                # A chain fits exactly what ``fit`` fits on the same inputs (both
                # spawn (fit, draw) from SeedSequence(seed) and fit targets in
                # order from the fit stream; tests/test_selection.py
                # ``test_chain_fit_matches_fit``), so it is captured as a fit.
                resolved = _mod._resolve_qrf_fit_input(frame_or_df, list(predictors),
                                                       list(targets), weights)
                cols = list(dict.fromkeys([*predictors, *targets]))
                captured.append(CapturedSpec(
                    table=resolved.table.loc[:, cols].reset_index(drop=True).copy(),
                    predictors=list(predictors), targets=list(targets),
                    weights=None if resolved.weights is None else np.asarray(
                        resolved.weights, dtype=np.float64).copy(),
                    n_estimators=int(self.n_estimators), seed=int(self.seed),
                    zero_atol=float(self.zero_atol), max_samples_leaf=self.max_samples_leaf,
                    module=_name + ":chain",
                    source_index=np.asarray(resolved.table.index).copy()))
                state = _real(self, frame_or_df, predictors, targets, weights=weights)
                self._capture_chain = getattr(self, "_capture_chain", {})
                self._capture_chain[_chain_key(state)] = len(captured) - 1
                return state

            def step_wrapper(self, frame_or_df, recipient_predictors, raw_prior_draws, *a,
                             _real=real_step, **k):
                result = _real(self, frame_or_df, recipient_predictors, raw_prior_draws, *a, **k)
                state = k.get("state")
                idx = (getattr(self, "_capture_chain", {}).get(_chain_key(state))
                       if state is not None else None)
                if idx is not None:
                    spec = captured[idx]
                    if not state.completed_targets:
                        rec = self._chain_recipient_table(recipient_predictors, state)
                        feats = rec.loc[:, list(state.predictors)]
                        spec.recipients.append({
                            "table": feats.reset_index(drop=True).copy(),
                            "index": np.asarray(feats.index), "entity": None,
                            "method": "chain"
                                      + ("_uniforms" if k.get("quantiles") is not None else "")})
                    spec.recipients[-1]["table"][result.target] = np.asarray(
                        result.raw_draw, dtype=np.float64)
                return result

            originals.append((cls, "start_chain", real_start))
            originals.append((cls, "fit_draw_next", real_step))
            cls.start_chain = start_wrapper
            cls.fit_draw_next = step_wrapper
        fitted_cls = mod.FittedRegimeGatedQRF
        for method in ("predict", "predict_from_uniforms", "predict_positive_from_uniforms"):
            if not hasattr(fitted_cls, method):
                continue

            def predict_wrapper(self, frame_or_df, *args, _real=getattr(fitted_cls, method),
                                _method=method, **kwargs):
                out = _real(self, frame_or_df, *args, **kwargs)
                idx = getattr(self, "_capture_index", None)
                if idx is not None:
                    _record_recipients(captured[idx], self, frame_or_df, out, _method)
                return out

            originals.append((fitted_cls, method, getattr(fitted_cls, method)))
            setattr(fitted_cls, method, predict_wrapper)
    try:
        yield captured
    finally:
        for cls, name, real in originals:
            setattr(cls, name, real)


def _record_recipients(spec: CapturedSpec, model, frame_or_df, out: pd.DataFrame,
                       method: str) -> None:
    features = model._predictor_frame(frame_or_df)
    table = features.reset_index(drop=True).copy()
    for target in out.columns:
        table[target] = np.asarray(out[target], dtype=np.float64)
    entity = None if isinstance(frame_or_df, pd.DataFrame) else getattr(model, "entity", None)
    spec.recipients.append({"table": table, "index": np.asarray(features.index),
                            "entity": entity, "method": method})


def _chain_key(state) -> tuple:
    """A chain's identity across its steps (states are replaced at every step)."""
    return (tuple(state.predictors), tuple(state.targets), int(state.seed))


class _Stop(Exception):
    """Raised after capture when the caller does not want the real fit."""


def save(spec: CapturedSpec, name: str, extra: dict | None = None) -> Path:
    DONORS.mkdir(parents=True, exist_ok=True)
    table = spec.table.copy()
    table["__weight__"] = spec.weights if spec.weights is not None else 1.0
    path = DONORS / f"{name}.parquet"
    table.to_parquet(path, index=False)
    if spec.source_index is not None:
        np.save(DONORS / f"{name}__source_index.npy", spec.source_index, allow_pickle=False)
    meta = spec.meta()
    meta.update(extra or {})
    meta["recipients"] = []
    for j, rec in enumerate(spec.recipients):
        rtab = rec["table"].copy()
        rtab["__index__"] = rec["index"]
        if "weight" in rec:
            rtab["__weight__"] = rec["weight"]
        if "in_production" in rec:
            rtab["__in_production__"] = rec["in_production"]
        rtab.to_parquet(DONORS / f"{name}__rec{j}.parquet", index=False)
        meta["recipients"].append({"file": f"{name}__rec{j}.parquet", "n": int(len(rtab)),
                                   "entity": rec.get("entity"),
                                   "method": rec.get("method"),
                                   "weight_source": rec.get("weight_source"),
                                   "check": rec.get("check")})
    (DONORS / f"{name}.json").write_text(json.dumps(meta, indent=1, default=str))
    return path


def load(name: str):
    meta = json.loads((DONORS / f"{name}.json").read_text())
    table = pd.read_parquet(DONORS / f"{name}.parquet")
    return table, meta


def load_recipients(name: str, j: int = 0) -> pd.DataFrame:
    return pd.read_parquet(DONORS / f"{name}__rec{j}.parquet")
