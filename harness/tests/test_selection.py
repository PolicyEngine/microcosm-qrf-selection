"""Tests for the generic same-tree harness (``harness/selection.py``).

Invariants checked:

- capture fidelity: ``capture_fit`` returns the gates, regimes, chained
  columns, forest seeds and bootstrap rows a real ``RegimeGatedQRF.fit``
  uses, so forests rebuilt from them predict exactly what the real ones do;
- construction: ``grow(..., min_samples_leaf=1)`` builds the trees the earlier
  harness's ``microcosm_bias.forest`` builds;
- accounting: for any filled forest the catchment total equals the direct
  total (sum_j c_j y_j = sum_i scale_i est_i) and catchments conserve the
  recipients' mass (sum_j c_j = sum_i scale_i), in both modes;
- convexity: each recipient's estimate lies between the smallest and largest
  filling value;
- quantiles: the sparse pooled quantiles equal a brute-force dense
  computation and never decrease in tau;
- determinism: a stage run with the same seed reproduces itself exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import selection as S  # noqa: E402

PRIOR = Path(__file__).resolve().parents[1] / "vendor"  # qrf-moop-shortfall@dba3940
sys.path.insert(0, str(PRIOR))


def synthetic_table(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.integers(0, 4, size=n).astype(float)
    x3 = rng.lognormal(size=n)
    signed = np.where(rng.random(n) < 0.3, 0.0, np.exp(1 + x1 + rng.normal(size=n)))
    signed = np.where(rng.random(n) < 0.15, -np.exp(rng.normal(size=n)), signed)
    zip_pos = np.where(rng.random(n) < 0.5, 0.0, x3 * np.exp(rng.normal(size=n)) + signed.clip(0))
    pos_only = np.exp(0.5 * x1 + 0.1 * x2 + rng.normal(size=n))
    binary = (rng.random(n) < 0.4).astype(float)
    w = rng.uniform(0.5, 3.0, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "signed": signed, "zip": zip_pos,
                         "pos": pos_only, "flag": binary, "w": w})


def test_capture_matches_real_fit():
    import microcosm.fit.qrf as Q

    table = synthetic_table()
    predictors, targets = ["x1", "x2", "x3"], ["signed", "zip", "pos", "flag"]
    real = Q.RegimeGatedQRF(n_estimators=12, seed=7).fit(table, predictors, targets, weights="w")
    cap = S.capture_fit(table, predictors, targets, "w", seed=7, n_estimators=12)
    probe = synthetic_table(300, seed=2)
    for target in targets:
        tm, ct = real._target_models[target], cap.targets[target]
        assert tm.regime == ct.regime
        assert tuple(tm.columns) == ct.columns
        Xp = probe[list(ct.columns)].to_numpy(dtype=np.float64)
        if tm.gate is not None:
            np.testing.assert_array_equal(tm.gate.predict_proba(Xp), ct.gate.predict_proba(Xp))
        y = table[target].to_numpy(dtype=np.float64)
        X = table[list(ct.columns)].to_numpy(dtype=np.float64)
        w = table["w"].to_numpy(dtype=np.float64)
        for sign, forest, record in ((1, tm.positive, ct.positive), (-1, tm.negative, ct.negative)):
            if forest is None:
                assert record is None
                continue
            if ct.regime in (Q.Regime.POSITIVE_ONLY, Q.Regime.NEGATIVE_ONLY):
                mask = np.ones(len(y), dtype=bool)
            else:
                mask = y > 1e-6 if sign == 1 else y < -1e-6
            rebuilt = S.grow(X[mask][record.selected], y[mask][record.selected],
                             record.seed, record.n_estimators, store_leaf_samples=True)
            if forest.model.max_samples_leaf is None:
                np.testing.assert_array_equal(
                    forest.model.predict(Xp, quantiles="mean"),
                    rebuilt.predict(Xp, quantiles="mean"))
            # Same trees either way (the leaf-sample cap stores fewer samples).
            for a, b in zip(forest.model.estimators_, rebuilt.estimators_):
                np.testing.assert_array_equal(a.tree_.feature, b.tree_.feature)
                np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)


def earlier_forest(X, y, seed):
    """``forest`` from qrf-moop-shortfall@dba3940 ``harness/microcosm_bias.py:85-88``."""
    from quantile_forest import RandomForestQuantileRegressor

    model = RandomForestQuantileRegressor(n_estimators=100, max_samples_leaf=None,
                                          random_state=seed, n_jobs=4)
    return model.fit(X, y)


def test_grow_matches_earlier_forest():
    table = synthetic_table(800)
    X = table[["x1", "x2", "x3"]].to_numpy()
    y = table["pos"].to_numpy()
    a = earlier_forest(X, y, 11)
    for b in (S.grow(X, y, 11, 100), S.grow(X, y, 11, 100, store_leaf_samples=True)):
        np.testing.assert_array_equal(a.apply(X[:50]), b.apply(X[:50]))
        for ta, tb in zip(a.estimators_, b.estimators_):
            np.testing.assert_array_equal(ta.tree_.threshold, tb.tree_.threshold)
    np.testing.assert_array_equal(
        a.predict(X[:50], quantiles="mean"),
        S.grow(X, y, 11, 100, store_leaf_samples=True).predict(X[:50], quantiles="mean"))


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 10_000), levels=st.integers(2, 6))
def test_stored_leaf_samples_leave_trees_unchanged(seed, levels):
    """Storing one sample per leaf changes no tree: same splits, same leaves.

    The harness reads leaves through ``apply`` and bootstrap counts, so the
    forests it fills are the same whatever quantile-forest stores. A target
    with a few distinct values grows large pure leaves, the case that made
    ``max_samples_leaf=None`` run out of memory.
    """
    rng = np.random.default_rng(seed)
    n = 3000
    X = rng.normal(size=(n, 4))
    y = np.clip(np.rint(X[:, 0] + rng.normal(size=n)), 0, levels - 1)
    stored, slim = (S.grow(X, y, seed, 6, store_leaf_samples=True), S.grow(X, y, seed, 6))
    np.testing.assert_array_equal(stored.apply(X), slim.apply(X))
    for a, b in zip(stored.estimators_, slim.estimators_):
        np.testing.assert_array_equal(a.tree_.feature, b.tree_.feature)
        np.testing.assert_array_equal(a.tree_.threshold, b.tree_.threshold)
        np.testing.assert_array_equal(a.tree_.value, b.tree_.value)


@settings(max_examples=25, deadline=None)
@given(n=st.integers(30, 150), seed=st.integers(0, 10_000), msl=st.sampled_from([1, 3]))
def test_fill_identity_mass_and_convexity(n, seed, msl):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = np.exp(X[:, 0] + rng.normal(size=n))
    w = rng.uniform(0.2, 4.0, size=n)
    Xr = rng.normal(size=(40, 3))
    scale = rng.uniform(0.1, 2.0, size=40)
    half = rng.random(n) < 0.5
    if half.sum() < 3 or (~half).sum() < 3:
        return
    idx_g, idx_f = np.flatnonzero(half), np.flatnonzero(~half)
    model = S.grow(X[idx_g], y[idx_g], seed, 8, msl)
    cnt = S.qd.bootstrap_counts(model, len(idx_g))
    for X_fill, mass, y_fill, rows in (
        (X[idx_g], cnt, y[idx_g], idx_g),
        (X[idx_f], np.broadcast_to(w[idx_f] / w[idx_f].mean(), (8, len(idx_f))), y[idx_f], idx_f),
    ):
        res = S.fill_forest(model, X_fill, mass, y_fill, Xr, scale, rows, n)
        for mode in S.MODES:
            if res[mode]["n_empty"]:
                # An honest fill can leave a recipient no donor in any tree; its
                # total skips that recipient and the identity is not checked.
                assert mode == "tree" or X_fill is not X[idx_g]
                continue
            assert res[mode]["identity_rel_err"] < 1e-10
            assert abs(res[mode]["catchment_sum_over_scale"] - 1) < 1e-10
        nodes = [e.tree_.node_count for e in model.estimators_]
        lf, lr = model.apply(X_fill), model.apply(Xr)
        for mode in S.MODES:
            est = S.recipient_estimates(lf, mass, y_fill, lr, nodes, mode)
            finite = np.isfinite(est)
            assert (est[finite] >= y_fill.min() - 1e-9).all()
            assert (est[finite] <= y_fill.max() + 1e-9).all()


def brute_quantiles(lf, mass, y_fill, lr, nodes, mode, taus):
    n_rec, T = lr.shape
    out = np.empty((n_rec, len(taus)))
    for i in range(n_rec):
        wts = np.zeros(len(y_fill))
        for t in range(T):
            inleaf = (lf[:, t] == lr[i, t]) & (mass[t] > 0)
            if not inleaf.any():
                continue
            contrib = np.where(inleaf, mass[t], 0.0)
            if mode == "tree":
                contrib = contrib / contrib.sum()
            wts += contrib
        order = np.argsort(y_fill, kind="stable")
        cw = np.cumsum(wts[order]) / wts.sum()
        for k, tau in enumerate(taus):
            pos = np.searchsorted(cw, tau - 1e-12, side="left")
            out[i, k] = y_fill[order][min(pos, len(cw) - 1)]
    return out


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 10_000), mode=st.sampled_from(["pooled", "tree"]))
def test_sparse_quantiles_match_brute_force(seed, mode):
    rng = np.random.default_rng(seed)
    n = 120
    X = rng.normal(size=(n, 2))
    y = np.round(np.exp(X[:, 0] + rng.normal(size=n)), 1)  # ties on purpose
    model = S.grow(X, y, seed, 6, 2)
    cnt = S.qd.bootstrap_counts(model, n)
    Xr = rng.normal(size=(25, 2))
    nodes = [e.tree_.node_count for e in model.estimators_]
    lf, lr = model.apply(X), model.apply(Xr)
    order = np.argsort(y, kind="stable")
    A = S.sparse_pool(lf, cnt, lr, nodes, mode, order)
    y_eval = rng.normal(size=25)
    w_eval = np.ones(25)
    res = S.quantile_metrics(A, y[order], y_eval, w_eval)
    brute = brute_quantiles(lf, cnt, y, lr, nodes, mode, S.TAUS)
    for k, tau in enumerate(S.TAUS):
        q = brute[:, k]
        loss = (y_eval - q) * (tau - (y_eval < q))
        assert res["pinball"][str(tau)] == pytest.approx(loss.mean(), rel=1e-12, abs=1e-12)
    assert (np.diff(brute, axis=1) >= 0).all()


def test_stage_run_is_deterministic(tmp_path):
    table = synthetic_table(900, seed=3)
    kwargs = dict(seed=5, n_estimators=10, folds=2, variants=(1, 5))
    a = S.run_stage("t", table, ["x1", "x2", "x3"], ["signed", "zip"], "w",
                    out_path=tmp_path / "a.json", **kwargs)
    b = S.run_stage("t", table, ["x1", "x2", "x3"], ["signed", "zip"], "w",
                    out_path=tmp_path / "b.json", **kwargs)
    for fa, fb in zip(a["folds"], b["folds"]):
        for target in ("signed", "zip"):
            for sign, sa in fa["targets"][target]["signs"].items():
                sb = fb["targets"][target]["signs"][sign]
                assert sa["variants"] == sb["variants"]


def test_binary_target_has_no_selection(tmp_path):
    table = synthetic_table(900, seed=4)
    doc = S.run_stage("t", table, ["x1", "x2", "x3"], ["flag"], "w", seed=1,
                      n_estimators=10, folds=2, out_path=tmp_path / "f.json")
    for f in doc["folds"]:
        tot = f["targets"]["flag"]["signs"]["1"]["variants"]["1"]["totals"]
        assert tot["adaptive"]["pooled"] == tot["honest"]["pooled"]


def test_production_set_equal_to_heldout_matches():
    """Differential: the production-recipient path equals the held-out path on the same rows."""
    table = synthetic_table(2000, seed=6)  # enough trees and rows that tree-mode honest fills cover every recipient
    folds = S.assign_folds(len(table), None, 2, 3)
    train, rec = folds != 0, folds == 0
    prod_table = table.loc[rec].reset_index(drop=True)
    w = table["w"].to_numpy()
    out = S.analyse_fold(table, ["x1", "x2", "x3"], ["signed", "zip", "pos"], "w",
                         train=train, rec=rec, seed=4, n_estimators=40,
                         log=lambda s: None, production=(prod_table, w[rec]))
    for target, t in out["targets"].items():
        assert t["production"]["drawn_total"] == pytest.approx(t["reported"], rel=1e-12)
        for sign, s in t["signs"].items():
            ex = s["extra"]["production"]
            assert ex["max_empty"] == 0
            for fill in S.FILLS:
                for mode in S.MODES:
                    assert ex["variants"]["1"]["totals"][fill][mode] == pytest.approx(
                        s["variants"]["1"]["totals"][fill][mode], rel=1e-12)
            for mode in S.MODES:
                assert ex["own_forest"][mode] == pytest.approx(s["own_forest"][mode], rel=1e-12)


def test_chain_fit_matches_fit():
    """``start_chain`` + ``fit_draw_next`` (WAS, PUF chain) build the models ``fit`` builds.

    Both spawn (fit, draw) from ``SeedSequence(seed)`` and fit targets in order
    from the one fit stream, so ``capture_fit`` (which runs ``fit``) captures a
    chain's gates and forests too.
    """
    import microcosm.fit.qrf as Q

    table = synthetic_table(1200, seed=8)
    predictors, targets = ["x1", "x2", "x3"], ["signed", "zip", "pos", "flag"]
    real = Q.RegimeGatedQRF(n_estimators=6, seed=11).fit(table, predictors, targets, weights="w")
    model = Q.RegimeGatedQRF(n_estimators=6, seed=11)
    state = model.start_chain(table, predictors, targets, weights="w")
    recipients = table.loc[:200, predictors]
    prior = pd.DataFrame(index=recipients.index)
    probe = synthetic_table(300, seed=9)
    for target in targets:
        step = model.fit_draw_next(table, recipients, prior, state=state, weights="w")
        state = step.state
        prior[target] = step.raw_draw
        a, b = real._target_models[target], step.fitted._target_models[target]
        assert a.regime == b.regime and tuple(a.columns) == tuple(b.columns)
        Xp = probe[list(a.columns)].to_numpy(dtype=np.float64)
        if a.gate is not None:
            np.testing.assert_array_equal(a.gate.predict_proba(Xp), b.gate.predict_proba(Xp))
        for fa, fb in ((a.positive, b.positive), (a.negative, b.negative)):
            assert (fa is None) == (fb is None)
            if fa is None:
                continue
            for ta, tb in zip(fa.model.estimators_, fb.model.estimators_):
                np.testing.assert_array_equal(ta.tree_.threshold, tb.tree_.threshold)
                np.testing.assert_array_equal(ta.tree_.value, tb.tree_.value)
