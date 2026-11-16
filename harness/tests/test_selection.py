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
import scipy.sparse as sp
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
            pos = np.searchsorted(cw, tau - S.TIE_TOLERANCE, side="left")
            out[i, k] = y_fill[order][min(pos, len(cw) - 1)]
    return out


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 10_000), mode=st.sampled_from(["pooled", "tree"]),
       top=st.sampled_from([None, 4]))
def test_sparse_quantiles_match_brute_force(seed, mode, top):
    rng = np.random.default_rng(seed)
    n = 120
    X = rng.normal(size=(n, 2))
    y = np.round(np.exp(X[:, 0] + rng.normal(size=n)), 1)  # ties on purpose
    if top is not None:  # a near-discrete target: a handful of values
        y = np.minimum(np.round(y), top)
    model = S.grow(X, y, seed, 6, 2)
    cnt = S.qd.bootstrap_counts(model, n)
    Xr = rng.normal(size=(25, 2))
    nodes = [e.tree_.node_count for e in model.estimators_]
    lf, lr = model.apply(X), model.apply(Xr)
    levels, level_of = np.unique(y, return_inverse=True)
    A = S.sparse_pool(lf, cnt, lr, nodes, mode, level_of, len(levels))
    assert A.shape[1] == len(levels)
    y_eval = rng.normal(size=25)
    w_eval = np.ones(25)
    res = S.quantile_metrics(A, levels, y_eval, w_eval)
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


def test_recipients_no_honest_donor_reaches_add_nothing():
    """A recipient without an honest estimate keeps its adaptive estimate."""
    adaptive = np.array([10.0, 20.0, 30.0, 40.0])
    honest = np.array([8.0, np.nan, 33.0, np.nan])
    scale = np.array([1.0, 2.0, 3.0, 4.0])
    added, share = S.complete_honest(adaptive, honest, scale)
    assert added == 2.0 * 20.0 + 4.0 * 40.0
    assert share == pytest.approx((40.0 + 160.0) / (10.0 + 40.0 + 90.0 + 160.0))
    ok = np.isfinite(honest)
    honest_total = float((honest[ok] * scale[ok]).sum()) + added
    adaptive_total = float((adaptive * scale).sum())
    assert adaptive_total - honest_total == pytest.approx(
        float(((adaptive - honest)[ok] * scale[ok]).sum()))


def test_complete_honest_changes_nothing_when_every_recipient_is_reached():
    adaptive, honest, scale = np.array([1.0, 2.0]), np.array([1.5, 2.5]), np.array([3.0, 4.0])
    assert S.complete_honest(adaptive, honest, scale) == (0.0, 0.0)


def test_quantile_metrics_skip_rows_without_mass():
    A = sp.csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 2.0]]))
    y_sorted = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 5.0, 1.0])
    with_gap = S.quantile_metrics(A, y_sorted, np.array([1.5, 9.0, 2.5]), w)
    without = S.quantile_metrics(A[[0, 2]], y_sorted, np.array([1.5, 2.5]), w[[0, 2]])
    assert with_gap == without


def test_empty_recipients_never_inflate_selection():
    """With a sparse honest half, the forest's own record reports what it completed."""
    rng = np.random.default_rng(3)
    n = 60
    X = rng.normal(size=(n, 3))
    y = np.exp(X[:, 0]) + 1.0
    w = np.ones(n)
    X_rec = np.vstack([X, X * 8.0])  # recipients far out land in sparse leaves
    scale = np.ones(len(X_rec))
    out = S.same_tree_sign(X, y, w, X_rec, scale, seed=5, n_estimators=3, variants=(1,))
    totals = out["variants"]["1"]["totals"]
    assert np.isfinite(totals["honest"]["pooled"])
    assert np.isfinite(totals["adaptive"]["pooled"])
    share = out["max_empty_adaptive_share"]["pooled"]
    assert 0.0 <= share <= 1.0
    if out["max_empty_recipients"]["honest"]["pooled"]:
        assert share > 0.0


def test_compare_runs_counts_every_changed_number():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import compare_runs as C

    base = {"folds": [{"a": 1.0, "seconds": 3.0, "b": {"c": [1, float("nan")]}}]}
    same = {"folds": [{"a": 1.0, "seconds": 9.0, "b": {"c": [1, float("nan")]}}]}
    out = C.compare(base, same)
    assert (out["numbers"], out["differing"], out["unmatched"]) == (3, [], [])
    changed = {"folds": [{"a": 1.5, "seconds": 3.0, "b": {"c": [1, 2.0]}}]}
    assert len(C.compare(base, changed)["differing"]) == 2
    assert len(C.compare(base, changed)["outside_quantile"]) == 2
    a = {"folds": [{"quantile": {"pinball": 1.0}, "total": 2.0}]}
    b = {"folds": [{"quantile": {"pinball": 1.5}, "total": 2.0}]}
    out = C.compare(a, b)
    assert (len(out["differing"]), out["outside_quantile"]) == (1, [])
    assert out["max_rel_quantile"] == pytest.approx(1 / 3)
    missing = {"folds": [{"a": 1.0, "seconds": 3.0, "b": {}}]}
    assert C.compare(base, missing)["unmatched"] == [[0, "b", "c"]]
    assert C.compare(base, missing, common=True)["unmatched"] == []
    assert C.compare(base, missing, common=True)["numbers"] == 1


def test_quantile_ties_resolve_low_however_many_rows_precede():
    # Two values with exactly half the mass each: the median is the lower one.
    # A running sum over every row would let rounding flip late rows.
    n = 200_000
    rng = np.random.default_rng(0)
    half = rng.integers(1, 50, size=n).astype(np.float64)
    data = np.column_stack([half, half]).ravel()
    A = sp.csr_matrix((data, np.tile([0, 1], n), np.arange(0, 2 * n + 1, 2)), shape=(n, 2))
    y_eval, w = np.zeros(n), np.ones(n)
    res = S.quantile_metrics(A, np.array([1.0, 5.0]), y_eval, w, taus=(0.5,))
    assert res["pinball"]["0.5"] == pytest.approx(0.5 * 1.0)  # q = 1 on every row
    assert res["coverage"]["0.5"] == 1.0


def test_quantile_metrics_ignore_how_rows_are_grouped():
    rng = np.random.default_rng(3)
    A = sp.random(300, 40, density=0.3, random_state=4, format="csr")
    A.data = rng.integers(1, 6, size=A.nnz).astype(np.float64)
    keep = np.diff(A.indptr) > 0
    A = A[keep]
    A.sort_indices()
    y_sorted = np.sort(rng.normal(size=40))
    y_eval, w = rng.normal(size=A.shape[0]), rng.uniform(1, 3, size=A.shape[0])
    whole = S.quantile_metrics(A, y_sorted, y_eval, w)
    m = A.shape[0] // 2
    a = S.quantile_metrics(A[:m], y_sorted, y_eval[:m], w[:m])
    b = S.quantile_metrics(A[m:], y_sorted, y_eval[m:], w[m:])
    for tau in map(str, S.TAUS):
        pooled = (a["pinball"][tau] * w[:m].sum() + b["pinball"][tau] * w[m:].sum()) / w.sum()
        assert whole["pinball"][tau] == pytest.approx(pooled, rel=1e-12)


def _no_signal_differences(monkeypatch, *, blind, seeds=40, n=1200, trees=15):
    """Relative adaptive minus honest over seeds, with y independent of X (lognormal)."""
    real_grow = S.grow

    def grow_on_noise(X, y, seed, n_estimators, min_samples_leaf=1, store_leaf_samples=False):
        z = np.random.default_rng([int(seed), 99]).normal(size=len(y))
        return real_grow(X, z, seed, n_estimators, min_samples_leaf, store_leaf_samples)

    if blind:
        monkeypatch.setattr(S, "grow", grow_on_noise)
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng([seed, 5])
        X = rng.normal(size=(n, 3))
        y = np.exp(rng.normal(size=n))
        w = rng.uniform(1, 4, size=n)
        X_rec = rng.normal(size=(600, 3))
        res = S.same_tree_sign(X, y, w, X_rec, np.ones(600), seed=seed, n_estimators=trees)
        totals = res["variants"]["1"]["totals"]
        out.append(totals["adaptive"]["pooled"] / totals["honest"]["pooled"] - 1)
    return np.array(out)


def test_no_selection_when_the_splits_never_see_the_filling_values(monkeypatch):
    # Trees grown on noise, independent of y and of the predictors: the catchments
    # do not depend on y, so adaptive and honest fills have the same expected total.
    diffs = _no_signal_differences(monkeypatch, blind=True)
    assert abs(diffs.mean()) < 3 * diffs.std(ddof=1) / np.sqrt(len(diffs))


def test_selection_appears_when_the_splits_see_the_filling_values(monkeypatch):
    # The same data with trees grown on y itself: no signal to find, yet the
    # splits have chosen their cuts on the values that fill the leaves.
    diffs = _no_signal_differences(monkeypatch, blind=False)
    assert abs(diffs.mean()) > 3 * diffs.std(ddof=1) / np.sqrt(len(diffs))


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), top=st.sampled_from([None, 4]))
def test_forest_quantiles_and_draw_means_match_quantile_forest(seed, top):
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.normal(size=(n, 2))
    y = np.round(np.exp(X[:, 0] + rng.normal(size=n)), 1)
    if top is not None:
        y = np.minimum(np.round(y), top)
    if len(np.unique(y)) < 2:
        return
    model = S.grow(X, y, seed, 12, 2, store_leaf_samples=True)
    cnt = S.qd.bootstrap_counts(model, n)
    Xr = rng.normal(size=(15, 2))
    grid = S._qrf_module()._QUANTILE_GRID
    predicted = np.asarray(model.predict(Xr, quantiles=list(grid)))
    nodes = [e.tree_.node_count for e in model.estimators_]
    levels, level_of = np.unique(y, return_inverse=True)
    A = S.sparse_pool(model.apply(X), cnt, model.apply(Xr), nodes, "pooled", level_of, len(levels))
    A = A.tocsr()
    for i in range(Xr.shape[0]):
        lo, hi = A.indptr[i], A.indptr[i + 1]
        mine = S.forest_quantiles(levels[A.indices[lo:hi]], A.data[lo:hi], grid)
        assert mine == pytest.approx(predicted[i], rel=1e-9, abs=1e-9)
    expected = S.expected_draws(A, levels, grid)
    by_hand = [grid[0] * c[0] + (1 - grid[-1]) * c[-1]
               + np.trapezoid(c, grid) for c in predicted]
    assert expected == pytest.approx(np.array(by_hand), rel=1e-9, abs=1e-9)


def test_quantile_formula_depends_on_the_scale_of_fractional_masses():
    # The formula reads masses as frequencies, so scaling them moves the curve.
    values = np.array([1.0, 2.0, 3.0, 100.0])
    grid = np.linspace(0, 1, 201)
    assert np.all(S.forest_quantiles(values, np.full(4, 0.2), grid) == 100.0)  # total mass below one
    area = lambda m: np.trapezoid(S.forest_quantiles(values, m, grid), grid)
    assert area(np.full(4, 1000.0)) == pytest.approx(values.mean(), rel=1e-2)
    assert abs(area(np.full(4, 0.4)) - values.mean()) > 0.2 * values.mean()


def test_honest_resample_gives_integer_masses_of_the_bootstrap_shape():
    weights = np.array([1.0, 0.0, 3.0, 4.0, 2.0])
    picks, counts = S.honest_resample(weights, 7, np.random.default_rng(0))
    assert picks.shape == (5,) and counts.shape == (7, 5)
    assert 1 not in picks  # a zero-weight row never enters the resample
    assert np.issubdtype(counts.dtype, np.integer)
    assert (counts.sum(axis=1) == 5).all()


def test_draw_totals_carry_every_honest_reading():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(400, 2))
    y = np.exp(rng.normal(size=400))
    w = rng.uniform(1, 3, size=400)
    X_rec = rng.normal(size=(50, 2))
    res = S.same_tree_sign(X, y, w, X_rec, np.ones(50), seed=1, n_estimators=8,
                           draw_grid=np.linspace(1e-6, 1 - 1e-6, 51))
    assert set(res["draw_totals"]) == {"adaptive", "honest", "honest_mean", "honest_weight"}
    assert all(np.isfinite(v) for v in res["draw_totals"].values())


def test_forest_quantiles_match_the_real_formula_on_any_masses():
    # Differential: quantile-forest's own function, on integer and fractional
    # masses, including pools with less than one unit of mass.
    from quantile_forest._quantile_forest_fast import calc_weighted_quantile

    rng = np.random.default_rng(0)
    grid = np.linspace(0, 1, 41)
    for _ in range(3000):
        n = int(rng.integers(1, 8))
        values = np.sort(rng.lognormal(size=n))
        weights = [rng.integers(1, 6, size=n).astype(float), rng.uniform(0.01, 3, size=n),
                   rng.uniform(0.01, 0.2, size=n), rng.uniform(5, 500, size=n)][int(rng.integers(4))]
        real = np.array(calc_weighted_quantile(list(values), list(weights), list(grid),
                                               b"linear", True))
        assert S.forest_quantiles(values, weights, grid) == pytest.approx(real, rel=1e-9, abs=1e-9)
