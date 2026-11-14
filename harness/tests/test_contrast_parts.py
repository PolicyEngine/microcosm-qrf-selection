"""The split of the adaptive-minus-honest contrast into selection and smoothing.

Invariants checked:

- the split's contrast equals adaptive minus honest from the totals, on held-out
  and production recipients, in both modes (two implementations of one sum);
- selection is exactly zero when every filling row's value equals its
  conditional mean (no noise for the splits to select on);
- smoothing is exactly zero when the conditional mean is constant, since both
  fills' catchments sum to the same recipients' total;
- smoothing is linear in the conditional mean;
- a sign with too few held-out donors gets no estimate;
- the fitted mean stays between zero and the largest amount it fits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import selection as S  # noqa: E402
from test_selection import synthetic_table  # noqa: E402


def smoothing_fold(production=True, seed=4):
    table = synthetic_table(1500, seed=9)
    folds = S.assign_folds(len(table), None, 2, 3)
    train, rec = folds != 0, folds == 0
    prod = (table.loc[rec].reset_index(drop=True), table["w"].to_numpy()[rec]) if production else None
    return S.analyse_fold(table, ["x1", "x2", "x3"], ["signed", "zip", "pos"], "w",
                          train=train, rec=rec, seed=seed, n_estimators=12, log=lambda s: None,
                          production=prod, smoothing=True)


def test_split_contrast_equals_adaptive_minus_honest():
    out = smoothing_fold()
    checked = 0
    for t in out["targets"].values():
        for s in t["signs"].values():
            parts = s["smoothing"]
            if parts is None:
                continue
            tot = s["variants"]["1"]["totals"]
            xtot = s["extra"]["production"]["variants"]["1"]["totals"]
            for m in S.MODES:
                direct = tot["adaptive"][m] - tot["honest"][m]
                assert parts["held_out"][m]["contrast"] == pytest.approx(direct, rel=1e-9, abs=1e-9)
                xdirect = xtot["adaptive"][m] - xtot["honest"][m]
                assert parts["production"][m]["contrast"] == pytest.approx(xdirect, rel=1e-9, abs=1e-9)
                assert parts["held_out"][m]["selection"] + parts["held_out"][m]["smoothing"] == \
                    pytest.approx(parts["held_out"][m]["contrast"], rel=1e-12, abs=1e-9)
                checked += 1
    assert checked >= 4


def test_smoothing_run_skips_the_own_forest_and_keeps_the_totals():
    a = smoothing_fold(production=False)
    table = synthetic_table(1500, seed=9)
    folds = S.assign_folds(len(table), None, 2, 3)
    b = S.analyse_fold(table, ["x1", "x2", "x3"], ["signed", "zip", "pos"], "w",
                       train=folds != 0, rec=folds == 0, seed=4, n_estimators=12,
                       log=lambda s: None)
    for target, t in a["targets"].items():
        for sign, s in t["signs"].items():
            assert "own_forest" not in s
            for f in S.FILLS:
                for m in S.MODES:
                    assert s["variants"]["1"]["totals"][f][m] == \
                        b["targets"][target]["signs"][sign]["variants"]["1"]["totals"][f][m]


def one_forest(seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = np.exp(X[:, 0] + rng.normal(size=n))
    a, b = np.arange(n) < n // 2, np.arange(n) >= n // 2
    pick = np.flatnonzero(a)[rng.integers(0, n // 2, n // 2)]
    model = S.grow(X[pick], y[pick], seed, 8)
    cnt = S.qd.bootstrap_counts(model, len(pick))
    idx_b = np.flatnonzero(b)
    mass_b = np.broadcast_to(np.ones(len(idx_b)), (8, len(idx_b)))
    X_rec = rng.normal(size=(150, 3))
    return X, y, pick, cnt, idx_b, mass_b, model, X_rec, rng.uniform(1, 3, 150)


def parts_for(m, seed=0):
    X, y, pick, cnt, idx_b, mass_b, model, X_rec, scale = one_forest(seed)
    return S.contrast_parts(model, X[pick], cnt, y[pick], m[pick], X[idx_b], mass_b, y[idx_b],
                            m[idx_b], X_rec, scale)


def test_selection_is_zero_when_values_equal_their_means():
    X, y, *_ = one_forest(1)
    parts = parts_for(y, seed=1)
    for m in S.MODES:
        assert parts[m]["selection"] == pytest.approx(0.0, abs=1e-9 * abs(parts[m]["contrast"]) + 1e-9)


@settings(max_examples=25, deadline=None)
@given(c=st.floats(-1e3, 1e3, allow_nan=False), seed=st.integers(0, 5))
def test_smoothing_vanishes_for_a_constant_mean(c, seed):
    parts = parts_for(np.full(400, c), seed=seed)
    for m in S.MODES:
        assert parts[m]["smoothing"] == pytest.approx(0.0, abs=1e-8 * max(1.0, abs(c)) * 150 * 3)


@settings(max_examples=25, deadline=None)
@given(a=st.floats(-5, 5, allow_nan=False), b=st.floats(-5, 5, allow_nan=False),
       seed=st.integers(0, 5))
def test_smoothing_is_linear_in_the_mean(a, b, seed):
    X, *_ = one_forest(seed)
    m1, m2 = np.exp(X[:, 0]), X[:, 1] ** 2
    p1, p2 = parts_for(m1, seed), parts_for(m2, seed)
    p = parts_for(a * m1 + b * m2, seed)
    for m in S.MODES:
        want = a * p1[m]["smoothing"] + b * p2[m]["smoothing"]
        assert p[m]["smoothing"] == pytest.approx(want, rel=1e-9, abs=1e-7)


def test_too_few_held_out_donors_give_no_estimate():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 2))
    y = np.exp(X[:, 0] + rng.normal(size=300))
    few = (X[:10], y[:10], np.ones(10))
    out = S.same_tree_sign(X, y, np.ones(300), X[:50], np.ones(50), seed=1, n_estimators=5,
                           mean_fit=few)
    assert out["smoothing"] is None


def test_the_mean_model_stays_within_the_amounts_it_fits():
    """A few amounts far above the rest cannot push the fitted mean past them.

    The cases behind this test are the PUF chain's and SCF's top amounts, on
    which a Poisson-loss fit overflowed: here 40 of 8,000 rows sit at 1e8 to
    1e9 among lognormal incomes, and one predictor equals the target.
    """
    for seed in range(3):
        rng = np.random.default_rng(seed)
        n = 8000
        inc = np.round(rng.lognormal(10.5, 1.6, n))
        inc[:40] = rng.uniform(1e8, 1e9, 40)
        X = np.column_stack([inc, rng.integers(18, 85, n), rng.integers(0, 4, n)])
        w = rng.uniform(1, 3000, n)
        X_new = np.column_stack([inc * rng.uniform(0.5, 2.0, n), rng.integers(18, 85, n),
                                 rng.integers(0, 4, n)])
        for sign in (1.0, -1.0):
            model = S.mean_model(X, sign * inc, w, seed)
            for Z in (X, X_new):
                m = model.predict(Z)
                assert np.all(np.isfinite(m))
                assert m.min() >= 0.0 and m.max() <= inc.max()
            fitted, actual = w @ model.predict(X), w @ inc
            assert 0.5 < fitted / actual < 2.0
