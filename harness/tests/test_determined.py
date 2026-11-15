"""The scan for targets whose chained features fix them (``harness/determined.py``).

Invariants checked:

- a target equal to an earlier target's column less another (non-qualified
  dividends as the dividend predictor less qualified dividends) is found on
  both signs, with the identity named;
- a target equal to a predictor on its positive donors only (a positive part)
  is found on that sign;
- a target with noise is not found;
- an exact fit with fewer than ``MIN_DOF`` donors beyond its columns is not
  called determined, and one with exactly that many is;
- a residual just above ``REL_TOL`` of the largest amount is not determined,
  and one just below is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import determined as D  # noqa: E402


def chain_table(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    dividends = np.round(rng.lognormal(7, 2, n)) * (rng.uniform(size=n) < 0.6)
    qualified = np.round(dividends * rng.uniform(0, 1.1, n))  # can exceed the total
    gains = rng.normal(0, 5000, n)
    return pd.DataFrame({
        "dividend_predictor": dividends,
        "gains_predictor": gains,
        "age": rng.integers(18, 85, n).astype(float),
        "qualified": qualified,
        "non_qualified": dividends - qualified,
        "gains_positive": np.maximum(gains, 0.0),
        "noisy": np.maximum(dividends * rng.lognormal(0, 0.3, n), 0.0),
    })


def scan(table, targets):
    rows = D.scan_table("t", table, ["dividend_predictor", "gains_predictor", "age"], targets, 1e-6)
    return {(r["target"], r["sign"]): r for r in rows}


def test_a_chained_difference_is_found_on_both_signs():
    out = scan(chain_table(), ["qualified", "non_qualified"])
    for sign in (1, -1):
        r = out[("non_qualified", sign)]
        assert r["status"] == "determined", r
        assert "dividend_predictor" in r["identity"] and "qualified" in r["identity"]
    assert out[("qualified", 1)]["status"] == "not_determined"


def test_a_positive_part_is_found_on_its_sign():
    out = scan(chain_table(), ["gains_positive"])
    assert out[("gains_positive", 1)]["status"] == "determined"
    assert out[("gains_positive", -1)]["status"] == "too_few_donors"


def test_a_target_with_noise_is_not_found():
    out = scan(chain_table(), ["noisy"])
    assert out[("noisy", 1)]["status"] == "not_determined"


def test_an_exact_fit_on_few_donors_is_not_called_determined():
    # Four positive donors against six columns: least squares fits them exactly.
    out = scan(chain_table(n=8, seed=0), ["qualified", "non_qualified", "noisy"])
    r = out[("noisy", 1)]
    assert r["n"] <= r["n_columns"] and r["max_rel_resid"] <= D.REL_TOL
    assert r["status"] == "exact_but_few_donors"


def _exact_table(n, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    return pd.DataFrame({"a": X[:, 0], "b": X[:, 1], "c": X[:, 2],
                         "y": 5.0 + X[:, 0] - 2.0 * X[:, 1]})


def test_the_donor_rule_sits_at_min_dof_beyond_the_columns():
    p = 4  # intercept and three predictors
    for n, status in ((p + D.MIN_DOF - 1, "exact_but_few_donors"), (p + D.MIN_DOF, "determined")):
        rows = D.scan_table("t", _exact_table(n), ["a", "b", "c"], ["y"], 1e-6)
        r = {x["sign"]: x for x in rows}[1]
        assert r["n"] == n and r["n_columns"] == p, r
        assert r["status"] == status, (n, r)


def test_the_tolerance_sits_at_rel_tol():
    base = _exact_table(200).assign(y=lambda t: 10.0 + t["a"] + t["b"])  # all positive
    bump = np.zeros(len(base))
    bump[0] = 1.0  # one donor off the identity
    y_max = base["y"].abs().max()
    for scale, status in ((50 * D.REL_TOL, "not_determined"), (0.02 * D.REL_TOL, "determined")):
        t = base.assign(y=base["y"] + scale * y_max * bump)
        r = {x["sign"]: x for x in D.scan_table("t", t, ["a", "b", "c"], ["y"], 1e-6)}[1]
        assert r["status"] == status, (scale, r)
