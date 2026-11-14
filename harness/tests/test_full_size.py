"""The production-size forests and their summary.

Invariants checked:

- the leaf-1 full-size forest equals the fold's own forest, on totals over the
  held-out recipients and over the production recipients;
- every full-size forest honours its leaf floor;
- a full-size forest's quantile metrics score the same rows as the half-size
  fills;
- the summary sums signs, scales by the right totals and refuses a leaf-1
  forest that differs from the fold's own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import full_size as F  # noqa: E402
import selection as S  # noqa: E402
from test_selection import synthetic_table  # noqa: E402


def run_fold(leaves=(1, 5), quantiles=True, production=True):
    table = synthetic_table(1500, seed=9)
    folds = S.assign_folds(len(table), None, 2, 3)
    train, rec = folds != 0, folds == 0
    prod = (table.loc[rec].reset_index(drop=True), table["w"].to_numpy()[rec]) if production else None
    return S.analyse_fold(table, ["x1", "x2", "x3"], ["signed", "zip", "pos"], "w",
                          train=train, rec=rec, seed=4, n_estimators=20, log=lambda s: None,
                          quantiles=quantiles, production=prod, full_variants=leaves)


def test_leaf_one_full_size_forest_is_the_folds_own_forest():
    out = run_fold()
    for t in out["targets"].values():
        for s in t["signs"].values():
            for m in S.MODES:
                assert s["full_size"]["1"]["totals"][m] == pytest.approx(s["own_forest"][m], rel=1e-12)
                assert s["full_size"]["1"]["extra"]["production"][m] == pytest.approx(
                    s["extra"]["production"]["own_forest"][m], rel=1e-12)


def test_a_leaf_floor_moves_the_full_size_forest():
    out = run_fold()
    moved = [abs(s["full_size"]["5"]["totals"]["pooled"] - s["full_size"]["1"]["totals"]["pooled"])
             for t in out["targets"].values() for s in t["signs"].values()]
    assert max(moved) > 0


def test_full_size_quantiles_are_scored_and_bounded():
    out = run_fold()
    pos = out["targets"]["pos"]["signs"]["1"]
    for leaf in ("1", "5"):
        for m in S.MODES:
            q = pos["full_size"][leaf]["quantile"][m]
            assert q["mean_pinball"] > 0
            assert all(0 <= c <= 1 for c in q["coverage"].values())


@pytest.mark.parametrize("leaf", [5, 20])
def test_full_size_forest_honours_its_leaf_floor(leaf):
    table = synthetic_table(1500, seed=2)
    X = table[["x1", "x2", "x3"]].to_numpy()
    y = table["pos"].to_numpy()
    model = S.grow(X, y, 3, 10, leaf)
    for tree in model.estimators_:
        counts = tree.tree_.n_node_samples[tree.tree_.children_left == -1]
        assert counts.min() >= leaf


def test_keeps_restrict_full_size_metrics_to_the_rows_given():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 2))
    y = np.exp(X[:, 0] + rng.normal(size=600))
    selected = rng.integers(0, 600, 600)
    record = S.ForestRecord(seed=3, selected=selected, n_estimators=10, max_samples_leaf=None,
                            n_unique=600)
    X_eval, y_eval, w_eval = X[:100], y[:100], np.ones(100)
    everything = {m: np.ones(100, dtype=bool) for m in S.MODES}
    first_half = {m: np.arange(100) < 50 for m in S.MODES}
    a = S.full_size_forests(X, y, record, X[:50], np.ones(50), (1,),
                            eval_set=(X_eval, y_eval, w_eval), keeps=[everything])
    b = S.full_size_forests(X, y, record, X[:50], np.ones(50), (1,),
                            eval_set=(X_eval, y_eval, w_eval), keeps=[first_half])
    c = S.full_size_forests(X, y, record, X[:50], np.ones(50), (1,),
                            eval_set=(X_eval, y_eval, w_eval), keeps=[everything, first_half])
    pinballs = [d["1"]["quantile"]["pooled"]["mean_pinball"] for d in (a, b, c)]
    assert pinballs[0] != pinballs[1]
    assert pinballs[2] == pytest.approx(0.5 * (pinballs[0] + pinballs[1]))


def sign(own, full1, leaf5, leaf20, tree, adaptive, honest, *, pinball=None, prod=None):
    def q(p):
        return {"mean_pinball": p, "coverage": {"0.1": 0.2, "0.9": 0.8}}

    out = {"own_forest": {"pooled": own},
           "full_size": {"1": {"totals": {"pooled": full1, "tree": tree}},
                         "5": {"totals": {"pooled": leaf5, "tree": leaf5}},
                         "20": {"totals": {"pooled": leaf20, "tree": leaf20}}},
           "variants": {"1": {"totals": {"adaptive": {"pooled": adaptive},
                                         "honest": {"pooled": honest}}}}}
    if pinball is not None:
        out["full_size"]["1"]["quantile"] = {"pooled": q(pinball), "tree": q(pinball * 1.5)}
        out["full_size"]["5"]["quantile"] = {"pooled": q(pinball * 0.9)}
        out["full_size"]["20"]["quantile"] = {"pooled": q(pinball * 2.0)}
        out["variants"]["1"]["quantile"] = {"adaptive": {"pooled": q(pinball * 1.2)},
                                            "honest": {"pooled": q(pinball * 0.8)}}
    if prod is not None:
        for k in ("1", "5", "20"):
            out["full_size"][k]["extra"] = {"production": {"pooled": prod[k], "tree": prod["tree"]}}
        out["extra"] = {"production": {"variants": {"1": {"totals": {
            "adaptive": {"pooled": prod["adaptive"]}, "honest": {"pooled": prod["honest"]}}}}}}
    return out


def test_summary_sums_signs_and_scales_by_each_denominator(tmp_path, monkeypatch):
    prod_pos = {"1": 100.0, "5": 90.0, "20": 80.0, "tree": 95.0, "adaptive": 110.0, "honest": 70.0}
    prod_neg = {"1": -20.0, "5": -18.0, "20": -16.0, "tree": -19.0, "adaptive": -22.0, "honest": -14.0}
    fold = {"t": {"reported": 200.0,
                  "signs": {"1": sign(110.0, 110.0, 100.0, 90.0, 105.0, 115.0, 95.0,
                                      pinball=10.0, prod=prod_pos),
                            "-1": sign(-10.0, -10.0, -8.0, -7.0, -9.0, -12.0, -9.0, prod=prod_neg)}}}
    results = tmp_path / "results"
    results.mkdir()
    doc = {"targets": ["t"], "folds": [{"fold": 0, "targets": fold}, {"fold": 1, "targets": fold}]}
    (results / "selection_fit_fullsize.json").write_text(json.dumps(doc))
    monkeypatch.setattr(F, "ROOT", tmp_path)
    df = F.table().set_index("config")
    assert set(df.index) == set(F.CONFIGS)
    base_total = 110.0 - 10.0
    assert df.loc["production", "effect"] == pytest.approx(0.0)
    assert df.loc["production", "vs_reported"] == pytest.approx(100 * (base_total / 200 - 1))
    # Honest: 95 - 9 = 86 against 100.
    assert df.loc["honest fills", "effect"] == pytest.approx(100 * (86 - base_total) / 200)
    # Leaf 20: 90 - 7 = 83.
    assert df.loc["min_samples_leaf=20", "effect"] == pytest.approx(100 * (83 - base_total) / 200)
    # Tree mode comes from the leaf-1 forest.
    assert df.loc["weighted_leaves=True", "effect"] == pytest.approx(100 * (105 - 9 - base_total) / 200)
    # Production recipients: ratio of model totals. Base 100 - 20 = 80; honest 70 - 14 = 56.
    assert df.loc["honest fills", "prod_effect"] == pytest.approx(100 * (56 / 80 - 1))
    # Pinball reads the positive forest only.
    assert df.loc["honest fills", "pinball_change"] == pytest.approx(-20.0)
    assert df.loc["min_samples_leaf=20", "pinball_change"] == pytest.approx(100.0)
    assert df.loc["weighted_leaves=True", "pinball_change"] == pytest.approx(50.0)
    assert df.loc["production", "coverage_error"] == pytest.approx(0.1)


def test_summary_refuses_a_leaf_one_forest_that_differs_from_the_folds_own():
    target = {"reported": 100.0,
              "signs": {"1": sign(110.0, 111.0, 100.0, 90.0, 105.0, 115.0, 95.0)}}
    with pytest.raises(ValueError, match="regrown leaf-1"):
        F.fold_target(target, "fit/t/fold 0")


def test_summary_skips_a_sign_the_harness_skipped():
    target = {"reported": 100.0, "signs": {"1": {"skipped": "too few donors"}}}
    assert F.fold_target(target) is None
