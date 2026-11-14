"""draw_means.py sums each sign and reports percentages of the reported total."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import draw_means as D  # noqa: E402


def sign(adaptive_mean, honest_mean, adaptive_draw, honest_draw, own_mean, own_draw,
         boot_mean=None, weight_draw=None):
    return {"variants": {"1": {"totals": {"adaptive": {"pooled": adaptive_mean},
                                          "honest": {"pooled": honest_mean}}}},
            "draw_totals": {"adaptive": adaptive_draw, "honest": honest_draw,
                            "honest_mean": honest_mean if boot_mean is None else boot_mean,
                            "honest_weight": honest_draw if weight_draw is None else weight_draw},
            "own_forest": {"pooled": own_mean}, "own_forest_draw_total": own_draw}


def doc(targets):
    return {"targets": list(targets[0]),
            "folds": [{"targets": fold} for fold in targets]}


def test_table_sums_signs_and_scales_by_the_reported_total(tmp_path, monkeypatch):
    fold = {"t": {"reported": 200.0,
                  "signs": {"pos": sign(110, 100, 108, 99, 105, 103, boot_mean=102, weight_draw=120),
                            "neg": sign(-10, -12, -9, -11, -11, -10, boot_mean=-13, weight_draw=-12)}}}
    results = tmp_path / "results"
    results.mkdir()
    (results / "selection_fit_draws.json").write_text(json.dumps(doc([fold, fold])))
    monkeypatch.setattr(D, "ROOT", tmp_path)
    row = D.table().iloc[0]
    assert row["key"] == "fit" and row["folds"] == 2
    # (110-10) - (100-12) = 12 of 200; (108-9) - (99-11) = 11 of 200.
    assert row["sel_mean"] == pytest.approx(6.0)
    assert row["sel_draw"] == pytest.approx(5.5)
    # (110-10) - (102-13) = 11 of 200.
    assert row["sel_boot_mean"] == pytest.approx(5.5)
    # (100-12) - (102-13) = -1 of 200.
    assert row["mass_effect"] == pytest.approx(-0.5)
    # (103-10) / (105-11) - 1
    assert row["own_draw_gap"] == pytest.approx(100 * (93 / 94 - 1))
    assert row["adaptive_draw_gap"] == pytest.approx(100 * (99 / 100 - 1))
    assert row["honest_draw_gap"] == pytest.approx(100 * (88 / 89 - 1))
    assert row["honest_weight_draw_gap"] == pytest.approx(100 * (108 / 88 - 1))


def test_a_skipped_sign_drops_the_fold_and_a_constant_sign_keeps_its_mean(tmp_path, monkeypatch):
    constant = sign(50, 50, 50, 50, 50, 50)
    del constant["draw_totals"], constant["own_forest_draw_total"]
    kept = {"t": {"reported": 100.0, "signs": {"pos": constant}}}
    skipped = {"t": {"reported": 100.0, "signs": {"pos": {"skipped": "one donor"}}}}
    results = tmp_path / "results"
    results.mkdir()
    (results / "selection_fit_draws.json").write_text(json.dumps(doc([kept, skipped])))
    monkeypatch.setattr(D, "ROOT", tmp_path)
    row = D.table().iloc[0]
    assert row["folds"] == 1
    assert row["sel_draw"] == 0.0 and row["own_draw_gap"] == 0.0
