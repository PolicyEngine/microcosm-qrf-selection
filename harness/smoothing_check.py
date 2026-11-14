"""Synthetic data with a known conditional mean: what adaptive minus honest carries.

``contrast_parts`` splits the same-tree contrast exactly into selection (the
catchments' dependence on the donors' own noise) and smoothing (the two fills'
different averaging of the conditional mean). This script runs that split on
processes where the conditional mean m(x) is known, and asks two questions:

1. How large is smoothing beside selection, for Microcosm-style trees grown on
   the target, for trees grown on separate noise (the reviewer's control) and
   for trees grown on the target permuted across donors?
2. Does an estimate of m, fitted on the held-out donors (who never fill a
   tree), recover the smoothing part? On real data the true m is unknown, and
   that estimate is the only route to the split.

Every number is a percent of the recipients' true total, mean over seeds with
its standard error. Writes ``results/public/smoothing_check.csv``.

Usage: python smoothing_check.py [--seeds 40] [--out ../results/public/smoothing_check.csv]
       python smoothing_check.py --first-seed 40 --seeds 200 --sizes 1200 \
           --processes lognormal "lognormal, heavy tail" --grown-on noise \
           --out ../results/public/smoothing_check_more_seeds.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import selection as S  # noqa: E402

OUT = HERE.parent / "results/public/smoothing_check.csv"
TREES = 30
#: Held-out donors per training donor: five folds hold one out for four in.
HELD_OUT_SHARE = 0.25


def _lognormal(X, rng, sigma, coef):
    lin = X[:, : len(coef)] @ np.asarray(coef)
    return np.exp(lin + sigma * rng.normal(size=len(X))), np.exp(lin + sigma ** 2 / 2)


def draw_process(name, n, rng):
    """Predictors, target and its conditional mean for one sample."""
    if name == "deterministic":  # the reviewer's process: y = m(x) exactly
        X = rng.normal(size=(n, 3))
        m = np.exp(X[:, 0])
        return X, m.copy(), m
    if name == "lognormal":
        X = rng.normal(size=(n, 5))
        return (X, *_lognormal(X, rng, 1.0, (0.8, 0.4)))
    if name == "lognormal, discrete predictors":
        X = np.column_stack([rng.normal(size=n), rng.integers(0, 6, n), rng.integers(0, 2, n),
                             rng.integers(0, 4, n), rng.integers(0, 10, n)]).astype(float)
        return (X, *_lognormal(X, rng, 1.0, (0.8, 0.25, 0.5)))
    if name == "lognormal, heavy tail":
        X = rng.normal(size=(n, 5))
        return (X, *_lognormal(X, rng, 1.5, (0.8, 0.4)))
    if name == "no signal":  # known_answer.py's process: m constant
        X = rng.normal(size=(n, 3))
        y = np.exp(rng.normal(size=n))
        return X, y, np.full(n, np.exp(0.5))
    raise ValueError(name)


PROCESSES = ("deterministic", "lognormal", "lognormal, heavy tail", "lognormal, discrete predictors",
             "no signal")
GROWN_ON = ("target", "noise", "permuted target")


def m_hat(X_fit, y_fit, w_fit, seed):
    """The conditional-mean estimate the real-data run uses (``selection.mean_model``)."""
    return S.mean_model(X_fit, y_fit, w_fit, seed)


def one_sample(process, n, grown_on, seed):
    rng = np.random.default_rng([seed, 11])
    X, y, m = draw_process(process, n, rng)
    w = rng.uniform(1, 4, size=n)
    n_rec = int(round(HELD_OUT_SHARE * n))
    X_r, y_r, m_r = draw_process(process, n_rec, rng)
    w_r = rng.uniform(1, 4, size=n_rec)
    truth = float(w_r @ m_r)
    model_hat = m_hat(X_r, y_r, w_r, seed)
    mh = model_hat.predict(X)
    half = S.halves(n, None, np.random.default_rng([seed, 31]))
    sums = {(kind, mode): np.zeros(3) for kind in ("exact", "estimated") for mode in S.MODES}
    for k, grow_mask in enumerate((half, ~half)):
        idx_g, idx_f = np.flatnonzero(grow_mask), np.flatnonzero(~grow_mask)
        pick = idx_g[S.qd.weighted_bootstrap(len(idx_g), w[idx_g], np.random.default_rng([seed, 7, k]))]
        if grown_on == "target":
            z = y[pick]
        elif grown_on == "noise":
            z = np.random.default_rng([seed, 99, k]).normal(size=len(pick))
        else:
            z = np.random.default_rng([seed, 98, k]).permutation(y[pick])
        model = S.grow(X[pick], z, seed, TREES)
        cnt = S.qd.bootstrap_counts(model, len(pick))
        mass_f = np.broadcast_to(w[idx_f] / w[idx_f].mean(), (model.n_estimators, len(idx_f)))
        mm = np.column_stack([m, mh])
        parts = S.contrast_parts(model, X[pick], cnt, y[pick], mm[pick],
                                 X[idx_f], mass_f, y[idx_f], mm[idx_f], X_r, w_r)
        for mode in S.MODES:
            p = parts[mode]
            for i, kind in enumerate(("exact", "estimated")):
                sums[(kind, mode)] += 0.5 * np.array(
                    [p["contrast"], p["selection"][i], p["smoothing"][i]])
    return {key: 100 * v / truth for key, v in sums.items()}


def _one(args):
    S.N_JOBS = 1  # tiny forests: one thread each, four samples at a time
    return one_sample(*args)


def _se(a):
    return a.std(ddof=1) / np.sqrt(len(a))


def table(seeds: int, sizes=(1200, 4800), workers: int = 4, first_seed: int = 0,
          processes=PROCESSES, grown=GROWN_ON) -> pd.DataFrame:
    from multiprocessing import get_context

    cases = [(process, n, grown_on) for process in processes for n in sizes for grown_on in grown]
    rows = []
    with get_context("spawn").Pool(workers) as pool:
        for process, n, grown_on in cases:
            res = pool.map(_one, [(process, n, grown_on, seed)
                                  for seed in range(first_seed, first_seed + seeds)])
            for mode in S.MODES:
                ex = np.array([r[("exact", mode)] for r in res])
                es = np.array([r[("estimated", mode)] for r in res])
                rows.append({
                    "process": process, "donors": n, "trees grown on": grown_on, "mode": mode,
                    "seeds": seeds, "first seed": first_seed, "trees": TREES,
                    "contrast": ex[:, 0].mean(), "contrast_se": _se(ex[:, 0]),
                    "selection": ex[:, 1].mean(), "selection_se": _se(ex[:, 1]),
                    "smoothing": ex[:, 2].mean(), "smoothing_se": _se(ex[:, 2]),
                    "smoothing_estimated": es[:, 2].mean(), "smoothing_estimated_se": _se(es[:, 2]),
                    "selection_estimated": es[:, 1].mean(), "selection_estimated_se": _se(es[:, 1]),
                    "estimate_error": (es[:, 2] - ex[:, 2]).mean(),
                    "estimate_error_se": _se(es[:, 2] - ex[:, 2]),
                })
            print(process, n, grown_on, "done", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1200, 4800])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--processes", nargs="+", default=list(PROCESSES), choices=PROCESSES)
    ap.add_argument("--grown-on", nargs="+", default=list(GROWN_ON), choices=GROWN_ON)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    df = table(args.seeds, tuple(args.sizes), args.workers, args.first_seed,
               tuple(args.processes), tuple(args.grown_on))
    df.to_csv(args.out, index=False)
    cols = ["process", "donors", "trees grown on", "mode", "contrast", "selection", "smoothing",
            "smoothing_estimated", "estimate_error"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))


if __name__ == "__main__":
    main()
