"""Same-tree selection test for any target Microcosm's RegimeGatedQRF imputes.

For one donor frame (predictors, chained targets, survey weights) and the
configuration a production stage fits it with (``n_estimators``, ``seed``),
this runs the test ``harness/microcosm_cps.py`` in PolicyEngine/qrf-moop-shortfall ran
on CPS MOOP, for every target and every sign forest:

1. Split the donors into folds (whole households when rows share one). For
   fold k, fit ``RegimeGatedQRF`` on the other folds with the production
   configuration. The fit runs Microcosm's own code with ``_fit_forest``
   replaced by a recorder that draws the weighted bootstrap exactly as the
   real one does (``rng.choice`` with the same arguments, so the RNG stream
   matches) but grows no trees. The gates, regimes, chained columns, forest
   seeds and bootstrap rows therefore equal a real fit's
   (``tests/test_selection.py`` checks this against a real fit).
2. For each sign forest, take that sign's training donors. Grow forests the
   way Microcosm grows its amount forest (``RandomForestQuantileRegressor``
   with ``max_samples_leaf=None`` and every growth parameter at its default,
   read by count pooling) on a weighted bootstrap of one random half of them.
   Fill the same trees with their own in-bag rows (adaptive) and with the
   other half's donors, each carrying survey-weight mass (honest). Swap the
   halves and average.
3. Weight each held-out recipient's estimate by its survey weight times the
   fold gate's probability of that sign, and sum over signs. Adaptive minus
   honest isolates what the splits' sight of the donors' own values does to
   the total. A recipient no honest donor reaches in any tree keeps its
   adaptive estimate (``complete_honest``), so the difference runs over the
   recipients both fills reach; ``max_empty_adaptive_share`` records how much
   of the adaptive total such recipients hold.

Remedy variants share each half's bootstrap rows and ``random_state``: leaves
of at least 5 or 20 samples (``min_samples_leaf``), per-tree leaf weights
("tree" mode, which is ``predict(weighted_leaves=True)`` under adaptive fills)
and honest fills. With ``quantiles=True`` each variant also reports pinball
loss and coverage of its conditional quantiles on the held-out recipients of
that sign.

Reuses ``qrfdecomp.weighted_bootstrap`` / ``bootstrap_counts`` and
``forest_bias.fill_catchments`` / ``recipient_estimates`` (``harness/vendor``,
byte-identical copies from qrf-moop-shortfall at dba3940) unchanged.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from quantile_forest import RandomForestQuantileRegressor

PRIOR = Path(__file__).resolve().parent / "vendor"  # qrf-moop-shortfall@dba3940
sys.path.insert(0, str(PRIOR))

import qrfdecomp as qd  # noqa: E402
from forest_bias import fill_catchments, recipient_estimates  # noqa: E402

N_JOBS = 4
POOL_CHECK_MAX_ROWS = 50_000
TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)
MODES = ("pooled", "tree")
FILLS = ("adaptive", "honest")

os.environ.setdefault("POPULACE_FIT_N_JOBS", str(N_JOBS))
os.environ.setdefault("POPULACE_FIT_PREDICT_WORKERS", str(N_JOBS))


# --------------------------------------------------------------------------
# Microcosm's fit, without growing its forests
# --------------------------------------------------------------------------


@dataclass
class ForestRecord:
    """What a real ``_fit_forest`` call would have grown its forest from."""

    seed: int
    selected: np.ndarray  # bootstrap rows, positions within the sign's rows
    n_estimators: int
    max_samples_leaf: object
    n_unique: int


@dataclass
class CapturedTarget:
    regime: str
    columns: tuple[str, ...]
    gate: object | None
    positive: ForestRecord | None
    negative: ForestRecord | None


@dataclass
class CapturedFit:
    weights: np.ndarray | None
    targets: dict[str, CapturedTarget] = field(default_factory=dict)


def _qrf_module():
    import microcosm.fit.qrf as Q

    return Q


def capture_fit(
    table: pd.DataFrame,
    predictors: list[str],
    targets: list[str],
    weights,
    *,
    seed: int,
    n_estimators: int,
    zero_atol: float | None = None,
    max_samples_leaf=None,
) -> CapturedFit:
    """Run ``RegimeGatedQRF.fit`` with forests recorded instead of grown."""
    Q = _qrf_module()
    if zero_atol is None:
        zero_atol = Q.DEFAULT_ZERO_ATOL

    def record_forest(x, y, columns, weights, *, seed, n_estimators, max_samples_leaf, rng):
        # Same draw as Q._weighted_bootstrap: rng.choice(len(x), len(x), p=w/sum(w)).
        if weights is None:
            selected = np.arange(len(x))
        else:
            total = float(weights.sum())
            selected = rng.choice(len(x), size=len(x), replace=True, p=weights / total)
        record = ForestRecord(int(seed), selected, int(n_estimators), max_samples_leaf,
                              int(len(np.unique(y))))
        return Q._Forest(model=record, columns=tuple(columns))

    resolved = Q._resolve_qrf_fit_input(table, list(predictors), list(targets), weights)
    real = Q._fit_forest
    Q._fit_forest = record_forest
    try:
        fitted = Q.RegimeGatedQRF(n_estimators=n_estimators, zero_atol=zero_atol,
                                  max_samples_leaf=max_samples_leaf, seed=seed).fit(
            table, list(predictors), list(targets), weights=weights)
    finally:
        Q._fit_forest = real
    out = CapturedFit(weights=resolved.weights)
    for target in targets:
        tm = fitted._target_models[target]
        out.targets[target] = CapturedTarget(
            regime=tm.regime, columns=tuple(tm.columns), gate=tm.gate,
            positive=None if tm.positive is None else tm.positive.model,
            negative=None if tm.negative is None else tm.negative.model)
    return out


# --------------------------------------------------------------------------
# Forests built Microcosm's way, and their fills
# --------------------------------------------------------------------------


def grow(X, y, seed, n_estimators, min_samples_leaf=1, store_leaf_samples=False):
    """A forest built like Microcosm's amount forest (``qrf.py`` ``_fit_forest``).

    ``min_samples_leaf`` other than 1 is a remedy variant, not Microcosm's
    construction. ``max_samples_leaf`` only decides which leaf samples a
    fitted quantile forest stores, after the trees grow; the fills below read
    leaves from ``apply`` and bootstrap counts, never from stored samples. So
    the forest stores one sample per leaf: with ``None`` quantile-forest pads
    every tree's stored samples to the largest leaf, which on a target with a
    handful of distinct values (pure leaves of tens of thousands of donors)
    runs to tens of gigabytes. Tree structure is identical either way
    (``tests/test_selection.py::test_stored_leaf_samples_leave_trees_unchanged``).
    ``store_leaf_samples=True`` keeps every sample, for the one check that
    calls ``predict``.
    """
    model = RandomForestQuantileRegressor(
        n_estimators=int(n_estimators), max_samples_leaf=None if store_leaf_samples else 1,
        min_samples_leaf=int(min_samples_leaf), random_state=int(seed), n_jobs=N_JOBS)
    return model.fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))


def sparse_pool(leaves_fill, mass, leaves_rec, node_counts, mode, order):
    """Each recipient's normalized weight on each filling row (CSR).

    Columns follow ``order`` (filling rows sorted by value), so each row's
    entries come out in value order once indices are sorted.
    """
    n_fill, n_trees = leaves_fill.shape
    col_of = np.empty(n_fill, dtype=np.int64)
    col_of[order] = np.arange(n_fill)
    A = None
    for t in range(n_trees):
        m = np.asarray(mass[t], dtype=np.float64)
        nz = m > 0
        lf = leaves_fill[nz, t]
        val = m[nz]
        if mode == "tree":
            M = np.bincount(leaves_fill[:, t], weights=m, minlength=node_counts[t])
            val = val / M[lf]
        F = sp.csr_matrix((val, (lf, col_of[np.flatnonzero(nz)])),
                          shape=(node_counts[t], n_fill))
        At = F[leaves_rec[:, t], :]
        A = At if A is None else A + At
    A = A.tocsr()
    A.sum_duplicates()
    A.sort_indices()
    return A


def quantile_metrics(A, y_sorted, y_eval, w_eval, taus=TAUS):
    """Pinball loss and coverage of each row's pooled conditional quantiles.

    A row with no filling mass has no quantiles and scores nothing.
    """
    counts = np.diff(A.indptr)
    if (counts == 0).any():
        keep = counts > 0
        A, y_eval, w_eval = A[keep], y_eval[keep], w_eval[keep]
        counts = np.diff(A.indptr)
    if A.shape[0] == 0:
        return None
    rs = np.asarray(A.sum(axis=1)).ravel()
    data = A.data / np.repeat(rs, counts)
    cum = np.cumsum(data)
    base = np.concatenate([[0.0], cum])[A.indptr[:-1]]
    last = A.indptr[1:] - 1
    out = {"pinball": {}, "coverage": {}}
    W = w_eval.sum()
    for tau in taus:
        pos = np.searchsorted(cum, base + tau - 1e-12, side="left")
        pos = np.minimum(np.maximum(pos, A.indptr[:-1]), last)
        q = y_sorted[A.indices[pos]]
        loss = (y_eval - q) * (tau - (y_eval < q))
        out["pinball"][str(tau)] = float((w_eval * loss).sum() / W)
        out["coverage"][str(tau)] = float((w_eval * (y_eval <= q)).sum() / W)
    out["mean_pinball"] = float(np.mean(list(out["pinball"].values())))
    return out


def fill_forest(model, X_fill, mass, y_fill, X_rec, scale_rec, rows_to_donor, n_donor,
                *, eval_set=None, eval_keep=None, chunk=2000):
    """Direct totals (and catchment identity) for one filled forest, both modes.

    Each mode's result carries ``est``, the per-recipient estimates (NaN where
    no tree holds filling mass for the recipient); ``same_tree_sign`` uses them
    and does not store them. ``eval_set`` = (X_eval, y_eval, w_eval) adds
    quantile metrics per mode over the rows ``eval_keep[mode]`` selects
    (default: every row with filling mass, returned as ``eval_keep``, so that
    every fill and variant of a half can score the same rows).
    """
    nodes = [e.tree_.node_count for e in model.estimators_]
    lf, lr = model.apply(X_fill), model.apply(X_rec)
    out = {}
    order = np.argsort(y_fill, kind="stable")
    y_sorted = np.asarray(y_fill, dtype=np.float64)[order]
    le = model.apply(eval_set[0]) if eval_set is not None else None
    for mode in MODES:
        with np.errstate(invalid="ignore", divide="ignore"):
            est = recipient_estimates(lf, mass, y_fill, lr, nodes, mode)
        ok = np.isfinite(est)
        direct = float((est[ok] * scale_rec[ok]).sum())
        res = {"total": direct, "n_empty": int((~ok).sum()), "est": est}
        if ok.all():
            c_rows = fill_catchments(lf, mass, lr, scale_rec, nodes, mode)
            via_catchment = float((c_rows * y_fill).sum())
            res["identity_rel_err"] = abs(via_catchment / direct - 1) if direct else 0.0
            res["catchment_sum_over_scale"] = float(c_rows.sum() / scale_rec.sum())
        else:
            # A recipient with no filling mass in any tree has no estimate; the
            # total skips it and the catchment identity is not checked.
            res["identity_rel_err"] = 0.0
            res["catchment_sum_over_scale"] = float("nan")
        if eval_set is not None:
            _, y_eval, w_eval = eval_set
            parts = []
            for start in range(0, len(y_eval), chunk):
                stop = min(start + chunk, len(y_eval))
                A = sparse_pool(lf, mass, le[start:stop], nodes, mode, order)
                parts.append((A, start, stop))
            # Stack chunks back into one CSR for the metric pass.
            A_all = sp.vstack([p[0] for p in parts]).tocsr()
            keep = np.diff(A_all.indptr) > 0 if eval_keep is None else eval_keep[mode]
            res["eval_keep"] = keep
            res["quantile"] = quantile_metrics(A_all[keep], y_sorted, y_eval[keep], w_eval[keep])
        out[mode] = res
    return out


def recipient_totals(model, X_fill, mass, y_fill, X_rec, scale_rec):
    """Direct totals over another recipient set (no catchments, no quantiles).

    A recipient whose leaves hold no filling mass in any tree gets no estimate
    (NaN in ``est_<mode>``); ``n_empty_<mode>`` counts them.
    """
    nodes = [e.tree_.node_count for e in model.estimators_]
    lf, lr = model.apply(X_fill), model.apply(X_rec)
    out = {}
    for mode in MODES:
        with np.errstate(invalid="ignore", divide="ignore"):
            est = recipient_estimates(lf, mass, y_fill, lr, nodes, mode)
        ok = np.isfinite(est)
        out[mode] = float((est[ok] * scale_rec[ok]).sum())
        out[f"n_empty_{mode}"] = int((~ok).sum())
        out[f"est_{mode}"] = est
    return out


def complete_honest(adaptive_est, honest_est, scale):
    """Add to the honest total the recipients no honest donor reaches.

    A recipient whose leaves hold no honest donor in any tree has no honest
    estimate. It keeps its adaptive estimate, so it adds nothing to adaptive
    minus honest: the difference runs over the recipients both fills reach.
    Returns the amount to add to the honest total and those recipients' share
    of the adaptive total (absolute values).
    """
    gap = ~np.isfinite(honest_est)
    if not gap.any():
        return 0.0, 0.0
    added = adaptive_est[gap] * scale[gap]
    denom = float(np.abs(adaptive_est * scale).sum())
    return float(added.sum()), (float(np.abs(added).sum()) / denom if denom else 0.0)


# --------------------------------------------------------------------------
# The same-tree test for one sign forest
# --------------------------------------------------------------------------


def halves(n, groups, rng):
    """Boolean grow-half mask, whole groups together when ``groups`` is given."""
    if groups is None:
        return rng.random(n) < 0.5
    codes, inverse = np.unique(groups, return_inverse=True)
    return (rng.random(len(codes)) < 0.5)[inverse]


def same_tree_sign(X_s, y_s, w_s, X_rec, scale, *, seed, n_estimators, variants=(1,),
                   groups=None, eval_set=None, full_record=None, extra=None):
    """Adaptive and honest totals of half-sample forests for one sign.

    Seeds follow ``microcosm_cps.analyse_fold``: halves from
    ``default_rng([seed, 31])``, the k-th half's bootstrap from
    ``default_rng([seed, 7, k])``, forests at ``random_state=seed``.
    ``full_record`` (a ``ForestRecord`` from ``capture_fit``) adds the fold's
    own Microcosm forest, rebuilt from its seed and bootstrap rows.
    ``extra`` maps a name to (X, scale) for further recipient sets (the
    production recipients); the same forests and fills give their totals.
    """
    n = len(y_s)
    extra = extra or {}
    out = {"n": int(n), "variants": {}}
    totals = {v: {f: {m: 0.0 for m in MODES} for f in FILLS} for v in variants}
    xtot = {e: {v: {f: {m: 0.0 for m in MODES} for f in FILLS} for v in variants}
            for e in extra}
    xempty = {e: 0 for e in extra}
    xgap = {e: {m: 0.0 for m in MODES} for e in extra}
    qfit = {v: {f: {m: [] for m in MODES} for f in FILLS} for v in variants}
    identity = 0.0
    empty = {f: {m: 0 for m in MODES} for f in FILLS}
    gap_share = {m: 0.0 for m in MODES}
    half = halves(n, groups, np.random.default_rng([seed, 31]))
    for k, grow_mask in enumerate((half, ~half)):
        idx_g, idx_f = np.flatnonzero(grow_mask), np.flatnonzero(~grow_mask)
        if len(idx_g) == 0 or len(idx_f) == 0:
            raise RuntimeError("an empty half; too few donors for the same-tree test")
        pick_g = idx_g[qd.weighted_bootstrap(len(idx_g), w_s[idx_g],
                                             np.random.default_rng([seed, 7, k]))]
        eval_keep = None  # rows the first variant's honest fill reaches; every fill scores them
        for v in variants:
            model = grow(X_s[pick_g], y_s[pick_g], seed, n_estimators, v)
            cnt = qd.bootstrap_counts(model, len(pick_g))
            mass_f = np.broadcast_to(w_s[idx_f] / w_s[idx_f].mean(),
                                     (model.n_estimators, len(idx_f)))
            honest = fill_forest(model, X_s[idx_f], mass_f, y_s[idx_f], X_rec, scale,
                                 idx_f, n, eval_set=eval_set, eval_keep=eval_keep)
            if eval_set is not None and eval_keep is None:
                eval_keep = {m: honest[m]["eval_keep"] for m in MODES}
            adaptive = fill_forest(model, X_s[pick_g], cnt, y_s[pick_g], X_rec, scale,
                                   pick_g, n, eval_set=eval_set, eval_keep=eval_keep)
            fills = {"adaptive": adaptive, "honest": honest}
            for m in MODES:
                added, share = complete_honest(adaptive[m]["est"], honest[m]["est"], scale)
                honest[m]["total"] += added
                gap_share[m] = max(gap_share[m], share)
            for f in FILLS:
                for m in MODES:
                    totals[v][f][m] += 0.5 * fills[f][m]["total"]
                    identity = max(identity, fills[f][m]["identity_rel_err"])
                    empty[f][m] = max(empty[f][m], fills[f][m]["n_empty"])
                    if eval_set is not None:
                        qfit[v][f][m].append(fills[f][m]["quantile"])
            for e, (X_e, scale_e) in extra.items():
                tots = {"adaptive": recipient_totals(model, X_s[pick_g], cnt, y_s[pick_g],
                                                     X_e, scale_e),
                        "honest": recipient_totals(model, X_s[idx_f], mass_f, y_s[idx_f],
                                                   X_e, scale_e)}
                for m in MODES:
                    added, share = complete_honest(tots["adaptive"][f"est_{m}"],
                                                   tots["honest"][f"est_{m}"], scale_e)
                    tots["honest"][m] += added
                    xgap[e][m] = max(xgap[e][m], share)
                    for f in FILLS:
                        xtot[e][v][f][m] += 0.5 * tots[f][m]
                        xempty[e] = max(xempty[e], tots[f][f"n_empty_{m}"])
    for v in variants:
        rec = {"totals": totals[v]}
        if eval_set is not None:
            rec["quantile"] = {
                f: {m: _average_metrics(qfit[v][f][m]) for m in MODES} for f in FILLS}
        out["variants"][str(v)] = rec
    out["identity_max_rel_err"] = identity
    out["max_empty_recipients"] = empty
    out["max_empty_adaptive_share"] = gap_share
    if full_record is not None:
        # The pooled-mean cross-check calls quantile-forest's own ``predict``,
        # which reads stored leaf samples; it runs where storing them is cheap.
        check = len(full_record.selected) <= POOL_CHECK_MAX_ROWS
        model = grow(X_s[full_record.selected], y_s[full_record.selected],
                     full_record.seed, full_record.n_estimators, store_leaf_samples=check)
        cnt = qd.bootstrap_counts(model, n)
        full = fill_forest(model, X_s[full_record.selected], cnt, y_s[full_record.selected],
                           X_rec, scale, full_record.selected, n)
        out["own_forest"] = {m: full[m]["total"] for m in MODES}
        out["own_forest_pool_mean_check"] = (float(
            (scale * np.asarray(model.predict(X_rec, quantiles="mean"))).sum())
            if check else None)
        own_extra = {e: recipient_totals(model, X_s[full_record.selected], cnt,
                                         y_s[full_record.selected], X_e, scale_e)
                     for e, (X_e, scale_e) in extra.items()}
    else:
        own_extra = {}
    if extra:
        out["extra"] = {e: {"variants": {str(v): {"totals": xtot[e][v]} for v in variants},
                            "own_forest": {m: own_extra[e][m] for m in MODES}
                            if e in own_extra else None,
                            "max_empty": xempty[e],
                            "max_empty_adaptive_share": xgap[e]} for e in extra}
    return out


def _average_metrics(items):
    items = [it for it in items if it is not None]
    if not items:
        return None
    out = {"pinball": {}, "coverage": {}}
    for key in ("pinball", "coverage"):
        for tau in items[0][key]:
            out[key][tau] = float(np.mean([it[key][tau] for it in items]))
    out["mean_pinball"] = float(np.mean([it["mean_pinball"] for it in items]))
    return out


# --------------------------------------------------------------------------
# A whole stage: folds x targets x signs
# --------------------------------------------------------------------------


def assign_folds(n, groups, k, seed):
    """Fold per row; whole groups share a fold."""
    rng = np.random.default_rng([int(seed), 5])
    if groups is None:
        return rng.integers(0, k, size=n)
    codes, inverse = np.unique(groups, return_inverse=True)
    return rng.integers(0, k, size=len(codes))[inverse]


def sign_probability(gate, X, sign):
    classes = list(np.asarray(gate.classes_).tolist())
    return np.asarray(gate.predict_proba(X))[:, classes.index(sign)]


def analyse_fold(table, predictors, targets, weights, *, train, rec, seed, n_estimators,
                 groups=None, variants=(1,), quantiles=False, only=None, zero_atol=None,
                 max_samples_leaf=None, log=print, production=None):
    """One fold: capture Microcosm's fit on ``train``; test every target on ``rec``.

    ``production`` = (table, weights): the stage's production recipients, with
    the predictor columns and each target's production draw (the values a
    chained target conditions on). The same forests then also give totals over
    them, with each recipient's amount weighted by the fold gate's probability.
    """
    Q = _qrf_module()
    zero_atol = Q.DEFAULT_ZERO_ATOL if zero_atol is None else zero_atol
    t0 = time.time()
    train_table = table.loc[train].reset_index(drop=True)
    rec_table = table.loc[rec].reset_index(drop=True)
    if isinstance(weights, str):
        w_all = table[weights].to_numpy(dtype=np.float64)
        w_train_spec = weights
    else:
        w_all = np.asarray(weights, dtype=np.float64)
        w_train_spec = w_all[train]
    w_rec = w_all[rec]
    # A sign class with one training donor cannot pass the gate's early-stopping
    # split (HistGradientBoostingClassifier stratifies its validation split once
    # a fit has more than 10,000 rows), so Microcosm's fit raises. Production
    # never meets this on its full donor set; a training fold can, for a target
    # with two or three donors of one sign. Such a target leaves this fold's fit
    # (the others keep their order) and the fold records why. Later targets in
    # the chain then lose its column as a feature in that fold; it is nearly
    # all zeros (one donor of a sign), so the forests barely change.
    gate_skips = {}
    if len(train_table) > 10_000:
        for target in targets:
            y = train_table[target].to_numpy(dtype=np.float64)
            labels = np.where(y > zero_atol, 1, np.where(y < -zero_atol, -1, 0))
            _, counts = np.unique(labels, return_counts=True)
            if len(counts) > 1 and counts.min() == 1:
                gate_skips[target] = ("a sign class with one training donor in this fold; "
                                      "the gate's stratified early-stopping split cannot fit")
    fit_targets = [t for t in targets if t not in gate_skips]
    captured = capture_fit(train_table, predictors, fit_targets, w_train_spec, seed=seed,
                           n_estimators=n_estimators, zero_atol=zero_atol,
                           max_samples_leaf=max_samples_leaf)
    w_train = captured.weights
    if w_train is None:
        w_train = np.ones(len(train_table))
    g_train = None if groups is None else np.asarray(groups)[train]
    log(f"  fit captured in {time.time() - t0:.0f}s")
    out = {"n_train": int(train.sum()), "n_rec": int(rec.sum()), "targets": {}}
    for target in targets:
        if only is not None and target not in only:
            continue
        if target in gate_skips:
            out["targets"][target] = {"regime": "gate_unfittable_in_fold",
                                      "signs": {"gate": {"skipped": gate_skips[target]}}}
            log(f"  {target}: not measured in this fold: {gate_skips[target]}")
            continue
        ct = captured.targets[target]
        y_train = train_table[target].to_numpy(dtype=np.float64)
        y_rec = rec_table[target].to_numpy(dtype=np.float64)
        rec_out = {
            "regime": ct.regime,
            "reported": float((w_rec * y_rec).sum()),
            "reported_abs": float((w_rec * np.abs(y_rec)).sum()),
            "reported_pos": float((w_rec * np.where(y_rec > zero_atol, y_rec, 0)).sum()),
            "reported_neg": float((w_rec * np.where(y_rec < -zero_atol, y_rec, 0)).sum()),
            "n_unique_train": int(len(np.unique(y_train))),
            "signs": {},
        }
        if production is not None:
            p_table, p_w = production
            drawn = p_table[target].to_numpy(dtype=np.float64)
            rec_out["production"] = {"drawn_total": float((p_w * drawn).sum()),
                                     "drawn_abs": float((p_w * np.abs(drawn)).sum()),
                                     "n": int(len(drawn))}
        if ct.regime == Q.Regime.DEGENERATE_ZERO:
            out["targets"][target] = rec_out
            continue
        cols = list(ct.columns)
        X_train = train_table[cols].to_numpy(dtype=np.float64)
        X_rec = rec_table[cols].to_numpy(dtype=np.float64)
        X_prod = (None if production is None
                  else production[0][cols].to_numpy(dtype=np.float64))
        for sign, record in ((1, ct.positive), (-1, ct.negative)):
            if record is None:
                continue
            mask = y_train > zero_atol if sign == 1 else y_train < -zero_atol
            extra = {}
            if ct.regime in (Q.Regime.POSITIVE_ONLY, Q.Regime.NEGATIVE_ONLY):
                mask = np.ones(len(y_train), dtype=bool)
                p = np.ones(len(y_rec))
                if X_prod is not None:
                    extra["production"] = (X_prod, production[1].copy())
            else:
                p = sign_probability(ct.gate, X_rec, sign)
                if X_prod is not None:
                    extra["production"] = (X_prod, production[1]
                                           * sign_probability(ct.gate, X_prod, sign))
            scale = w_rec * p
            X_s, y_s, w_s = X_train[mask], y_train[mask], w_train[mask]
            eval_set = None
            if quantiles:
                emask = y_rec > zero_atol if sign == 1 else y_rec < -zero_atol
                if emask.sum() >= 5:
                    eval_set = (X_rec[emask], y_rec[emask], w_rec[emask])
            unique_y = len(np.unique(y_s))
            res = {"n_unique": int(unique_y), "p_weighted_share": float(np.average(p, weights=w_rec))}
            if unique_y == 1:
                # A constant forest: every fill returns the one value.
                value = float(y_s[0])
                res["own_forest"] = {m: float((scale * value).sum()) for m in MODES}
                res.update({"constant_value": value, "variants": {
                    str(v): {"totals": {f: {m: float((scale * value).sum()) for m in MODES}
                                        for f in FILLS}} for v in variants}})
                if extra:
                    pt = float((extra["production"][1] * value).sum())
                    res["extra"] = {"production": {
                        "variants": {str(v): {"totals": {f: {m: pt for m in MODES}
                                                         for f in FILLS}} for v in variants},
                        "own_forest": {m: pt for m in MODES}, "max_empty": 0}}
            else:
                try:
                    res.update(same_tree_sign(
                        X_s, y_s, w_s, X_rec, scale, seed=seed, n_estimators=n_estimators,
                        variants=variants, groups=None if g_train is None else g_train[mask],
                        eval_set=eval_set, full_record=record, extra=extra))
                except RuntimeError as exc:
                    # Too few donors of this sign to split into two non-empty
                    # halves: the test cannot run, and the target is reported
                    # as unmeasured with its donor count.
                    res["skipped"] = f"{exc} (n={int(mask.sum())} donors of sign {sign})"
            rec_out["signs"][str(sign)] = res
        out["targets"][target] = rec_out
        if any("skipped" in s for s in rec_out["signs"].values()):
            log(f"  {target}: {ct.regime}, not measured: "
                + "; ".join(s["skipped"] for s in rec_out["signs"].values() if "skipped" in s)
                + f" ({time.time() - t0:.0f}s)")
            continue
        sel = sum(s["variants"]["1"]["totals"]["adaptive"]["pooled"]
                  - s["variants"]["1"]["totals"]["honest"]["pooled"]
                  for s in rec_out["signs"].values())
        denom = rec_out["reported"] if rec_out["reported"] else float("nan")
        msg = f"  {target}: {ct.regime}, selection {100 * sel / denom:+.2f}% of reported"
        if production is not None and rec_out["signs"]:
            psel = sum(s["extra"]["production"]["variants"]["1"]["totals"]["adaptive"]["pooled"]
                       - s["extra"]["production"]["variants"]["1"]["totals"]["honest"]["pooled"]
                       for s in rec_out["signs"].values())
            pden = rec_out["production"]["drawn_total"] or float("nan")
            msg += f"; production {100 * psel / pden:+.2f}% of drawn"
        log(f"{msg} ({time.time() - t0:.0f}s)")
    out["seconds"] = time.time() - t0
    return out


def run_stage(name, table, predictors, targets, weights, *, seed, n_estimators, out_path,
              groups=None, folds=5, variants=(1,), quantiles=False, only=None,
              fold_ids=None, fold_seeds=None, max_samples_leaf=None, meta=None,
              production=None):
    """All folds of one stage; resumable (finished folds are kept)."""
    out_path = Path(out_path)
    if out_path.exists():
        doc = json.loads(out_path.read_text())
    else:
        doc = {"stage": name, "meta": meta or {}, "predictors": list(predictors),
               "targets": list(targets), "seed": int(seed), "n_estimators": int(n_estimators),
               "variants": list(variants), "quantiles": bool(quantiles), "folds": []}
    done = {f["fold"] for f in doc["folds"]}
    if fold_ids is None:
        fold_ids = assign_folds(len(table), groups, folds, seed)
    for k in range(folds):
        if k in done:
            continue
        print(f"{name}: fold {k}", flush=True)
        fold_seed = int(seed) + k if fold_seeds is None else int(fold_seeds[k])
        res = analyse_fold(table, predictors, targets, weights, train=fold_ids != k,
                           rec=fold_ids == k, seed=fold_seed, n_estimators=n_estimators,
                           groups=groups, variants=variants, quantiles=quantiles, only=only,
                           max_samples_leaf=max_samples_leaf,
                           log=lambda s: print(s, flush=True), production=production)
        res["fold"] = k
        res["seed"] = fold_seed
        doc["folds"].append(res)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, indent=1))
    return doc


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def summarize(doc, variant="1"):
    """Per target: selection and remedies as % of the fold's reported total."""
    rows = []
    targets = doc["targets"]
    for target in targets:
        per_fold = []
        seen, skipped = 0, []
        for f in doc["folds"]:
            t = f["targets"].get(target)
            if t is None:
                continue  # excluded by ``only``: not run in this fold
            seen += 1
            if not t["signs"]:
                per_fold.append(None)
                continue
            if any("skipped" in s for s in t["signs"].values()):
                skipped.extend(s["skipped"] for s in t["signs"].values() if "skipped" in s)
                per_fold.append(None)
                continue
            agg = {}
            for v in t["signs"][next(iter(t["signs"]))]["variants"]:
                for fill in FILLS:
                    for m in MODES:
                        agg[(v, fill, m)] = sum(s["variants"][v]["totals"][fill][m]
                                                for s in t["signs"].values())
            own = {m: sum(s.get("own_forest", {}).get(m, np.nan) for s in t["signs"].values())
                   for m in MODES}
            if all("extra" in s for s in t["signs"].values()):
                for v_ in t["signs"][next(iter(t["signs"]))]["variants"]:
                    for fill in FILLS:
                        for m in MODES:
                            agg[("prod", v_, fill, m)] = sum(
                                s["extra"]["production"]["variants"][v_]["totals"][fill][m]
                                for s in t["signs"].values())
                for m in MODES:
                    own[("prod", m)] = sum(
                        (s["extra"]["production"]["own_forest"] or {}).get(m, np.nan)
                        for s in t["signs"].values())
            per_fold.append((t, agg, own))
        live = [p for p in per_fold if p is not None]
        if not seen:
            rows.append({"target": target, "regime": "not_run"})
            continue
        if not live:
            rows.append({"target": target,
                         "regime": "too_few_donors" if skipped else "degenerate_zero",
                         "note": skipped[0] if skipped else None})
            continue
        t0 = live[0][0]

        def pct(fn):
            # A fold whose denominator is zero (no positive donor of the target
            # in the held-out fold) has no percentage; the mean and standard
            # error run over the folds that have one.
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = []
                for t, agg, own in live:
                    try:
                        vals.append(fn(t, agg, own))
                    except ZeroDivisionError:
                        vals.append(np.nan)
            vals = np.asarray(vals, dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                return float("nan"), float("nan")
            return float(vals.mean()), (float(vals.std(ddof=1) / np.sqrt(len(vals)))
                                        if len(vals) > 1 else float("nan"))

        row = {"target": target, "regime": t0["regime"], "folds": len(live),
               "reported_total_mean": float(np.mean([t["reported"] for t, _, _ in live])),
               "reported_abs_mean": float(np.mean([t["reported_abs"] for t, _, _ in live])),
               "signs_skipped": len(skipped)}
        v = variant
        for m in MODES:
            row[f"sel_{m}"], row[f"sel_{m}_se"] = pct(
                lambda t, a, o, m=m: 100 * (a[(v, "adaptive", m)] - a[(v, "honest", m)]) / t["reported"])
            row[f"sel_{m}_abs"], row[f"sel_{m}_abs_se"] = pct(
                lambda t, a, o, m=m: 100 * (a[(v, "adaptive", m)] - a[(v, "honest", m)]) / t["reported_abs"])
        # Adaptive relative to honest fills of the same trees: a ratio of two
        # model totals, stable when a small held-out fold reports a total near
        # zero (rare targets), where the share of the reported total explodes.
        row["sel_vs_honest"], row["sel_vs_honest_se"] = pct(
            lambda t, a, o: 100 * (a[(v, "adaptive", "pooled")] / a[(v, "honest", "pooled")] - 1))
        row["own_vs_reported"], row["own_vs_reported_se"] = pct(
            lambda t, a, o: 100 * (o["pooled"] / t["reported"] - 1))
        row["adaptive_vs_reported"], row["adaptive_vs_reported_se"] = pct(
            lambda t, a, o: 100 * (a[(v, "adaptive", "pooled")] / t["reported"] - 1))
        row["honest_vs_reported"], row["honest_vs_reported_se"] = pct(
            lambda t, a, o: 100 * (a[(v, "honest", "pooled")] / t["reported"] - 1))
        if all(("prod", v, "adaptive", "pooled") in a for _, a, _ in live):
            row["prod_drawn_total"] = live[0][0]["production"]["drawn_total"]
            row["prod_drawn_abs"] = live[0][0]["production"]["drawn_abs"]
            for m in MODES:
                row[f"prod_sel_{m}"], row[f"prod_sel_{m}_se"] = pct(
                    lambda t, a, o, m=m: 100 * (a[("prod", v, "adaptive", m)]
                                                - a[("prod", v, "honest", m)])
                    / t["production"]["drawn_total"])
                row[f"prod_sel_{m}_abs"], row[f"prod_sel_{m}_abs_se"] = pct(
                    lambda t, a, o, m=m: 100 * (a[("prod", v, "adaptive", m)]
                                                - a[("prod", v, "honest", m)])
                    / t["production"]["drawn_abs"])
            row["prod_sel_vs_honest"], row["prod_sel_vs_honest_se"] = pct(
                lambda t, a, o: 100 * (a[("prod", v, "adaptive", "pooled")]
                                       / a[("prod", v, "honest", "pooled")] - 1))
            row["prod_own_vs_drawn"], row["prod_own_vs_drawn_se"] = pct(
                lambda t, a, o: 100 * (o[("prod", "pooled")] / t["production"]["drawn_total"] - 1))
            row["prod_honest_vs_drawn"], row["prod_honest_vs_drawn_se"] = pct(
                lambda t, a, o: 100 * (a[("prod", v, "honest", "pooled")]
                                       / t["production"]["drawn_total"] - 1))
        rows.append(row)
    return pd.DataFrame(rows)
