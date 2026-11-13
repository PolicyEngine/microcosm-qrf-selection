# Adaptive-split selection in Microcosm's QRF imputations

Status: 2026-11-13, awaiting independent review. Every measurement comes from
this repository's harness, on five folds per fit. The harness's later additions
left earlier numbers unchanged: two fits rerun on the final code match their
base runs number for number (`harness/compare_runs.py`). The split of each contrast into selection and smoothing (sections 2 and 4)
covers 12 of the 39 public fits so far; the rest are running, and every split
count below covers only the fits done.

This public version leaves out every number derived from the HMRC Survey of
Personal Incomes (SPI) public use tape and every UK production-recipient
number (the UK build frame holds the SPI-channel households), because the
SPI licence sends any analysis to HMRC ten working days before anyone
distributes or publishes it. It also suppresses every target whose forests
grew on fewer than 50 nonzero donors. A private addendum outside this
repository holds those numbers.

## Bottom line

- Microcosm's amount forests fill their leaves with the same rows that chose
  the splits. On held-out donors, filling the same trees with those rows
  (adaptive, as production does) instead of donors the trees never saw
  (honest) shifts the imputed total by more than its fold standard error and
  0.5% of the total on 58 of the 83 US targets and 55 of the 88 UK targets
  this repository reports. Ten targets whose nonzero values all match show
  exactly zero.
- That adaptive-minus-honest contrast is a total contrast. It carries
  selection, the splits' use of the filling rows' own values, and the two
  fills' different smoothing of the conditional mean, which differs even with
  the trees held fixed, so it doesn't measure selection by itself (section 2).
- Splitting each contrast with a conditional mean fitted on held-out donors, who
  fill no tree, leaves estimated selection above its fold standard error and
  0.5% of the total on 1 of the 4 US targets and 0 of the 2 UK targets with an
  estimate. Over those 6 targets the median absolute contrast is 1.35 points,
  estimated smoothing 0.26 and estimated selection 1.21. On synthetic
  processes with a known conditional mean, the fitted mean's smoothing misses
  the true smoothing by at most 0.77 points (section 2).
- The sign differs by target. Against honest fills, adaptive fills raise UK
  consumption (LCFS furnishings +17.4% of the held-out total, health +15.6%,
  recreation +12.0%, transport +9.4%) and US liquid assets and net worth (SCF
  net worth +6.9%, SIPP bank accounts +5.6%), and lower UK property wealth
  (WAS non-residential property −9.9%, other property −10.2%) and UK vehicle
  counts (−9.0%).
- Honest fills usually land nearer the held-out totals. Taking the absolute
  value of each fill's fold-mean signed gap to the held-out donors' reported
  total, honest fills land nearer than adaptive fills on 114 of the 161
  targets with a nonzero contrast. The median gap falls from 4.3% (adaptive)
  to 1.8% (honest); the forests as production fits them land 3.7% away.
- On US production recipients the contrast runs at about the size it runs on
  held-out donors. Outside the PUF chain, its median absolute value is 5.3% of
  the drawn total against 5.6% of the held-out total, with the same sign on 21
  of 23 targets with a nonzero contrast (SCF net worth +11.1%, SIPP bank
  accounts +11.7%, SCF auto loan balances +6.3%, SIPP vehicles +5.5%, ACS rent
  +4.7%).
- The PUF chain shows why the two can differ. In the PUF donor table six of
  the chain's eight predictors equal chain targets (dividends as the sum of
  qualified and non-qualified), so on held-out donors those forests split on
  the value they impute; on production recipients those predictors hold CPS
  values instead (section 4). Taxable interest shows −7.7% on held-out donors
  and −0.06% on the chain's ASEC-twin recipients. Targets without such a
  predictor go both ways: charitable non-cash donations show −28.8% on
  held-out donors and adaptive over honest minus one of −37.8% on the twins,
  and taxable IRA distributions +6.5% and +18.0%, while charitable cash
  donations go from +4.0% to −3.4% and partnership income from +3.3% to
  −4.9%. The twins hold 33% of the PUF units' weight and carry released values
  of the chain's earlier targets where production conditions on raw draws, so
  their column is a sensitivity on reconstructed features.
- These totals compare conditional means, and production draws from quantile
  curves. On the 36 targets where this analysis computed both, reading
  production's quantile curve moves the contrast by a median 0.06 points and
  at most 0.43, keeping every sign, against the same integer-mass honest fill.
  Against the survey-weight honest fill the rest of this report uses, the
  expected draws also carry a fresh resampling of the honest half: they keep
  the sign on 35 of 36 and move a median 0.70 points, up to 7.03 on SCF bank
  accounts (+5.77 in means, −1.26 in draws; section 2).
- The forests' conditionals also run too narrow on held-out positive donors.
  Across the 16 LCFS targets the remedies cover, 11-15% of held-out donors
  fall at or below the production-size forest's 10th percentile and 85-91% at
  or below its 90th; honest fills give 8-12% and 89-92%. Microcosm draws each
  value at a uniform random quantile of its conditional, so conditionals too
  narrow on average compress the imputed spread (section 5).
- On the 113 targets with a material held-out contrast, against the forest
  production would grow on all of a fold's donors (section 6), honest fills
  lower mean pinball loss by a median 8.5% (lower on 102 of the 113) and bring
  the imputed total a median 1.4 points nearer the held-out reported total.
  `min_samples_leaf=20` lowers pinball by a median 8.4% (lower on 97) and
  brings the total 1.3 points nearer; `min_samples_leaf=5` lowers pinball by
  6.2% and `weighted_leaves=True` by 1.9%. Honest fills raise pinball by 100%
  or more on 3 targets and `min_samples_leaf=20` on 4. An honest fill's
  residual contrast is zero by definition; on the test's half-size forests
  `min_samples_leaf=20` leaves a median residual contrast of 1.1% of the
  total, `min_samples_leaf=5` 2.4% and `weighted_leaves=True` 4.3%.
- Changing Microcosm's fit is Max's call (decision filed with this report).
  Section 9 recommends trying `min_samples_leaf=20` first, behind a setting,
  with a shadow build of each country before any switch. That recommendation
  rests on the quantile fit and the gaps to held-out totals, which don't
  depend on how the contrast splits. This analysis changed no Microcosm or
  populace code.

## 1. What production fits

PolicyEngine's app bundle 6.16.0 pins the UK release
`microcosm-uk-2024-25-national-20261025T091414Z-a6299504`, built at
microcosm 67ee2ddf3, and the US release `populace_us_2024` at
`populace-us-2024-spm-20260915`. The US release's refresh stages ran at
populace cae8640 and its base pool (the PUF tax-detail chain and the
PUF-support fills) at populace 6d06410bb.

Every QRF imputation in both builds goes through `RegimeGatedQRF`
(`microcosm/fit/qrf.py` at 67ee2ddf3; populace's `populace/fit/qrf.py` at
cae8640 and at 6d06410bb carries the same `fit`, `_fit_target`, `_fit_forest`,
`_weighted_bootstrap`, `_make_gate`, `_fit_gate` and `detect_regime`, apart
from the package name in one docstring and one log message). Line numbers
below refer to microcosm 67ee2ddf3.

- `_fit_target` (lines 1642-1690) detects each target's sign regime
  (`detect_regime`, line 119; regimes at 93-118). A strictly positive or
  strictly negative target gets one forest on all its donors; every regime
  with more than one sign class (zero-inflated, sign-only, three-sign) gets a
  sign gate and one amount forest per nonzero sign, each on that sign's
  donors only.
- The gate (`_make_gate`, 221-229; `_fit_gate`, 1698-1750) runs
  `HistGradientBoostingClassifier(random_state=seed)` with the survey weights
  as `sample_weight`.
- `_fit_forest` (634-685) resamples the sign's donors with replacement, with
  probability proportional to weight (`_weighted_bootstrap`, 189-219:
  `probabilities = weights / total`, then
  `rng.choice(len(x), size=len(x), replace=True, p=probabilities)`), then
  fits `RandomForestQuantileRegressor(n_estimators, max_samples_leaf,
  random_state=seed)` on the resample. Every other forest parameter keeps
  quantile-forest 1.4.2's default: `min_samples_leaf=1`, `bootstrap=True`,
  `max_features=1.0`. `max_samples_leaf=None` keeps every in-bag sample in
  its leaf, except on targets with 2 to 32 distinct values, where the forest
  keeps a seeded subsample of 4,096 per leaf (lines 564-572, 666-668).
- `draw` (376 onward) queries `model.predict(features, quantiles=grid)`
  (line 502) at quantile-forest's defaults, `weighted_quantile=True`,
  `weighted_leaves=False`: each recipient's conditional pools, across trees,
  the in-bag samples of the leaves it lands in, counting each by its
  bootstrap multiplicity (the harness's pooled fill reproduces
  quantile-forest's own `predict` mean to floating-point precision). Each
  recipient then takes the value at its own quantile, a uniform draw
  (`_draw_target_with_rng`, 1168-1203: `quantiles = rng.random(n)`; or the
  caller's uniforms in `predict_from_uniforms`).

So each tree's splits and its leaf contents come from the same in-bag rows.
The splits chose their cut points to separate those rows' values, and the
same values then fill the leaves.

## 2. The test

For one production fit (donor table, predictors, chained targets, weights,
seed, forest size), `harness/selection.py` runs, on 5 folds (whole households
together where rows share one):

1. Fit Microcosm's own `RegimeGatedQRF` on the other 4 folds, with
   `_fit_forest` swapped for a recorder that replays its `rng.choice` call
   (same arguments, so the random stream matches) and grows nothing. The
   gates, regimes, chained columns, forest seeds and bootstrap rows then
   equal a real fit's; `harness/tests/test_selection.py` checks that against a
   real fit.
2. For each sign forest, split that sign's training donors into two random
   halves. Grow a forest the way `_fit_forest` grows it on a weighted
   bootstrap of one half. Fill its leaves twice: with its own in-bag rows and
   their bootstrap counts (adaptive, production's construction) and with the
   other half's donors at their survey-weight mass (honest). Swap the halves
   and average.
3. Weight each held-out donor's estimate by its survey weight times the fold
   gate's probability of that sign, sum over signs, and compare with the
   held-out donors' reported total. Adaptive minus honest, with the trees
   held fixed, gives the contrast this report measures; the subsections
   below say what it carries.
4. Where the stage's production recipients exist, the same forests and fills
   give their total too, each recipient weighted by its weight times the gate
   probability.

The forests take the stage's own number of trees (100 on every stage, 32 on
the PUF chain, which `harness/capture_us_puf_chain.py` reads from
production's configuration).

This analysis calls a target's contrast material when it exceeds its fold
standard error and 0.5% of the total. For rare targets the held-out
reported total rests on few donors, so the table also gives adaptive over
honest minus one, a ratio of two model totals that small folds do not blow up.

A recipient that no honest donor reaches in any tree of a half-forest has no
honest estimate. It keeps its adaptive estimate (`complete_honest` in
`harness/selection.py`), so it adds nothing to the contrast. Such
recipients held at most 0.63% of a reported target's adaptive total on
held-out donors and 1.27% on production recipients, both on SCF net worth, and
`honest_gap_share` reports that share for every target. The share says how
much of a total takes this fallback; it doesn't bound what honest estimates
for those recipients would have given. An earlier version of the harness
dropped them from the honest total; correcting that moved no reported
target's contrast by more than 0.23 points in any fold.

The harness grows its forests with `max_samples_leaf=1` because it reads
leaves through `apply` and bootstrap counts, never through stored samples;
the stored-sample setting changes no split
(`test_stored_leaf_samples_leave_trees_unchanged`), and on near-discrete
targets `None` padded the stored samples to tens of gigabytes. One check per
sign forest still calls quantile-forest's own `predict` with every sample
stored and matches the harness's pooled total. A target whose training fold
holds one donor of a sign cannot pass the gate's stratified early-stopping
split (HistGradientBoosting turns early stopping on above 10,000 rows); the
harness drops that target from that fold's fit and records why. Every test
in `harness/tests` passes (`pytest harness/tests`, run on every push).

### A known-answer check

`harness/known_answer.py` runs the same test on synthetic data whose answer
the construction fixes: a lognormal target independent of three normal
predictors, weights uniform on 1 to 4, 15 trees, 600 recipients, 60 seeds. In
one arm the trees grow on separate noise, so the splits never see the values
that fill the leaves and the two fills have the same expected total. In the
other arm the trees grow on the target, as production's do. Both arms use the
same halves, bootstrap counts and weight masses.

| Donors | Trees grown on | Adaptive over honest − 1, %, mean over seeds |
|---|---|---|
| 1,200 | separate noise | −0.43 ± 0.67 |
| 1,200 | the target | +3.54 ± 0.72 |
| 4,800 | separate noise | +0.14 ± 0.34 |
| 4,800 | the target | +4.53 ± 0.36 |

With the splits blind to the filling values the test reads zero within one
standard error, so on this design half-sample noise, the swap of halves and
the difference between bootstrap counts and survey-weight masses do not produce
a contrast. With the splits on the target it reads a positive contrast even
though no predictor carries any signal. This design has a constant
conditional mean, which gives zero smoothing under any fills (next
subsection), so it shows selection and can't show smoothing.

### What the contrast carries

Each fill estimates a recipient's conditional mean as a weighted average of
donor values, so its total over the recipients equals the sum over donors of
c_j y_j, where donor j's catchment c_j sums its weights across recipients
(`fill_catchments` in `harness/selection.py`). Catchments depend on where the
rows sit and the mass each carries, never on the values that fill them. For
any function m of the predictors, the contrast splits exactly into two parts
(`contrast_parts`), writing a for the adaptive fill's in-bag rows and h for
the honest half's donors:

```text
adaptive − honest = [Σ c_a (y_a − m_a) − Σ c_h (y_h − m_h)]    selection
                  + [Σ c_a m_a − Σ c_h m_h]                     smoothing
```

With m the true conditional mean, the first bracket is selection. An honest
fill's catchments don't depend on its donors' noise, so its term averages
zero; the adaptive term measures how far the splits steer catchment toward
donors whose values sit above or below their conditional mean. The second
bracket is smoothing: the two fills put different donors behind each
recipient, so they average the conditional mean differently even inside the
same leaves. Both fills' catchments sum to the same recipient mass, so
smoothing vanishes when m is constant. `harness/tests/test_contrast_parts.py`
checks the identity against the fills' own totals on both averaging modes and
both recipient sets, that selection vanishes when m equals y, and that
smoothing vanishes for a constant m and scales linearly with m.

An independent reviewer of this report's previous version grew the trees on
separate noise and filled them with y = exp(x1), which carries no noise
(60 seeds, 600 donors, 15 trees). The contrast came to −3.77 ± 0.44 points of
the true total pooled and −4.35 ± 0.35 per tree. There y equals its
conditional mean, so selection is zero whatever the trees see, and the whole
contrast is smoothing.

`harness/smoothing_check.py` runs the split on five processes whose
conditional mean the construction fixes: 40 seeds each, 30 trees, pooled as
production pools, and a quarter as many held-out donors as training donors,
as five folds give. Percentages of the recipients' true total, mean over
seeds ± its standard error:

| Process | Trees grown on | Donors | Contrast | Selection | Smoothing | Smoothing from the fitted mean |
|---|---|---|---|---|---|---|
| y = exp(x1), no noise | the target | 1,200 | −1.20 ± 0.26 | +0.00 ± 0.00 | −1.20 ± 0.26 | −0.43 ± 0.14 |
| y = exp(x1), no noise | the target | 4,800 | −0.42 ± 0.08 | +0.00 ± 0.00 | −0.42 ± 0.08 | −0.00 ± 0.05 |
| y = exp(x1), no noise | separate noise | 1,200 | −3.48 ± 0.37 | +0.00 ± 0.00 | −3.48 ± 0.37 | −2.25 ± 0.25 |
| y = exp(x1), no noise | separate noise | 4,800 | −2.67 ± 0.17 | +0.00 ± 0.00 | −2.67 ± 0.17 | −2.01 ± 0.15 |
| lognormal | the target | 1,200 | +6.07 ± 1.26 | +7.01 ± 1.24 | −0.93 ± 0.32 | −0.69 ± 0.33 |
| lognormal | the target | 4,800 | +5.73 ± 0.67 | +6.24 ± 0.69 | −0.51 ± 0.14 | −0.44 ± 0.15 |
| lognormal | separate noise | 1,200 | +0.80 ± 1.04 | +3.20 ± 0.94 | −2.41 ± 0.43 | −2.00 ± 0.48 |
| lognormal | separate noise | 4,800 | −1.80 ± 0.62 | +0.09 ± 0.61 | −1.89 ± 0.15 | −1.46 ± 0.17 |
| lognormal, heavy tail | the target | 1,200 | +12.19 ± 2.32 | +13.10 ± 2.34 | −0.91 ± 0.35 | −1.02 ± 0.48 |
| lognormal, heavy tail | the target | 4,800 | +10.68 ± 1.69 | +11.07 ± 1.69 | −0.39 ± 0.12 | −0.79 ± 0.18 |
| lognormal, heavy tail | separate noise | 1,200 | +3.72 ± 1.87 | +6.13 ± 1.82 | −2.41 ± 0.43 | −2.26 ± 0.59 |
| lognormal, heavy tail | separate noise | 4,800 | −1.22 ± 1.33 | +0.67 ± 1.33 | −1.89 ± 0.15 | −1.37 ± 0.18 |
| lognormal, discrete predictors | the target | 1,200 | +6.98 ± 1.34 | +8.12 ± 1.32 | −1.14 ± 0.43 | −0.47 ± 0.28 |
| lognormal, discrete predictors | the target | 4,800 | +7.54 ± 0.69 | +7.96 ± 0.65 | −0.41 ± 0.14 | −0.34 ± 0.15 |
| lognormal, discrete predictors | separate noise | 1,200 | −2.95 ± 1.11 | −0.42 ± 1.13 | −2.54 ± 0.38 | −1.05 ± 0.25 |
| lognormal, discrete predictors | separate noise | 4,800 | −1.41 ± 0.49 | +0.17 ± 0.48 | −1.58 ± 0.13 | −1.09 ± 0.17 |
| no signal (the known-answer process) | the target | 1,200 | +3.50 ± 0.69 | +3.50 ± 0.69 | +0.00 ± 0.00 | −0.21 ± 0.10 |
| no signal (the known-answer process) | the target | 4,800 | +5.29 ± 0.42 | +5.29 ± 0.42 | +0.00 ± 0.00 | −0.16 ± 0.06 |
| no signal (the known-answer process) | separate noise | 1,200 | −0.29 ± 0.64 | −0.29 ± 0.64 | −0.00 ± 0.00 | −0.02 ± 0.11 |
| no signal (the known-answer process) | separate noise | 4,800 | +0.60 ± 0.38 | +0.60 ± 0.38 | −0.00 ± 0.00 | −0.14 ± 0.05 |

On y = exp(x1), where selection is zero, trees grown on the target carry −1.20
points of smoothing at 1,200 donors and trees grown on noise −3.48. Splits
that follow the conditional mean leave less of it varying within each leaf, so
the two fills' different donors average it more alike, and the reviewer's
noise control shows more smoothing than production-style trees carry. On the
three lognormal processes with trees grown on the target, smoothing runs −1.14
to −0.39 points and selection +6.24 to +13.10, so there the contrast falls
short of selection by up to 1.14 points. Smoothing is zero on the no-signal
process, whose conditional mean is constant.

Trees grown on noise or on the permuted target see nothing of the filling
values, so their selection should average zero. 18 of those 20 rows read
within two standard errors of zero. The other two, the lognormal process and
its heavy-tailed counterpart with trees grown on noise at 1,200 donors, read
+3.20 ± 0.94 and +6.13 ± 1.82 over 40 seeds, and they share their predictor,
noise and half draws. Over 200 further seeds they read −0.13 ± 0.41 and −0.14
± 0.82 (`results/public/smoothing_check_more_seeds.csv`), so this report reads
the first 40 as sampling noise in a skewed process.

The last column fits the conditional mean as the real-data split fits it. On
trees grown on the target its smoothing misses the true smoothing by at most
0.77 points across these processes and donor counts, and it understates the
size of smoothing on 6 of the 10 rows. Estimated selection is the contrast
minus estimated smoothing, so where the estimate understates negative
smoothing, estimated selection runs low by the same amount (`TABLES.md` gives
each error with its standard error).

`results/public/TABLES.md` adds per-tree averaging and trees grown on the
target permuted across donors.

### Splitting each target's contrast

On real data the conditional mean is unknown. `run_selection.py --smoothing`
reruns each fit's test with the same seeds, halves, bootstraps and trees. For
each fold and sign it fits `mean_model` (`HistGradientBoostingRegressor` with
`loss="poisson"` and early stopping, in `harness/selection.py`) on that sign's
held-out donors, who fill no tree, and evaluates it at the filling rows.
Estimated smoothing is then Σ c_a m̂_a − Σ c_h m̂_h and estimated selection
the contrast minus it (`harness/smoothing.py`, which refuses a run whose
contrast differs from the base run's). A sign with fewer than 100 held-out
donors in some fold gets no estimate. Where the fitted mean errs, the error
moves between the two parts and leaves the contrast unchanged; the table's
last column, fitted the same way on a held-out sample, shows how far it moves
on processes like these. Section 4 reports each target's split.

### Totals under production's draw

The test compares conditional means. Production draws one value per recipient:
`draw` queries each recipient's curve on a 203-point grid (`_grid_predictions`,
`qrf.py:474`; `_QUANTILE_GRID`, 533-540) and reads it at the recipient's
uniform quantile with linear interpolation (`_interp_rows`, 232).
quantile-forest's `calc_weighted_quantile` interpolates between neighbouring
rows, so a recipient's expected draw sits off its pooled mean. The same
formula reads masses as frequencies, so its curve depends on their scale, and
a pool holding less than one unit of mass reads as its maximum
(`test_quantile_formula_depends_on_the_scale_of_fractional_masses`;
`test_forest_quantiles_match_the_real_formula_on_any_masses` checks the
harness's copy of the formula against quantile-forest's function). The honest
fill's survey-weight masses have no natural unit. For this comparison
the honest half therefore takes integer masses built as production builds
in-bag ones (`honest_resample`: one weighted bootstrap of the half, then each
tree's bootstrap of that resample).

`run_selection.py --draw-means` computes each fill's expected total under that
curve (`expected_draws`;
`test_forest_quantiles_and_draw_means_match_quantile_forest` checks the curve
against `model.predict`). The results below cover the 36 targets the remedy
runs cover in 13 fits, the 10 fits of the scaling check (section 8) and the
three SCF fits:

| Expected draws against the pooled means of | Same sign | Median absolute contrast in means | In expected draws | Median shift | Largest shift |
|---|---|---|---|---|---|
| the honest fill at integer masses, as production builds in-bag ones | 36 of 36 | 5.11 | 5.12 | 0.06 | 0.43 |
| the honest fill at survey-weight masses, as in the rest of this report | 35 of 36 | 5.06 | 5.12 | 0.70 | 7.03 |

Percentages of the held-out reported total; the shift is the absolute
difference between the contrast in expected draws and in means. The first row
holds the honest fill fixed, so its shift measures what reading production's
quantile curve does: a median 0.06 points and at most 0.43, with the sign kept
on all 36 targets. The second row compares the same expected draws with the
survey-weight fill the rest of this report uses, so its shift also carries the
resampling that integer masses need. Resampling alone moves the contrast in
means by a median 0.79 points in absolute terms, in both directions (22
targets down, 14 up), and leaves the median absolute contrast about where it
was (5.06 at survey-weight mass, 5.11 at integer mass). Single targets move
more: SCF bank accounts goes from +5.77 at survey-weight mass to −1.70 at
integer mass and −1.26 in draws, a mass effect of −7.47 ± 5.90 driven by one
fold (−29.6). One resampling per fold can't say how stable a single target's
draw contrast is. Everywhere outside this comparison the report uses
survey-weight masses, because they carry no resampling noise.

The draw gap, the fold's full forest's expected draw over its pooled mean
minus one, runs a median 0.50% below the pooled mean and 1.49% below it on
LCFS health, a level effect the totals in this report leave out.

## 3. Reproduction

Before testing a fit, the harness reran the stage's own code on the pinned
inputs and checked the result against the release.

| Fits | Check | Result |
|---|---|---|
| US refresh stages (SCF net worth, financial assets, auto loans; SIPP assets, vehicles, tips; ORG wages; SSI criteria; Head Start; voluntary filing) | replay the stage on the release's own tables and compare every value | every value equal to the last bit; weighted totals equal |
| US ACS rent (base pool) | redraw on recipients rebuilt from the release | 54% of draws equal; total 2.9% below the release's |
| US PUF tax-detail chain (base pool) | donor table from the pinned PUF through 6d06410bb's own code; target order against `PRIMARY_QRF_TARGET_ORDER_SHA256` | donor table and fit settings are production's; production recipients rebuilt from ASEC twins (13,834 of 48,755 PUF units, 33% of their weight) |
| US PUF-support fills (base pool) | rerun each handler at 6d06410bb on the release's ASEC persons | reconstructions: the base pool that trained production's fills no longer exists, so the training pool differs; totals land within 10-35% of the release's on 11 of 17 targets and far off on the retirement distributions, disability benefits, child support paid and weeks unemployed |
| UK survey-donor stages (LCFS, ETB, WAS, LISA, NTS) | rebuild each donor with the stage's own reader, cleaning, uprating and seed from the pinned files | training rows match the build record wherever it records them (LISA 118; LCFS road-fuel redraw 2,819 against the record's rounded 2,800) |
| UK FRS-trained stages (SPI housing shell, SPI-channel stage 2, UC reporter redraw, salary sacrifice) | rebuild training rows from the release, fit at the calibration artifact's initial weights | training rows match the build's receipts (salary sacrifice 4,970; housing shell 16,288 households; stage 2's 8,556 positive rows; UC 4,286 benefit units against the receipt's 4,287); the recipient-side checks run on SPI-channel rows, so the private addendum holds them |

`results/public/TABLES.md` lists each US check.

## 4. Results

`results/public/TABLES.md` lists every measured target;
`results/public/all_targets.csv` and `results/public/smoothing.csv` hold the
numbers. The contrast below is adaptive minus honest, as a percent of the
held-out reported total, mean over 5 folds ± its standard error. Estimated
smoothing and estimated selection split it as section 2 describes, with a
dash where some fold left fewer than 100 held-out donors of a sign. The
production column uses the release's drawn total as its denominator and
carries smoothing as the held-out contrast does. Of the 171 targets this
repository reports, 113 carry a material held-out contrast: 58 of 83 in the US
(15 of 26 outside the PUF chain, 43 of its 57) and 55 of 88 in the UK.

Of the 12 reported targets, 6 have an estimated split (4 US, 2 UK); the other
6 had a fold with fewer than 100 held-out donors of a sign. Over those 6, the
median absolute contrast is 1.35 points, estimated smoothing 0.26 and
estimated selection 1.21, and estimated smoothing exceeds half the contrast on
1. Estimated selection is material on 1 of them: 1 of 4 in the US (1 of 4
outside the PUF chain, 0 of 0 on it) and 0 of 2 in the UK. Where both the
contrast and estimated selection are material (1 targets), they share a sign
on 1.

### US

| Fit | Target | Held-out contrast | Estimated smoothing | Estimated selection | Production recipients |
|---|---|---|---|---|---|
| SCF 2022 | net worth | +6.86 ± 2.76 | — | — | +11.05 ± 2.67 |
| SCF 2022 | bank account assets | +5.77 ± 1.18 | — | — | +5.15 ± 2.22 |
| SCF 2022 | auto loan balance | +1.14 ± 0.38 | — | — | +6.26 ± 0.95 |
| SCF 2022 | auto loan interest | +1.35 ± 0.40 | — | — | +4.35 ± 1.22 |
| SIPP | bank account assets | +5.56 ± 1.13 | — | — | +11.67 ± 1.35 |
| SIPP | stock assets | −3.09 ± 2.45 | — | — | +10.46 ± 6.16 |
| SIPP | household vehicles value | +1.63 ± 0.81 | — | — | +5.48 ± 1.37 |
| ACS 2022 | rent | +2.03 ± 1.46 | +0.02 ± 0.14 | +2.01 ± 1.41 | +4.71 ± 0.89 |
| CPS ORG | hourly wage | +0.70 ± 0.05 | — | — | +0.54 ± 0.15 |
| CPS ASEC (PUF-support reconstruction) | childcare expenses | +15.19 ± 7.32 | — | — | +8.44 ± 5.32 |
| CPS ASEC (PUF-support reconstruction) | weeks unemployed | +11.75 ± 4.59 | — | — | +7.34 ± 4.66 |

The SCF, SIPP and ORG fits and recipients are production's own (section 3),
and so is the ACS rent fit (its recipients come from a rebuild), so these
numbers describe the release's fits. The PUF-support numbers describe the same
method on the same kind of training sample, not production's exact fit.
Across the 23 US targets outside the PUF chain with a nonzero contrast and a
production column, the contrast on production recipients has the same sign
as on held-out donors on 21 of them, and a Spearman rank correlation of 0.81
with it. `TABLES.md`
gives the same comparison for the whole US set.

### PUF tax-detail chain

The chain fills 64 tax-detail targets from eight predictors: filing status,
tax-unit size and six income columns (`_tax_unit_feature_frame`,
`puf_support.py:1169-1196` at 6d06410bb). In the PUF donor table those income
predictors equal the chain's targets: on every row,
`puf_predictor_employment_income` equals `employment_income_before_lsr`, and
likewise for self-employment income, taxable interest and short- and long-term
capital gains, while `puf_predictor_dividend_income` equals qualified plus
non-qualified dividends. So on held-out donors those forests split on the
value they impute. Production recipients carry CPS values of those predictors
(the ASEC twins' `_tax_unit_feature_frame` values), so the held-out and
production-recipient columns can differ:

| Target | Held-out contrast | Estimated smoothing | Estimated selection | Adaptive over honest − 1, ASEC-twin recipients |
|---|---|---|---|---|
| taxable interest income | −7.70 ± 0.80 | — | — | −0.06 ± 0.02 |
| qualified dividend income | −6.88 ± 1.41 | — | — | +0.26 ± 0.67 |
| non-qualified dividend income | −8.38 ± 0.66 | — | — | −6.21 ± 1.09 |
| long-term capital gains | −8.70 ± 0.92 | — | — | −0.03 ± 0.03 |
| tax-exempt interest income | +6.70 ± 1.88 | — | — | +8.52 ± 4.70 |
| taxable private pension income | +1.52 ± 0.71 | — | — | +5.46 ± 1.96 |
| taxable IRA distributions | +6.50 ± 0.63 | — | — | +17.97 ± 2.09 |
| charitable cash donations | +4.00 ± 1.52 | — | — | −3.38 ± 1.78 |
| charitable non-cash donations | −28.79 ± 3.58 | — | — | −37.80 ± 6.24 |
| real estate taxes | +1.72 ± 0.14 | — | — | +6.91 ± 0.34 |
| student loan interest | +9.36 ± 0.35 | — | — | +11.54 ± 1.17 |
| qualified tuition expenses | +14.26 ± 1.29 | — | — | +15.83 ± 1.54 |
| partnership income | +3.30 ± 1.27 | — | — | −4.89 ± 2.84 |
| SSTB self-employment income | +3.26 ± 1.23 | — | — | +32.77 ± 5.45 |
| health savings account deduction | +6.40 ± 0.38 | — | — | +8.82 ± 0.53 |

The production column reports adaptive over honest minus one on the chain's
ASEC-twin recipients (the PUF-half tax units whose ASEC twin the release kept:
13,834 of 48,755, 33% of the PUF units' weight, `harness/puf_chain_twin.py`).
This report uses the ratio there because percentages of the drawn total read
unreliably on this chain: the fold forest's own total on the twins differs
from the released draws by more than 25% on 22 of the 57 targets the chain
measures, so a drawn-total denominator measures something other than the
forest's scale. `all_targets.csv` carries both. Of the 57 chain targets the
public files report, 43 show a material held-out contrast, 40 a material
production contrast as a percent of the drawn total and 37 on the ratio.

These twin columns are a sensitivity on reconstructed features. Production
draws the chain in order and conditions each later target on the raw draws of
the earlier ones: `QRF.predict` appends each drawn target to the features
before the next draw (`populace/fit/qrf.py:1524-1530` at 6d06410bb). The twin
builder carries the PUF unit's released, finalized values of those targets
instead (`harness/capture_us_puf_chain.py:121-127`), and the ratio changes
only the denominator, so the conditioning values stay the twin builder's. The
twins also hold 28.4% of the PUF tax units and 32.9% of their weight, and each
target's released total on them runs from 11.8% (`farm_income`) to 105.5%
(`farm_operations_income`) of its released total on all PUF units, median
35.9% (`results/public/puf_twin_coverage.csv`). So these columns describe
adaptive over honest on the rebuilt twin subset; they don't establish what the
adaptive fill does to production's PUF-half draws.

PUF home mortgage interest (−58.17 ± 6.68) and first home mortgage interest
(−59.76 ± 4.81) say little about ordinary returns. The 6d06410bb donor table
carries the cohort that populace#516 screens out from cae8640 on: the
comment at `puf_support.py:62-72` (cae8640) counts 3,066 tax units at or
above $10 million of annual mortgage interest, holding $2.947 trillion of
mortgage-interest mass against $418 billion retained, and 6d06410bb has no
such screen. The captured donors' weighted mortgage-interest total, $3.37
trillion, matches that sum and comes to nine times the release's PUF-half
total. On those two targets the test measures how the two fills place a few
extreme donors, so this report counts them but does not read them as
describing ordinary returns.

### UK

| Donors | Target | Held-out contrast | Estimated smoothing | Estimated selection |
|---|---|---|---|---|
| LCFS 2024-25 | household furnishings | +17.43 ± 3.13 | — | — |
| LCFS 2024-25 | health | +15.57 ± 2.89 | — | — |
| LCFS 2024-25 | recreation | +12.00 ± 2.21 | — | — |
| LCFS 2024-25 | transport | +9.44 ± 1.98 | — | — |
| LCFS 2024-25 | education | +8.55 ± 2.57 | — | — |
| LCFS 2024-25 | restaurants and hotels | +6.41 ± 1.11 | — | — |
| LCFS 2024-25 | alcohol and tobacco | +4.92 ± 0.82 | — | — |
| LCFS 2024-25 | food | +0.25 ± 0.33 | — | — |
| LCFS 2024-25 | petrol share of road fuel (redraw) | +5.21 ± 0.49 | — | — |
| WAS round 8 | corporate wealth excluding ISAs | +11.18 ± 2.71 | — | — |
| WAS round 8 | other property value | −10.20 ± 3.79 | — | — |
| WAS round 8 | non-residential property value | −9.92 ± 2.55 | — | — |
| WAS round 8 | number of vehicles | −9.03 ± 0.34 | — | — |
| WAS round 8 | other residential property secured debt | −8.11 ± 2.83 | — | — |
| WAS round 8 | other mortgage | −7.48 ± 3.51 | — | — |
| WAS round 8 | cash ISA | +6.38 ± 1.50 | — | — |
| WAS round 8 | main residence value | +0.48 ± 0.38 | — | — |
| WAS round 8 | main residence mortgage | −0.15 ± 0.62 | — | — |
| WAS round 8 (LISA holders) | lifetime ISA balance | −9.15 ± 2.77 | — | — |
| ETB 1977-2024 | rail subsidy | +6.21 ± 1.75 | — | — |
| ETB 1977-2024 | DfE education spending | −2.75 ± 1.07 | — | — |
| NTS | local bus use band | +0.55 ± 0.22 | — | — |
| FRS (SPI housing shell) | mortgage capital repayment | +5.24 ± 2.82 | — | — |
| FRS (SPI housing shell) | structural insurance | +4.32 ± 1.20 | — | — |
| FRS (SPI housing shell) | mortgage interest repayment | +2.02 ± 0.31 | — | — |
| FRS (SPI-channel stage 2) | pension contributions via salary sacrifice | +13.78 ± 4.47 | — | — |
| FRS (SPI-channel stage 2) | child benefit | −6.21 ± 0.62 | — | — |
| FRS (SPI-channel stage 2) | PIP mobility | +5.02 ± 0.65 | — | — |
| FRS (SPI-channel stage 2) | tax-free savings income | +4.47 ± 1.82 | — | — |
| FRS (SPI-channel stage 2) | attendance allowance | +3.71 ± 0.46 | — | — |
| FRS (UC reporter redraw) | universal credit | +0.66 ± 1.07 | +0.42 ± 0.36 | +0.24 ± 1.16 |
| FRS (salary sacrifice) | pension contributions via salary sacrifice | −1.12 ± 2.71 | — | — |

Microcosm's build already pulls part of the LCFS over-draw back. It levels
each COICOP division whose drawn spend per frame household, over the donor's
mean per donor household, exceeds the divisions' shared ratio, down to that
ratio (`division_excess_factors`, `lcfs_consumption.py:2121-2177` at
67ee2ddf3; rule constants at lines 460-473; microcosm#1278). Other things
equal, an adaptive fill that runs above the honest fill raises that ratio.
The private addendum compares the build's ratios with these contrasts; the
two rank the divisions similarly.

### Which total lands nearer the held-out survey total

The held-out test gives one more reading: both fills' totals, and the forest
as production fits it, against the held-out donors' reported total. For each
target the harness averages each fold's signed gap, % of that total, and
takes the absolute value of that mean, so opposite fold errors cancel; the
table gives medians of those over targets with a measured, nonzero contrast
(a few rare targets run to thousands of percent). "Nearer" compares the same
absolute fold-mean gaps:

| Targets | n | Forest as fitted | Adaptive fill | Honest fill | Honest nearer than adaptive | Honest nearer than forest as fitted |
|---|---|---|---|---|---|---|
| US | 80 | 4.75 | 4.56 | 2.60 | 53 | 57 |
| US outside the PUF chain | 23 | 4.63 | 4.09 | 2.67 | 13 | 15 |
| PUF chain | 57 | 4.81 | 4.72 | 2.55 | 40 | 42 |
| UK | 81 | 3.24 | 4.02 | 1.16 | 61 | 57 |
| All | 161 | 3.73 | 4.30 | 1.82 | 114 | 114 |
| All, material targets only | 113 | 4.69 | 5.22 | 1.98 | 86 | 80 |

The forest as fitted is the fold's full forest, read as production reads it
(`own_vs_reported`); the two fills belong to the half-sample trees. Each fill
draws on half of a fold's donors and the forest as fitted on all of them.
Within a tree, though, an honest fill puts every donor of its half in the
leaves, while a tree's in-bag rows, after the weighted bootstrap and the
tree's own, hold only part of its half's distinct donors, so the honest fill
rests on more distinct donors per tree. Targets share donors
within a fit, so these counts describe the targets rather than independent
trials. These gaps compare totals and don't depend on how the contrast
splits.

## 5. Distribution: conditionals too narrow on held-out donors

The remedy and production-size runs (section 6) also score each
configuration's conditional quantiles on the held-out positive donors, at taus
0.1, 0.25, 0.5, 0.75 and 0.9 (`quantile_metrics` in `harness/selection.py`, on
the positive forest only). The forest production would grow puts too many
held-out donors below its low quantiles and too few below its high ones. The
half-size adaptive fill does so by more:

| LCFS division | Configuration | At or below q10 | q50 | At or below q90 |
|---|---|---|---|---|
| household furnishings | full-size forest, leaf 1 (production) | 12.5% | 51.1% | 87.5% |
|  | half-size forest, adaptive fill | 13.4% | 52.3% | 86.8% |
|  | half-size forest, honest fill | 10.4% | 49.9% | 89.3% |
|  | full-size forest, leaf 20 | 10.2% | 51.0% | 89.0% |
| health | full-size forest, leaf 1 (production) | 13.8% | 50.4% | 86.1% |
|  | half-size forest, adaptive fill | 14.6% | 51.1% | 85.5% |
|  | half-size forest, honest fill | 11.5% | 50.6% | 89.7% |
| housing, water and electricity | full-size forest, leaf 1 (production) | 14.6% | 51.6% | 85.6% |
|  | half-size forest, adaptive fill | 15.3% | 51.6% | 84.4% |
|  | half-size forest, honest fill | 10.5% | 49.8% | 89.1% |

Across the 16 LCFS targets the remedies ran on, the production-size forest's
mean absolute gap between each tau and its coverage runs 0.014-0.063. An
honest fill lowers it on 15 of the 16 (to 0.009-0.045), and
`min_samples_leaf=20` lowers it on 15 of the 16 (to 0.011-0.057). Across the
113 material targets the median change in that gap runs −0.035 for honest
fills, −0.032 for `min_samples_leaf=20`, −0.013 for `min_samples_leaf=5` and
−0.009 for `weighted_leaves=True` (section 6).

These coverages pool the held-out positive donors, so they show the
conditionals running too narrow on average over those donors; they don't
order the spread recipient by recipient, and they leave out the sign gates
and negative forests. Since `draw` reads each recipient's value at a uniform
quantile of its conditional, conditionals too narrow on average give an
imputed distribution with too little spread.

## 6. Remedies

Each remedy changes one thing: which rows fill the leaves (honest fills), how
large a leaf must be (`min_samples_leaf` 5 or 20) or how the pool averages
across trees (`weighted_leaves=True`, in
`RandomForestQuantileRegressor.predict`). The remedies ran on every target
whose contrast is material on held-out donors or on production recipients.

Two comparisons follow, because the same-tree test needs an honest half and
production does not. The first runs on the test's half-size forests, where
the test can measure adaptive minus honest under each remedy. The second scores the
same remedies against the forest production would grow on all of a fold's
donors, the comparison a change to production has to win.

The measures:

- Residual contrast: adaptive minus honest left under the remedy, % of the
  held-out reported total. It carries smoothing as the contrast does. An
  honest fill's residual is zero by definition, and its catchments don't
  depend on its donors' values, so it carries no selection.
- Pinball: percent change in mean pinball loss across the five taus, on
  held-out positive donors (negative fits better).
- Coverage gap: change in the mean absolute difference between each tau and
  the share of held-out positive donors at or below that quantile.
- Gap to the reported total: change in the absolute gap between the imputed
  and the held-out reported total, in points of that total.

### On the test's half-size forests

Each remedy reuses the test's halves, bootstrap rows and seeds, so it differs
from the half-size adaptive, `min_samples_leaf=1`, pooled forest only in what
it changes.

| Group | Remedy | Targets | Median contrast, absolute | Median residual contrast, absolute | Smaller residual | Residual below 0.5% | Median pinball change, % | Lower pinball | Pinball up more than 1% | Median coverage gap change | Median change in gap to reported total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| US outside the PUF chain | honest fills | 15 | 5.56 | 0.00 | 15 | 15 | −13.30 | 13 | 2 | −0.045 | −1.35 |
| US outside the PUF chain | min_samples_leaf=5 | 15 | 5.56 | 3.54 | 11 | 3 | −8.23 | 13 | 2 | −0.038 | −0.03 |
| US outside the PUF chain | min_samples_leaf=20 | 15 | 5.56 | 2.24 | 13 | 5 | −11.78 | 12 | 3 | −0.059 | −0.45 |
| US outside the PUF chain | weighted_leaves=True | 15 | 5.56 | 3.54 | 5 | 3 | −2.60 | 12 | 3 | −0.015 | +0.25 |
| PUF chain | honest fills | 43 | 6.50 | 0.00 | 43 | 43 | −10.26 | 43 | 0 | −0.041 | −3.34 |
| PUF chain | min_samples_leaf=5 | 43 | 6.50 | 4.02 | 31 | 3 | −5.14 | 35 | 8 | −0.018 | −0.22 |
| PUF chain | min_samples_leaf=20 | 43 | 6.50 | 1.87 | 33 | 9 | −7.00 | 32 | 10 | −0.046 | −1.71 |
| PUF chain | weighted_leaves=True | 43 | 6.50 | 5.57 | 21 | 6 | −0.88 | 31 | 5 | −0.006 | +0.07 |
| UK | honest fills | 55 | 4.51 | 0.00 | 55 | 55 | −11.40 | 54 | 1 | −0.024 | −2.78 |
| UK | min_samples_leaf=5 | 55 | 4.51 | 1.52 | 49 | 9 | −6.47 | 53 | 2 | −0.012 | −2.12 |
| UK | min_samples_leaf=20 | 55 | 4.51 | 0.86 | 52 | 21 | −8.41 | 53 | 2 | −0.022 | −2.74 |
| UK | weighted_leaves=True | 55 | 4.51 | 3.57 | 36 | 7 | −2.25 | 48 | 4 | −0.010 | +0.34 |
| All | honest fills | 113 | 4.92 | 0.00 | 113 | 113 | −11.04 | 110 | 3 | −0.034 | −3.04 |
| All | min_samples_leaf=5 | 113 | 4.92 | 2.39 | 91 | 15 | −6.34 | 101 | 12 | −0.017 | −1.35 |
| All | min_samples_leaf=20 | 113 | 4.92 | 1.06 | 98 | 35 | −8.41 | 97 | 15 | −0.035 | −1.94 |
| All | weighted_leaves=True | 113 | 4.92 | 4.34 | 62 | 16 | −1.93 | 91 | 12 | −0.010 | +0.32 |

An honest fill's residual is zero by definition. Honest fills lower pinball on
110 of the 113 material targets (median −11.0%) and bring the imputed total a
median 3.0 points nearer the held-out total. `min_samples_leaf=20` leaves a
median residual contrast of 1.06% of the total, below 0.5% on 35 targets,,
lowers pinball by a median 8.4% and narrows the gap to the held-out total by
1.9 points. `min_samples_leaf=5` leaves a median residual of 2.39% and lowers
pinball by 6.3%. `weighted_leaves=True` leaves a median residual of 4.34%,
lowers pinball by 1.9% and moves the imputed total a median 0.32 points
further from the held-out total. These gains compare each remedy with a
half-size adaptive forest; the next comparison uses the forest production
would grow.

### At production's forest size

`harness/run_full_size.sh` grows each fold's forest as production would: every
training donor of the sign, through the bootstrap rows and seed `_fit_forest`
would use, at `min_samples_leaf` 1, 5 and 20 (`full_size_forests` in
`harness/selection.py`). `harness/full_size.py` refuses a leaf-1 forest whose
total differs from the fold's own forest. Each row below compares a
configuration with that leaf-1 pooled forest, on the same held-out rows: the
leaf floors and `weighted_leaves=True` at full size, honest fills as the test
builds them, and the half-size adaptive fill for reference. The test's honest
fill grows trees on one half and fills them with the other, then swaps the
halves. Its totals average the two half-forests, and its quantile scores
score each half-forest's conditionals on their own and average the scores.
A shipped version of option 3 in section 9 needs a rule for combining the two
half-forests' conditionals into one per recipient, which this analysis didn't
specify or score.

The effect columns give the configuration's total against the full-size
forest's: as a percent of the held-out reported total on held-out donors, and
as total over total, minus one, on production recipients (the US only, in this
public version; medians of absolute values).

| Group | Configuration | Targets | Median effect on the total, held-out | Median effect on the total, production recipients | Median pinball change, % | Lower pinball | Pinball up more than 1% | Pinball up 100% or more | Median coverage gap change | Median change in gap to reported total |
|---|---|---|---|---|---|---|---|---|---|---|
| US outside the PUF chain | honest fills | 15 | 1.96 | 6.00 | −11.01 | 13 | 2 | 2 | −0.047 | −0.83 |
| US outside the PUF chain | min_samples_leaf=5 | 15 | 2.36 | 2.09 | −8.80 | 14 | 1 | 0 | −0.027 | −0.53 |
| US outside the PUF chain | min_samples_leaf=20 | 15 | 3.54 | 3.53 | −11.42 | 12 | 3 | 1 | −0.055 | −0.82 |
| US outside the PUF chain | weighted_leaves=True | 15 | 0.84 | 0.96 | −2.08 | 12 | 2 | 0 | −0.012 | +0.35 |
| US outside the PUF chain | half-size adaptive | 15 | 2.41 | 2.20 | +2.38 | 4 | 10 | 1 | +0.001 | −0.13 |
| PUF chain | honest fills | 43 | 5.12 | 8.14 | −8.19 | 36 | 6 | 0 | −0.040 | −2.90 |
| PUF chain | min_samples_leaf=5 | 43 | 2.43 | 2.97 | −5.58 | 35 | 8 | 0 | −0.016 | −0.36 |
| PUF chain | min_samples_leaf=20 | 43 | 2.86 | 5.38 | −8.32 | 32 | 10 | 2 | −0.041 | −1.87 |
| PUF chain | weighted_leaves=True | 43 | 3.77 | 4.31 | −1.32 | 29 | 6 | 0 | −0.008 | +0.53 |
| PUF chain | half-size adaptive | 43 | 0.93 | 1.96 | +2.14 | 7 | 32 | 0 | +0.001 | +0.10 |
| UK | honest fills | 55 | 3.10 | — | −8.15 | 53 | 2 | 1 | −0.022 | −1.31 |
| UK | min_samples_leaf=5 | 55 | 2.94 | — | −6.08 | 54 | 1 | 0 | −0.009 | −1.53 |
| UK | min_samples_leaf=20 | 55 | 3.53 | — | −8.25 | 53 | 2 | 1 | −0.014 | −1.45 |
| UK | weighted_leaves=True | 55 | 2.05 | — | −2.10 | 49 | 4 | 0 | −0.008 | +0.57 |
| UK | half-size adaptive | 55 | 1.72 | — | +3.28 | 4 | 46 | 1 | +0.002 | +0.46 |
| All | honest fills | 113 | 3.83 | — | −8.45 | 102 | 10 | 3 | −0.035 | −1.43 |
| All | min_samples_leaf=5 | 113 | 2.83 | — | −6.16 | 103 | 10 | 0 | −0.013 | −0.78 |
| All | min_samples_leaf=20 | 113 | 3.37 | — | −8.42 | 97 | 15 | 4 | −0.032 | −1.28 |
| All | weighted_leaves=True | 113 | 2.34 | — | −1.92 | 90 | 12 | 0 | −0.009 | +0.56 |
| All | half-size adaptive | 113 | 1.42 | — | +2.68 | 15 | 88 | 2 | +0.001 | +0.18 |

Against the forest production would grow, honest fills lower pinball by a
median 8.5% and `min_samples_leaf=20` by 8.4%, and the coverage gap falls by
0.035 and 0.032. Honest fills bring the imputed total a median 1.4 points
nearer the held-out total and `min_samples_leaf=20` 1.3 points. That
difference sits in the PUF chain (−2.9 against −1.9 points there, −0.8 against
−0.8 on the other US targets and −1.3 against −1.4 in the UK).
`min_samples_leaf=5` lowers pinball by 6.2% and brings the total 0.8 points
nearer. `weighted_leaves=True` lowers pinball by 1.9% and moves the total 0.6
points further away.

The half-size adaptive fill runs a median 2.7% worse on pinball than the
full-size forest, so the half-size table overstates the honest fill's gain on
pinball (−11.0% there, −8.5% here). The leaf floors hold about their half-size
gains (`min_samples_leaf=20` −8.4% there, −8.4% here; `min_samples_leaf=5`
−6.3% there, −6.2% here).

On US production recipients honest fills shift the imputed total by a median
6.0% outside the PUF chain and 8.1% on it, `min_samples_leaf=20` by 3.5% and
5.4%, and `min_samples_leaf=5` by 2.1% and 3.0%.

### Where remedies hurt

A remedy that lowers error at the median can still raise it on particular
targets. `TABLES.md` lists every public target where a configuration raises
mean pinball loss by more than 1% against the production-size forest (mean
over folds ± its standard error).

- Honest fills raise it on 10 of the 113 material targets, by 100% or more on
  3: traditional IRA contributions (+189.2 ± 98.2), Roth 401(k) contributions
  (+165.0 ± 19.6) and WAS other residential property secured debt (+152.9 ±
  40.4).
- `min_samples_leaf=5` raises it on 10 targets, none by 100%.
- `min_samples_leaf=20` raises it on 15 targets, by 100% or more on 4: WAS
  other residential property secured debt (+184.0 ± 34.9), PUF self-employment
  income (+179.2 ± 18.9), Roth 401(k) contributions (+136.4 ± 31.0) and PUF
  short-term capital gains (+103.8 ± 12.6).
- `weighted_leaves=True` raises it on 12 targets, by at most +18.3 ± 1.3 (the
  PUF chain's `farm_operations_income_would_be_qualified`).

Two patterns account for every increase of 100% or more. One is the PUF
chain's targets whose predictors equal them or, for dividends, their sum
(section 4), which hold 4 of the honest fills' 10 increases and 6 of the
leaf-20 floor's 15. On held-out donors those forests split on the value they
impute, so a small leaf reproduces a donor's value and a floor of 20 rows
cannot. Production recipients carry CPS values of those predictors, so a
held-out loss on these targets does not describe production's fit. The other
pattern is small donor pools: Roth 401(k) and traditional IRA contributions
and WAS other residential property secured debt have 603-673 positive training
donors. On those three even the half-size adaptive fill, which changes only
the sample size, runs 82-131% above the full-size forest.

### LCFS divisions in detail

On the test's half-size forests. Effect: the remedy's held-out total minus the
half-size adaptive forest's, % of the held-out reported total.

| LCFS division | Contrast | Honest fills: effect / pinball | Leaf 5: effect / residual / pinball | Leaf 20: effect / residual / pinball | `weighted_leaves=True`: effect / residual / pinball |
|---|---|---|---|---|---|
| furnishings | +17.4 | −17.4 / −13.5 | −10.0 / +7.3 / −8.7 | −14.8 / +2.5 / −12.6 | +2.3 / +16.9 / −2.5 |
| health | +15.6 | −15.6 / −16.0 | −10.4 / +5.3 / −11.9 | −12.7 / +3.2 / −15.5 | +5.6 / +21.0 / −1.6 |
| recreation | +12.0 | −12.0 / −9.0 | −6.4 / +4.9 / −6.0 | −9.2 / +1.7 / −8.0 | +2.3 / +10.1 / −2.4 |
| transport | +9.4 | −9.4 / −11.8 | −6.6 / +2.4 / −7.0 | −7.1 / +2.0 / −10.1 | +0.3 / +9.2 / −2.5 |
| education | +8.6 | −8.6 / −15.6 | −1.0 / +7.6 / −6.0 | −3.0 / +6.9 / −11.9 | +0.1 / +8.1 / −1.3 |
| restaurants and hotels | +6.4 | −6.4 / −7.3 | −4.7 / +1.2 / −4.5 | −5.2 / +0.9 / −6.4 | −0.8 / +5.9 / −2.7 |
| alcohol and tobacco | +4.9 | −4.9 / −11.4 | −2.8 / +2.0 / −7.2 | −3.4 / +1.1 / −10.9 | +1.6 / +5.5 / −2.1 |
| housing, water and electricity | +2.3 | −2.3 / −11.0 | −2.7 / −0.2 / −7.1 | −3.8 / −1.0 / −6.6 | −0.9 / +2.3 / −2.3 |

`results/public/TABLES.md` has every target the remedies covered.

## 7. Coverage of the production QRF outputs

`harness/coverage.py` classifies every output of every stage whose manifest
declares a QRF operation, plus the QRF fits the manifests do not list (the
SPI channel's FRS-only fill, the housing shell, the UC reporter redraw, the
LCFS road-fuel redraw and the chains' internal targets), and writes each
one's code reference and reason to `results/public/coverage.csv`.

| Status | UK | US |
|---|---|---|
| measured | 121 | 98 |
| measured under the fit's own name | 0 | 1 |
| derived from measured draws | 16 | 4 |
| binary target (a gate, no amount forest) | 1 | 2 |
| another method (carry, rule, classifier, rates) | 15 | 17 |
| declared but absent from the release | 2 | 19 |
| QRF target this analysis cannot rebuild | 0 | 2 |

The 219 measured targets split in these public files into 171 reported, 26
suppressed for resting on fewer than 50 nonzero training donors, and 22
trained on the SPI tape, which only the private addendum reports.

The two targets this analysis cannot rebuild are prior-year income fills.
`employment_income_last_year` is the first target of its stage's joint QRF;
PolicyEngine-US owns it by formula, so the stage drops it and the release
lacks it, and its donor is the adjacent-year ASEC join on a base pool the
release does not carry. `self_employment_income_last_year` is the second
target of the same joint QRF and conditions on the dropped first target.

## 8. Caveats

- **Half-size forests.** The test grows each forest on half a sign's training
  donors, so that an honest half remains; production grows on all of them,
  and a full-size forest leaves no honest half to measure the contrast with.
  `harness/scaling.py` thinned each fold's training donors to half and reran
  10 fits, 32 targets (`results/public/scaling.csv`). The contrast on the
  thinned donors ran at a median 0.75 of the contrast as run (quartiles 0.45
  to 1.05; above 1 on 9 of the 32). That compares quarter-size with half-size
  forests, and thinning refits the gates as well, so it moves sign
  probabilities along with forest size; it doesn't say how the contrast
  behaves at production's size. The 10 fits are the ones that ran fastest,
  not a random draw; the check leaves out the PUF chain; every UK row in it
  carries a material contrast; and single ratios carry wide errors (LCFS
  education runs 13.5 ± 13.3 on the thinned donors against 8.6 ± 2.6 as
  run). Section 6 scores every remedy against the forest production would
  grow on all of a fold's donors as well, so only the residual-contrast column
  stays at half size.
- **The split's estimate.** Estimated selection rests on a conditional mean
  fitted to held-out donors. Its error moves between estimated selection and
  estimated smoothing, so a target's split carries that error even where its
  contrast is precise; section 2's synthetic processes bound it only for
  processes like theirs. On those processes the fitted mean's smoothing missed the true smoothing by at
most 0.77 points on trees grown on the target.
- **Chained predictors.** A later target in a chain conditions on earlier
  ones. Held-out donors bring their own reported values of those columns;
  production recipients bring the chain's earlier draws. Held-out and
  production columns can differ for that reason as well as the PUF chain's
  own-value predictors (section 4).
- **Standard errors and the threshold.** The five folds' training sets
  overlap, so a fold standard error understates sampling uncertainty. The
  materiality threshold screens and does not test: with no contrast at all
  and independent folds, a five-fold mean clears one standard error about 37%
  of the time, before the 0.5% floor.
- **Before levelling and calibration.** These numbers measure what the
  forests impute. Later build steps (the LCFS division levelling in section 4
  and each build's calibration to administrative targets) act on those draws.
  This analysis did not measure how much of the contrast reaches the
  released weighted totals or what calibration does in response.
- **Quantile scores.** Pinball and coverage score the positive forest on
  held-out positive donors. They do not test the sign gates or negative
  forests, and they stop at the 90th percentile, so they do not score the top
  tail. A leaf floor keeps a tree from giving an extreme donor its own leaf,
  so section 9 sends the top shares to the shadow build. The harness reads
  each quantile as the lowest value whose cumulative share reaches tau
  (`quantile_metrics`), where production's `predict` interpolates between
  rows; integer bootstrap masses make exact ties common, and the harness
  resolves each one to the lower value.
- **Draws against means.** Section 2 measures the contrast in expected draws
  on 13 fits. The other fits report it in pooled means only.
- **Reconstructed fits.** The PUF-support numbers come from reconstructions
  (section 3), the ACS rent recipients from a rebuild that matches 54% of
  draws exactly, and the PUF chain's production column from ASEC twins that
  carry released values of earlier chain targets where production conditions
  on raw draws (section 4).
- **UK FRS-trained rebuilds** fit at the calibration artifact's initial
  weights, which equal the stage's weights only when no later stage changes
  household weights; the private addendum records each rebuild's check
  against the release's draws.
- **Single-donor folds.** The harness drops a target from a fold's fit when
  the fold leaves one donor of a sign (the gate cannot fit); later targets in
  that chain lose that nearly all-zero column as a feature in that fold.
- **Near-discrete leaves.** On targets with 2 to 32 distinct values production
  keeps a seeded subsample of 4,096 rows per leaf (`qrf.py:564-572`, 666-668).
  The harness pools every in-bag row. This analysis did not measure what the
  subsample changes.
- **Rare targets.** Held-out shares of the total swing widely when a fold
  holds a few dozen nonzero donors. The adaptive-over-honest ratio gives a
  steadier reading, and this public version suppresses every target below 50
  nonzero training donors.

## 9. Decision for Max

Whether Microcosm changes how its amount forests fill their leaves, and if
so, how. Options 2 and 3 move every QRF output, so each means rebuilding both
countries' data.

**Recommendation: try option 2 first, behind a setting that defaults to
today's value, and compare a shadow build of each country with the current
release before switching.** Against the forest production would grow,
`min_samples_leaf=20` matches honest fills on quantile fit (−8.4% against
−8.5% on pinball, −0.032 against −0.035 on the coverage gap) and changes one
argument, where honest fills need new fill code. Honest fills keep two
advantages: they carry no selection, since their catchments don't depend on
their donors' values, and they bring totals nearer the held-out totals (1.4
points at the median against 1.3, a difference that sits in the PUF chain).
Build them (option 3) if the shadow build shows the floor leaving a gap that
matters. A setting keeps either change reversible, and a shadow build changes
nothing that ships.

| Option | Change | Median contrast left, 113 material targets (half-size test) | Median pinball change | Median coverage gap change | Median change in gap to reported total | Targets with pinball up more than 1% | Up 100% or more |
|---|---|---|---|---|---|---|---|
| 1. Keep adaptive fills and document them | none | 4.92% of the total | none | none | none | none | none |
| 2. `min_samples_leaf=20` in `_fit_forest` | one argument (`qrf.py:675-683`) | 1.06% | −8.4% | −0.032 | −1.3 points | 15 | 4 |
| 3. Cross-fitted honest fills | new fill code | 0 by construction | −8.5% | −0.035 | −1.4 points | 10 | 3 |

Every column but the contrast column compares the option with the forest
production would grow on all of a fold's donors (section 6).

- **Option 1** needs no rebuild. The contrast stays material on 113 of the 171
  reported targets (estimated selection on 1 of the 6 with an estimate), and
  the conditionals stay too narrow on held-out donors (section 5). The LCFS
  division levelling (section 4) absorbs part of the over-draw on the
  divisions it cuts; this analysis did not audit the other chains' post-draw
  steps.
- **Option 2** changes one argument. At that floor 4 targets' pinball rises
  100% or more and 15 rise by more than 1% (section 6); the half-size test
  leaves a median 1.06% of the total as residual contrast, below 0.5% on 35
  targets. This analysis cannot measure that residual on a full-size forest,
  which leaves no honest half. The floor applies to every amount forest, the
  remedy runs covered only targets with a material contrast, and the quantile
  scores stop at the 90th percentile, so the shadow build has to check the
  other targets and the top shares.
- **Option 3** carries no selection, since an honest fill's catchments don't
  depend on its donors' values, and raises pinball on fewer targets (10
  against 15). It needs new code: quantile-forest's `predict` reads the rows
  `fit` stored from each tree's bootstrap (`_get_y_train_leaves`,
  `quantile_forest/_quantile_forest.py:299`), so an honest fill needs its own
  store or reader. It grows two half-size forests per sign, at a fit cost this
  analysis did not measure. It needs a rule for combining the two
  half-forests' conditionals (section 6), its fill needs integer masses or a
  reader that does not depend on mass scale (section 2), and recipients no
  honest donor reaches need a rule (they held at most 0.63% of a target's
  adaptive total on held-out donors and 1.27% on production recipients).

Flip conditions: if the shadow builds show the floor moving each country's
calibrated aggregates and decile shares by less than the build's own
seed-to-seed spread, keep adaptive fills and document them (option 1). If a
shadow build shows the floor leaving a gap that matters against the
administrative targets, or harming a target that matters, build honest fills
behind their own setting and shadow-build those (option 3).

This analysis did not measure what any option does to released weighted totals
after calibration, to microsimulation outputs or to fit time.

## 10. Reproducing

The harness needs the pinned inputs (survey files under each provider's
licence, and the release artefacts), which this repository does not carry.
The path constants at the top of each capture script say where each input
goes. With them in place:

```bash
# Rebuild each production fit's donors, settings and recipients
python harness/capture_us.py --stages <refresh stages> --out results/capture_us.json
python harness/capture_us_acs_rent.py --out results/capture_us_acs_rent.json
python harness/capture_us_puf_chain.py --out results/capture_us_puf_chain.json
python harness/capture_us_puf_support.py --out results/capture_us_puf_support.json
python harness/puf_chain_twin.py --out results/capture_us_puf_chain_twin.json
python harness/capture_uk_stages.py lcfs etb_vat etb_services spi_stage1 property_finance was_lisa
python harness/capture_uk_was.py --out results/capture_uk_was.json
python harness/capture_uk_release.py salary_sacrifice   # and the other FRS-trained fits

# The same-tree test on every fit, then the remedies where the contrast is material
python harness/run_selection.py scf_net_worth           # one fit; run_*_batch.sh run the rest
python harness/aggregate.py && python harness/remedies_plan.py
zsh harness/run_remedies.sh results/remedies_plan.txt

# The donor-halving check on one fit
python harness/run_selection.py acs_rent --train-fraction 0.5 --tag _halved

# The contrast's split into selection and smoothing, and the synthetic check
python harness/smoothing_plan.py && zsh harness/run_smoothing.sh results/smoothing_plan.txt
python harness/smoothing_check.py

# The known-answer check, the contrast in production's expected draws, and the
# remedies against production-size forests
python harness/known_answer.py
zsh harness/run_draw_means.sh
zsh harness/run_full_size.sh results/remedies_plan.txt

# Every table and public file
zsh harness/make_tables.sh
```

`harness/compare_runs.py` compares two runs number by number. All 33 remedy fits
ran on the final harness code. A rerun of the 32 that had finished before the
last change to the quantile pass (`harness/rerun_check.sh`) changed no total,
residual or catchment check; the tie handling moved quantile metrics only, and
no target's pinball change by more than 0.06 points.

`harness/vendor/` holds byte-identical copies of two helpers from
PolicyEngine/qrf-moop-shortfall at dba3940.
