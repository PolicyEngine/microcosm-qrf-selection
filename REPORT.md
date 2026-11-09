# Adaptive-split selection in Microcosm's QRF imputations

Status: 2026-11-09, awaiting independent review. Every measurement comes from
this repository's harness, on five folds per fit. The harness's later additions
left earlier numbers unchanged: two fits rerun on the final code match their
base runs number for number (`harness/compare_runs.py`).

This public version leaves out every number derived from the HMRC Survey of
Personal Incomes (SPI) public use tape and every UK production-recipient
number (the UK build frame holds the SPI-channel households), because the
SPI licence sends any analysis to HMRC ten working days before anyone
distributes or publishes it. It also suppresses every target whose forests
grew on fewer than 50 nonzero donors. A private addendum outside this
repository holds those numbers.

## Bottom line

- Microcosm's amount forests read their leaves with the same rows that chose
  the splits, and that moves the totals they impute. On held-out donors,
  filling the same trees with those rows (adaptive, as production does)
  instead of donors the trees never saw (honest) shifts the imputed total by
  more than its fold standard error and 0.5% of the total on 58 of the 83 US
  targets and 55 of the 88 UK targets this repository reports. Ten targets
  whose nonzero values all match show exactly zero.
- The sign differs by target. Selection inflates UK consumption (LCFS
  furnishings +17.4% of the held-out total, health +15.6%, recreation +12.0%,
  transport +9.4%) and US liquid assets and net worth (SCF net worth +6.9%,
  SIPP bank accounts +5.6%), and deflates UK property wealth (WAS
  non-residential property −9.9%, other property −10.2%) and UK vehicle counts
  (−9.0%).
- Honest fills usually land nearer the held-out totals. Against the held-out
  donors' reported totals, honest fills beat adaptive fills on 114
  of the 161 targets with nonzero selection. The median absolute gap falls
  from 4.3% (adaptive) to 1.8% (honest); the forests as production fits them
  land 3.7% away.
- On US production recipients selection runs at about the size it runs on
  held-out donors. Outside the PUF chain, the median absolute selection is
  5.3% of the drawn total against 5.6% of the held-out total, with the same
  sign on 21 of 23 targets (SCF net worth +11.1%, SIPP bank accounts +11.7%,
  SCF auto loan balances +6.3%, SIPP vehicles +5.5%, ACS rent +4.7%).
- The PUF chain shows why the two can differ. In the PUF donor table six of
  the chain's eight predictors equal chain targets (dividends as the sum of
  qualified and non-qualified), so on held-out donors those forests split on
  the value they impute; on production recipients those predictors hold CPS
  values instead (section 4). Taxable interest shows −7.7% on held-out donors
  and −0.06% on the chain's ASEC-twin recipients. Targets without such a
  predictor carry over: charitable non-cash donations show −28.8% on held-out
  donors and adaptive over honest minus one of −37.8% on the twins, taxable
  IRA distributions +6.5% and +18.0%.
- These totals compare conditional means, and production draws from quantile
  curves. On the 36 targets where this analysis computed both, selection in
  expected draws has the same sign as selection in means and stays within 0.43
  points of it (section 2).
- Adaptive fills also narrow each recipient's conditional distribution.
  Across the 16 LCFS targets the remedies cover, 12-20% of held-out donors fall
  at or below the forest's 10th percentile and 80-89% at or below its 90th;
  honest fills give 8-12% and 89-92%. Microcosm draws each value at a uniform
  random quantile of that conditional, so the narrowing compresses the imputed
  spread.
- On the 113 targets with material held-out selection, honest fills leave no
  selection by construction, lower mean pinball loss by a median 11% (lower on
  110 of the 113) and bring the imputed total a median 3.0 points nearer the
  held-out reported total. `min_samples_leaf=20` cuts the median selection from
  4.9% to 1.1% of the total and pinball by a median 8.4%, but it raises pinball
  by more than 1% on 15 targets and by 100% or more on five.
  `min_samples_leaf=5` leaves a median 2.4%, and `weighted_leaves=True` leaves
  4.3%.
- Changing Microcosm's fit is Max's call (decision filed with this report).
  This analysis changed no Microcosm or populace code.

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
   held fixed, isolates what the splits' sight of the filling rows' values
   does to the total.
4. Where the stage's production recipients exist, the same forests and fills
   give their total too, each recipient weighted by its weight times the gate
   probability.

The forests take the stage's own number of trees (100 on every stage, 32 on
the PUF chain, which `harness/capture_us_puf_chain.py` reads from
production's configuration).

This analysis calls a target material when its selection exceeds its fold
standard error and 0.5% of the total. For rare targets the held-out
reported total rests on few donors, so the table also gives adaptive over
honest minus one, a ratio of two model totals that small folds do not blow up.

A recipient that no honest donor reaches in any tree of a half-forest has no
honest estimate. It keeps its adaptive estimate (`complete_honest` in
`harness/selection.py`), so it adds nothing to the difference. Such
recipients held at most 0.63% of a reported target's adaptive total on
held-out donors and 1.27% on production recipients, both on SCF net worth, and
`honest_gap_share` reports that share for every target. An earlier version of
the harness dropped them from the honest total; correcting that moved no
reported target's selection by more than 0.23 points in any fold.

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
it. With the splits on the target it reads a positive difference even though no predictor
carries any signal, and quadrupling the donors does not shrink it.

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

| Fits | Targets | Same sign | Median absolute selection in means | In expected draws | Median shift | Largest shift | Median draw gap | Lowest draw gap |
|---|---|---|---|---|---|---|---|---|
| 13 | 36 | 36 | 5.11 | 5.12 | 0.06 | 0.43 | −0.50 | −1.49 |

Percentages of the held-out reported total. The shift is the absolute
difference between selection in expected draws and selection in means against
the same integer-mass fill. The draw gap is the fold's full forest's expected
draw over its pooled mean, minus one. Production's draw keeps the selection
the means show: same sign on all 36 targets, within 0.43 points on the
largest shift. Draws run a median 0.50% below the pooled mean, 1.49% on LCFS
health, a level effect the totals in this report leave out.

Moving the honest fill from survey-weight masses to integer masses also moves
selection in means, by a median 0.79 points in absolute terms, in both
directions (22 targets down, 14 up): one weighted resample of a half adds
noise. Across the 36 targets the median absolute selection stays put (5.06 at
survey-weight mass, 5.11 at integer mass). Single targets can move more: SCF
bank accounts goes from +5.77 ± 1.18 to −1.70, with a mass effect of
−7.47 ± 5.90 driven by one fold (−29.6). This report uses survey-weight masses
throughout because they carry no resampling noise.

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
`results/public/all_targets.csv` holds the numbers. Selection below is
adaptive minus honest, as a percent of the held-out reported total, mean over
5 folds ± its standard error; the production column uses the release's drawn
total as its denominator. Of the 171 targets this repository reports, 113
carry material held-out selection: 58 of 83 in the US (15 of 26 outside the
PUF chain, 43 of its 57) and 55 of 88 in the UK.

### US

| Fit | Target | Held-out | Production recipients |
|---|---|---|---|
| SCF 2022 | net worth | +6.86 ± 2.76 | +11.05 ± 2.67 |
| SCF 2022 | bank account assets | +5.77 ± 1.18 | +5.15 ± 2.22 |
| SCF 2022 | auto loan balance | +1.14 ± 0.38 | +6.26 ± 0.95 |
| SCF 2022 | auto loan interest | +1.35 ± 0.40 | +4.35 ± 1.22 |
| SIPP | bank account assets | +5.56 ± 1.13 | +11.67 ± 1.35 |
| SIPP | stock assets | −3.09 ± 2.45 | +10.46 ± 6.16 |
| SIPP | household vehicles value | +1.63 ± 0.81 | +5.48 ± 1.37 |
| ACS 2022 | rent | +2.03 ± 1.46 | +4.71 ± 0.89 |
| CPS ORG | hourly wage | +0.70 ± 0.05 | +0.54 ± 0.15 |
| CPS ASEC (PUF-support reconstruction) | childcare expenses | +15.19 ± 7.32 | +8.44 ± 5.32 |
| CPS ASEC (PUF-support reconstruction) | weeks unemployed | +11.75 ± 4.59 | +7.34 ± 4.66 |

The SCF, SIPP and ORG fits and recipients are production's own (section 3),
and so is the ACS rent fit (its recipients come from a rebuild), so these
numbers describe the release's fits. The PUF-support numbers describe the same
method on the same kind of training sample, not production's exact fit.
Across the 23 US targets with a production column outside the PUF chain,
selection on production recipients has the same sign as on held-out donors on
21 of them, and a Spearman rank correlation of 0.81 with it. `TABLES.md`
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

| Target | Held-out | Adaptive over honest − 1, ASEC-twin recipients |
|---|---|---|
| taxable interest income | −7.70 ± 0.80 | −0.06 ± 0.02 |
| qualified dividend income | −6.88 ± 1.41 | +0.26 ± 0.67 |
| non-qualified dividend income | −8.38 ± 0.66 | −6.21 ± 1.09 |
| long-term capital gains | −8.70 ± 0.92 | −0.03 ± 0.03 |
| tax-exempt interest income | +6.70 ± 1.88 | +8.52 ± 4.70 |
| taxable private pension income | +1.52 ± 0.71 | +5.46 ± 1.96 |
| taxable IRA distributions | +6.50 ± 0.63 | +17.97 ± 2.09 |
| charitable cash donations | +4.00 ± 1.52 | −3.38 ± 1.78 |
| charitable non-cash donations | −28.79 ± 3.58 | −37.80 ± 6.24 |
| real estate taxes | +1.72 ± 0.14 | +6.91 ± 0.34 |
| student loan interest | +9.36 ± 0.35 | +11.54 ± 1.17 |
| qualified tuition expenses | +14.26 ± 1.29 | +15.83 ± 1.54 |
| partnership income | +3.30 ± 1.27 | −4.89 ± 2.84 |
| SSTB self-employment income | +3.26 ± 1.23 | +32.77 ± 5.45 |
| health savings account deduction | +6.40 ± 0.38 | +8.82 ± 0.53 |

The production column reports adaptive over honest minus one on the chain's
ASEC-twin recipients (the PUF-half tax units whose ASEC twin the release kept:
13,834 of 48,755, 33% of the PUF units' weight, `harness/puf_chain_twin.py`).
This report uses the ratio there because percentages of the drawn total read
unreliably on this chain: the fold forest's own total on the twins differs
from the released draws by more than 25% on 22 of the 57 targets the chain
measures, so a drawn-total denominator measures something other than the
forest's scale. `all_targets.csv` carries both. Of the 57 chain targets the
public files report, 43 show material held-out selection, 40 show material
production selection as a percent of the drawn total and 37 on the ratio.

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
selection on ordinary returns.

### UK

| Donors | Target | Held-out |
|---|---|---|
| LCFS 2024-25 | household furnishings | +17.43 ± 3.13 |
| LCFS 2024-25 | health | +15.57 ± 2.89 |
| LCFS 2024-25 | recreation | +12.00 ± 2.21 |
| LCFS 2024-25 | transport | +9.44 ± 1.98 |
| LCFS 2024-25 | education | +8.55 ± 2.57 |
| LCFS 2024-25 | restaurants and hotels | +6.41 ± 1.11 |
| LCFS 2024-25 | alcohol and tobacco | +4.92 ± 0.82 |
| LCFS 2024-25 | food | +0.25 ± 0.33 |
| LCFS 2024-25 | petrol share of road fuel (redraw) | +5.21 ± 0.49 |
| WAS round 8 | corporate wealth excluding ISAs | +11.18 ± 2.71 |
| WAS round 8 | other property value | −10.20 ± 3.79 |
| WAS round 8 | non-residential property value | −9.92 ± 2.55 |
| WAS round 8 | number of vehicles | −9.03 ± 0.34 |
| WAS round 8 | other residential property secured debt | −8.11 ± 2.83 |
| WAS round 8 | other mortgage | −7.48 ± 3.51 |
| WAS round 8 | cash ISA | +6.38 ± 1.50 |
| WAS round 8 | main residence value | +0.48 ± 0.38 |
| WAS round 8 | main residence mortgage | −0.15 ± 0.62 |
| WAS round 8 (LISA holders) | lifetime ISA balance | −9.15 ± 2.77 |
| ETB 1977-2024 | rail subsidy | +6.21 ± 1.75 |
| ETB 1977-2024 | DfE education spending | −2.75 ± 1.07 |
| NTS | local bus use band | +0.55 ± 0.22 |
| FRS (SPI housing shell) | mortgage capital repayment | +5.24 ± 2.82 |
| FRS (SPI housing shell) | structural insurance | +4.32 ± 1.20 |
| FRS (SPI housing shell) | mortgage interest repayment | +2.02 ± 0.31 |
| FRS (SPI-channel stage 2) | pension contributions via salary sacrifice | +13.78 ± 4.47 |
| FRS (SPI-channel stage 2) | child benefit | −6.21 ± 0.62 |
| FRS (SPI-channel stage 2) | PIP mobility | +5.02 ± 0.65 |
| FRS (SPI-channel stage 2) | tax-free savings income | +4.47 ± 1.82 |
| FRS (SPI-channel stage 2) | attendance allowance | +3.71 ± 0.46 |
| FRS (UC reporter redraw) | universal credit | +0.66 ± 1.07 |
| FRS (salary sacrifice) | pension contributions via salary sacrifice | −1.12 ± 2.71 |

Microcosm's build already pulls part of the LCFS over-draw back. It levels
each COICOP division whose drawn spend per frame household, over the donor's
mean per donor household, exceeds the divisions' shared ratio, down to that
ratio (`division_excess_factors`, `lcfs_consumption.py:2121-2177` at
67ee2ddf3; rule constants at lines 460-473; microcosm#1278). Other things
equal, positive selection raises that ratio. The private addendum compares
the build's ratios with these selection numbers; the two rank the divisions
similarly.

### Which total lands nearer the held-out survey total

The held-out test gives one more reading: both fills' totals, and the forest
as production fits it, against the held-out donors' reported total. These
are absolute gaps, % of that total, medians over targets with a measured,
nonzero selection (a few rare targets run to thousands of percent):

| Targets | n | Forest as fitted | Adaptive fill | Honest fill | Honest nearer than adaptive | Honest nearer than forest as fitted |
|---|---|---|---|---|---|---|
| US | 80 | 4.75 | 4.56 | 2.60 | 53 | 57 |
| US outside the PUF chain | 23 | 4.63 | 4.09 | 2.67 | 13 | 15 |
| PUF chain | 57 | 4.81 | 4.72 | 2.55 | 40 | 42 |
| UK | 81 | 3.24 | 4.02 | 1.16 | 61 | 57 |
| All | 161 | 3.73 | 4.30 | 1.82 | 114 | 114 |
| All, material targets only | 113 | 4.69 | 5.22 | 1.98 | 86 | 80 |

The forest as fitted is the fold's full forest, read as production reads it
(`own_vs_reported`); the two fills belong to the half-sample trees. Both fills
use half of a fold's donors and the forest as fitted all of them, so the
honest fill's lower median gap comes despite its smaller sample. Targets
share donors within a fit, so these counts describe the targets rather than
independent trials.

## 5. Distribution: adaptive fills narrow the conditionals

The remedy runs also score each configuration's conditional quantiles on the
held-out positive donors, at taus 0.1, 0.25, 0.5, 0.75 and 0.9
(`quantile_metrics` in `harness/selection.py`, on the positive forest only).
Production's configuration puts too many held-out donors below its low
quantiles and too few below its high ones:

| LCFS division | Fill | At or below q10 | q50 | At or below q90 |
|---|---|---|---|---|
| household furnishings | adaptive, leaf 1 (production) | 13.4% | 52.3% | 86.8% |
|  | honest, leaf 1 | 10.4% | 49.9% | 89.3% |
|  | adaptive, leaf 20 | 10.4% | 51.7% | 88.7% |
| health | adaptive, leaf 1 (production) | 14.6% | 51.1% | 85.5% |
|  | honest, leaf 1 | 11.5% | 50.6% | 89.7% |
| housing, water and electricity | adaptive, leaf 1 (production) | 15.3% | 51.6% | 84.4% |
|  | honest, leaf 1 | 10.5% | 49.8% | 89.1% |

Across the 16 LCFS targets the remedies ran on, production's mean absolute
gap between each tau and its coverage runs 0.016-0.077. Honest fills lower it
on 15 of the 16 (to 0.009-0.045; domestic energy rises from 0.016 to 0.019),
and `min_samples_leaf=20` lowers it on all 16 (to 0.012-0.053). Across
the 113 material targets the median change in that gap runs −0.034 for honest
fills, −0.035 for `min_samples_leaf=20`, −0.017 for `min_samples_leaf=5` and
−0.010 for `weighted_leaves=True` (section 6).

Since `draw` reads each recipient's value at a uniform quantile of that
conditional, a conditional too narrow at both ends gives an imputed
distribution with too little spread around each recipient's centre.

## 6. Remedies

Each remedy reuses the selection test's halves, bootstrap rows and seeds, so
it differs from production's configuration only in what it changes: which
rows fill the leaves (honest fills), how large a leaf must be
(`min_samples_leaf` 5 or 20) or how the pool averages across trees
(`weighted_leaves=True`, in `RandomForestQuantileRegressor.predict`). The
remedies ran on every target whose selection is material on held-out donors
or on production recipients.

Each measure below compares a remedy with production's configuration
(adaptive, `min_samples_leaf=1`, pooled leaves).

- Residual: adaptive minus honest left under the remedy, % of the held-out
  reported total. Honest fills leave none by construction.
- Pinball: percent change in mean pinball loss across the five taus, on
  held-out positive donors (negative fits better).
- Coverage gap: change in the mean absolute difference between each tau and
  the share of held-out positive donors at or below that quantile.
- Gap to the reported total: change in the absolute gap between the imputed
  and the held-out reported total, in points of that total.

| Group | Remedy | Targets | Median selection, absolute | Median residual, absolute | Smaller residual | Residual below 0.5% | Median pinball change, % | Lower pinball | Pinball up more than 1% | Median coverage gap change | Median change in gap to reported total |
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

Honest fills leave no residual by construction, lower pinball on 110 of the
113 material targets (median −11.0%) and bring the imputed total a median 3.0
points nearer the held-out total. `min_samples_leaf=20` removes most of the
selection (median residual 1.06% of the total, below 0.5% on 35 targets), lowers
pinball by a median 8.4% and narrows the gap to the held-out total by 1.9
points. `min_samples_leaf=5` leaves a median residual of 2.39% and lowers
pinball by 6.3%. `weighted_leaves=True` leaves a median residual of 4.34%,
lowers pinball by 1.9% and moves the imputed total a median 0.32 points further
from the held-out total. Every remedy here ran on half-size forests. Section 8
says what that means for the leaf floors.

### Where remedies hurt

A remedy can lower a target's error in general and still raise it elsewhere.
`TABLES.md` lists every public target where a remedy raises mean pinball loss
by more than 1% (mean over folds ± its standard error).

- Honest fills raise it on 3 of the 113 material targets: traditional IRA
  contributions (+51.1 ± 21.5), WAS other residential property secured debt
  (+25.4 ± 12.3) and Roth 401(k) contributions (+16.6 ± 9.8).
- `min_samples_leaf=5` raises it on 12 targets and `min_samples_leaf=20` on
  15. At 20 rows five targets rise by 100% or more: PUF self-employment income
  (+159.2 ± 6.9), Roth 401(k) contributions (+130.3 ± 16.6), traditional IRA
  contributions (+108.7 ± 54.2), PUF short-term capital gains (+102.5 ± 7.8)
  and WAS other residential property secured debt (+100.4 ± 20.4).
- `weighted_leaves=True` raises it on 12 targets, by at most +20.1 ± 0.7 (the
  PUF chain's `farm_operations_income_would_be_qualified`).

Two patterns account for most of the large increases. Six of the leaf-20 cases
sit on PUF-chain targets whose predictors equal them or, for dividends, their
sum (section 4). On held-out donors those forests split on that value, so a
small leaf reproduces a donor's value and a floor of 20 rows cannot; production
recipients carry CPS values of those predictors, so held-out
pinball says nothing about those recipients. The retirement contributions
bunch: Roth 401(k) contributions take a median 192 distinct values among 603
positive training donors, and traditional IRA contributions 205 among 620.

### LCFS divisions in detail

Effect: the remedy's held-out total minus production's, % of the held-out
reported total.

| LCFS division | Selection | Honest fills: effect / pinball | Leaf 5: effect / residual / pinball | Leaf 20: effect / residual / pinball | `weighted_leaves=True`: effect / residual / pinball |
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
| binary target, selection zero by construction | 1 | 2 |
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
  donors, so that an honest half remains; production grows on all of them.
  `harness/scaling.py` thinned each fold's training donors to half and reran
  10 fits, 32 targets (`results/public/scaling.csv`). Selection on the
  thinned donors ran at a median 0.75 of selection as run (quartiles 0.45 to
  1.05; above 1 on 9 of the 32). So selection did not grow as these forests
  shrank, and the half-size forests show no sign of overstating what larger
  forests carry. The check has limits: the 10 fits are the ones that ran
  fastest, not a random draw; it leaves out the PUF chain; every UK row in it
  carries material selection; and single ratios carry wide errors (LCFS
  education runs 13.5 ± 13.3 on the thinned donors against 8.6 ± 2.6 as run).
  The synthetic check in section 2 points the same way: quadrupling the donors
  left the difference in place (+3.5 ± 0.7 at 1,200 donors, +4.5 ± 0.4 at
  4,800). A floor of 20 rows per leaf takes twice the share of the sample on
  these forests that it takes on production's, so the leaf-floor results may
  not carry over at the same size.
- **Chained predictors.** A later target in a chain conditions on earlier
  ones. Held-out donors bring their own reported values of those columns;
  production recipients bring the chain's earlier draws. Held-out and
  production columns can differ for that reason as well as the PUF chain's
  own-value predictors (section 4).
- **Standard errors and the threshold.** The five folds' training sets
  overlap, so a fold standard error understates sampling uncertainty. The
  materiality threshold screens and does not test: with no selection at all
  and independent folds, a five-fold mean clears one standard error about 37%
  of the time, before the 0.5% floor.
- **Before levelling and calibration.** These numbers measure what the
  forests impute. Later build steps (the LCFS division levelling in section 4
  and each build's calibration to administrative targets) act on those draws.
  This analysis did not measure how much selection reaches the released
  weighted totals or what calibration does in response.
- **Quantile scores.** Pinball and coverage score the positive forest on
  held-out positive donors. They do not test the sign gates or negative
  forests. The harness reads each quantile as the lowest value whose
  cumulative share reaches tau (`quantile_metrics`), where production's
  `predict` interpolates between rows; integer bootstrap masses make exact
  ties common, and the harness resolves each one to the lower value.
- **Draws against means.** Section 2 measures selection in expected draws on
  13 fits. The other fits report selection in pooled means only.
- **Reconstructed fits.** The PUF-support numbers come from reconstructions
  (section 3), and the ACS rent recipients from a rebuild that matches 54% of
  draws exactly.
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

**Recommendation: build cross-fitted honest fills behind an option that
defaults off, and compare them in a shadow build of each country against the
current release before switching.** Option 2 is the fallback if the fill code
proves too costly. A flag keeps the change reversible, and a shadow build
changes nothing that ships.

| Option | Change | Median selection left, 113 material targets | Median pinball change | Median change in gap to reported total | Targets with pinball up more than 1% |
|---|---|---|---|---|---|
| 1. Keep adaptive fills and document them | none | 4.92% of the total | none | none | none |
| 2. `min_samples_leaf=20` in `_fit_forest` | one argument (`qrf.py:675-683`) | 1.06% | −8.4% | −1.9 points | 15 |
| 3. Cross-fitted honest fills | new fill code | 0 by construction | −11.0% | −3.0 points | 3 |

- **Option 1** needs no rebuild. Selection stays on 113 of the 171 reported
  targets, and each recipient's conditional stays too narrow (section 5).
  The LCFS division levelling (section 4) absorbs part of the selection on the
  divisions it cuts; this analysis did not audit the other chains' post-draw
  steps.
- **Option 2** changes one argument. Five targets' pinball rises 100% or more
  at that floor (section 6). A floor of 20 rows takes twice the share of the
  sample on the half-size forests tested that it takes on production's, so its
  residual and its hazards may differ in production.
- **Option 3** gives the largest cut in pinball and in the gap to the reported
  total and raises pinball on the fewest targets. It needs new code:
  quantile-forest's `predict` reads the rows `fit` stored from each tree's
  bootstrap (`_get_y_train_leaves`, `quantile_forest/_quantile_forest.py:299`),
  so an honest fill needs its own store or reader. It grows two half-size
  forests per sign, at a fit cost this analysis did not measure. Its fill needs
  integer masses or a reader that does not depend on mass scale (section 2),
  and recipients no honest donor reaches need a rule (they held at most 0.63%
  of a target's adaptive total on held-out donors and 1.27% on production
  recipients).

Flip conditions: if the shadow builds show honest fills moving each country's
calibrated aggregates and decile shares by less than the build's own
seed-to-seed spread, keep adaptive fills and document them (option 1). If the
honest fill's code or fit time proves too costly, take option 2 with its
hazards on the record.

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

# The same-tree test on every fit, then the remedies where selection is material
python harness/run_selection.py scf_net_worth           # one fit; run_*_batch.sh run the rest
python harness/aggregate.py && python harness/remedies_plan.py
zsh harness/run_remedies.sh results/remedies_plan.txt

# The donor-halving check on one fit
python harness/run_selection.py acs_rent --train-fraction 0.5 --tag _halved

# The known-answer check, and selection in production's expected draws
python harness/known_answer.py
zsh harness/run_draw_means.sh

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
