# Adaptive-split selection in Microcosm's QRF imputations

Status: draft of 2026-11-09, before independent review. Runs in progress:
the 44 PUF tax-detail targets outside the 20 largest, the PUF chain's
production side on ASEC-twin recipients, the NTS bus-use band, and remedies
beyond the LCFS chain. The harness also changed on 2026-11-09: a recipient no
honest donor reaches in any tree used to fall out of the honest total, which
inflated adaptive minus honest. It now keeps its adaptive estimate, and every
fit with such a recipient (the PUF chain, the SCF fits and a few others) is
running again. The numbers below for those fits predate the fix.

This public version leaves out every number derived from the HMRC Survey of
Personal Incomes (SPI) public use tape and every UK production-recipient
number (the UK build frame holds the SPI-channel households), because the
SPI licence sends any analysis to HMRC ten working days before anyone
distributes or publishes it. It also suppresses every target whose forests
grew on fewer than 50 nonzero donors. A private addendum outside this
repository holds those numbers.

## Bottom line

- Microcosm's amount forests read their leaves with the same rows that chose
  the splits. On held-out donors, filling the same trees with those rows
  (adaptive, as production does) instead of donors the trees never saw
  (honest) moves the imputed total by more than its fold standard error and
  0.5% of the total on 32 of the 46 US targets and 54 of the 87 UK targets
  this repository reports; constant-amount targets, whose nonzero values all
  match, show exactly zero.
- The sign differs by target. Selection inflates UK consumption (LCFS
  furnishings +17.4%, health +15.6%, recreation +12.0%, transport +9.4% of
  the held-out total) and US liquid assets (SCF bank accounts +5.8%, SIPP bank
  accounts +5.6%, SCF net worth +6.9%), and deflates PUF capital income
  (interest −7.7%, qualified dividends −6.9%, long-term gains −8.7%) and UK
  property wealth (WAS non-residential property −9.9%, other property −10.2%).
- On the US production recipients, where the stage's own recipients exist,
  selection often runs larger than on held-out donors: SCF net worth +11.1%,
  SIPP bank accounts +11.7%, ACS rent +4.7%, SCF auto loans +6.3%.
- Adaptive fills also narrow each recipient's conditional distribution. On
  the LCFS divisions, 13-15% of held-out donors fall at or below the forest's
  10th percentile and 84-87% at or below its 90th; honest fills put them back
  at 10-12% and 89-90%. Microcosm draws each value at a uniform random
  quantile of that conditional, so the narrowing compresses the imputed
  spread.
- Of the four remedies the brief named, honest fills remove selection by
  construction and cut held-out pinball loss by 4-16% on the LCFS chain;
  `min_samples_leaf=20` leaves −1% to +7% of the total and cuts pinball loss
  by 4-16%; `min_samples_leaf=5` leaves about 40% of the selection on the
  median division; `weighted_leaves=True` leaves it in place (it rises) and
  barely moves the fit.
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
  probability proportional to weight (`_weighted_bootstrap`, 189-219,
  `rng.choice(len(x), size=len(x), replace=True, p=weights / total)`), then
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

The brief's threshold calls a target material when its selection exceeds its
fold standard error and 0.5% of the total. For rare targets the held-out
reported total rests on few donors, so the table also gives adaptive over
honest minus one, a ratio of two model totals that small folds do not blow up.

The harness grows its forests with `max_samples_leaf=1` because it reads
leaves through `apply` and bootstrap counts, never through stored samples;
the stored-sample setting changes no split
(`test_stored_leaf_samples_leave_trees_unchanged`), and on near-discrete
targets `None` padded the stored samples to tens of gigabytes. One check per
sign forest still calls quantile-forest's own `predict` with every sample
stored and matches the harness's pooled total. A target whose training fold
holds one donor of a sign cannot pass the gate's stratified early-stopping
split (HistGradientBoosting turns early stopping on above 10,000 rows); the
harness drops that target from that fold's fit and records why. All 9 harness
tests pass (`pytest harness/tests`).

## 3. Reproduction

Before testing a fit, the harness reran the stage's own code on the pinned
inputs and checked the result against the release.

| Fits | Check | Result |
|---|---|---|
| US refresh stages (SCF net worth, financial assets, auto loans; SIPP assets, vehicles, tips; ORG wages; SSI criteria; Head Start; voluntary filing) | replay the stage on the release's own tables and compare every value | every value equal to the last bit; weighted totals equal |
| US ACS rent (base pool) | redraw on recipients rebuilt from the release | 54% of draws equal; total 2.9% below the release's |
| US PUF tax-detail chain (base pool) | donor table from the pinned PUF through 6d06410bb's own code; target order against `PRIMARY_QRF_TARGET_ORDER_SHA256` | donor table and fit settings are production's; production recipients rebuilt from ASEC twins (13,834 of 48,755 PUF units, 33% of their weight) |
| US PUF-support fills (base pool) | rerun each handler at 6d06410bb on the release's ASEC persons | reconstructions: the base pool that trained production's fills no longer exists, so the training pool differs; totals land within 10-35% of the release's on 11 of 17 targets and far off on the retirement distributions, disability benefits, child support paid and weeks unemployed |
| UK survey-donor stages (LCFS, ETB, WAS, LISA) | rebuild each donor with the stage's own reader, cleaning, uprating and seed from the pinned files | training rows match the build record wherever it records them (LISA 118; LCFS road-fuel redraw 2,819 against the record's rounded 2,800) |
| UK FRS-trained stages (SPI housing shell, SPI-channel stage 2, UC reporter redraw, salary sacrifice) | rebuild training rows from the release, fit at the calibration artifact's initial weights | training rows match the build's receipts (salary sacrifice 4,970; housing shell 16,288 households; stage 2's 8,556 positive rows; UC 4,286 benefit units against the receipt's 4,287); the recipient-side checks run on SPI-channel rows, so the private addendum holds them |

`results/public/TABLES.md` lists each US check.

## 4. Results

`results/public/TABLES.md` lists every measured target; `results/public/all_targets.csv`
holds the numbers. Selection below is adaptive minus honest, as a percent of
the held-out reported total, mean over 5 folds ± its standard error.

### US

| Fit | Target | Held-out | Production recipients |
|---|---|---|---|
| SCF 2022 | net worth | +6.86 ± 2.76 | +11.05 ± 2.67 |
| SCF 2022 | bank account assets | +5.78 ± 1.19 | +5.17 ± 2.24 |
| SCF 2022 | auto loan balance | +1.14 ± 0.38 | +6.26 ± 0.95 |
| SCF 2022 | auto loan interest | +1.35 ± 0.40 | +4.35 ± 1.22 |
| SIPP | bank account assets | +5.56 ± 1.13 | +11.67 ± 1.35 |
| SIPP | stock assets | −3.09 ± 2.45 | +10.46 ± 6.16 |
| SIPP | household vehicles value | +1.63 ± 0.81 | +5.48 ± 1.37 |
| ACS 2022 | rent | +2.03 ± 1.46 | +4.71 ± 0.89 |
| CPS ORG | hourly wage | +0.70 ± 0.05 | +0.54 ± 0.15 |
| IRS PUF | taxable interest income | −7.70 ± 0.80 | pending (twin recipients) |
| IRS PUF | qualified dividend income | −6.88 ± 1.41 | pending |
| IRS PUF | long-term capital gains | −8.70 ± 0.92 | pending |
| IRS PUF | taxable IRA distributions | +6.54 ± 0.63 | pending |
| IRS PUF | charitable cash donations | +4.05 ± 1.52 | pending |
| IRS PUF | partnership income | +3.30 ± 1.27 | pending |
| IRS PUF | real estate taxes | +1.72 ± 0.14 | pending |
| CPS ASEC (PUF-support reconstruction) | childcare expenses | +15.19 ± 7.32 | +8.44 ± 5.32 |
| CPS ASEC (PUF-support reconstruction) | weeks unemployed | +11.75 ± 4.59 | +7.34 ± 4.66 |

The SCF, SIPP and ORG fits and recipients are production's own (section 3),
and so is the ACS rent fit (its recipients come from a rebuild), so these
numbers describe the release's fits. The PUF-support numbers describe the same
method on the same kind of training sample, not production's exact fit.

PUF home mortgage interest (−58.17 ± 6.68) and first home mortgage interest
(−59.76 ± 4.81) say little about ordinary returns. The 6d06410bb donor table
carries the cohort that populace#516 screens out from cae8640 on: the
comment at `puf_support.py:62-74` (cae8640) counts 3,066 tax units at or
above $10 million of annual mortgage interest, holding $2.947 trillion of
mortgage-interest mass against $418 billion retained, and 6d06410bb has no
such screen. The captured donors' weighted mortgage-interest total, $3.37
trillion, matches that sum and comes to nine times the release's PUF-half
total. On those two targets the test measures how the two fills place a few
extreme donors, so this report counts them but does not read them as
selection on ordinary returns.

The PUF chain's production numbers in the CSV come from recipients whose
predictors hold the chain's own draws, which makes them circular; the run on
ASEC-twin recipients (`harness/puf_chain_twin.py`) replaces them.

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
| WAS round 8 | cash ISA | +6.38 ± 1.50 |
| WAS round 8 | main residence value | +0.48 ± 0.38 |
| WAS round 8 | main residence mortgage | −0.15 ± 0.62 |
| WAS round 8 (LISA holders) | lifetime ISA balance | −9.15 ± 2.77 |
| ETB 1977-2024 | rail subsidy | +6.21 ± 1.75 |
| ETB 1977-2024 | DfE education spending | −2.75 ± 1.07 |
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
ratio (`division_excess_factors`, `lcfs_consumption.py:2124-2175` at
67ee2ddf3; rule constants at lines 460-473; microcosm#1278). Other things
equal, positive selection raises that ratio. The private addendum compares
the build's ratios with these selection numbers; the two rank the divisions
similarly.

## 5. Distribution: adaptive fills narrow the conditionals

The remedy runs also score each configuration's conditional quantiles on the
held-out positive donors (taus 0.1, 0.25, 0.5, 0.75, 0.9). Production's
configuration puts too many held-out donors below its low quantiles and too
few below its high ones:

| LCFS division | Fill | At or below q10 | q50 | At or below q90 |
|---|---|---|---|---|
| household furnishings | adaptive, leaf 1 (production) | 13.4% | 52.3% | 86.8% |
| | honest, leaf 1 | 10.4% | 49.9% | 89.3% |
| | adaptive, leaf 20 | 10.4% | 51.7% | 88.7% |
| health | adaptive, leaf 1 (production) | 14.6% | 51.1% | 85.5% |
| | honest, leaf 1 | 11.5% | 50.6% | 89.7% |
| housing, water and electricity | adaptive, leaf 1 (production) | 15.3% | 51.6% | 84.4% |
| | honest, leaf 1 | 10.5% | 49.8% | 89.1% |

Across the 16 LCFS targets the remedies ran on, production's mean absolute
gap between each tau and its coverage runs 0.016-0.077. Honest fills lower it
on 15 of the 16 (to 0.009-0.045; domestic energy rises from 0.016 to 0.019),
and `min_samples_leaf=20` lowers it on all 16 (to 0.012-0.053). Since `draw`
reads each recipient's value at a uniform quantile of that conditional, a
conditional too narrow at both ends gives an imputed distribution with too
little spread around each recipient's centre.

## 6. Remedies

Each remedy reuses the same halves, bootstrap rows and seeds, so it differs
from production's configuration only in what it changes. Effect: the
remedy's held-out total minus production's, % of the held-out reported
total. Residual: adaptive minus honest left under the remedy. Pinball: change
in mean pinball loss against production's configuration (negative fits
better).

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

`results/public/TABLES.md` has all 16 LCFS targets the remedies covered.
What the table shows:

- Honest fills take selection to zero by construction and fit the held-out
  quantiles better on every LCFS target (pinball −4% to −16%). In the test
  both fills use half the sign's donors; a production version would grow on
  one half and fill with the other, or cross-fit two forests so every donor
  fills leaves it did not shape.
- `min_samples_leaf=20` removes most selection (residual −1.0% to +6.9%) and
  fits nearly as well as honest fills. In production it is one argument to
  `RandomForestQuantileRegressor` in `_fit_forest`. The test's forests grow on
  half the donors, so the same leaf floor smooths production's full-size
  forests somewhat less.
- `min_samples_leaf=5` leaves between −8% and 89% of the selection (about 40% on the
  median division) and improves pinball loss by 2-12%.
- `weighted_leaves=True` (per-tree leaf weights) leaves selection in place:
  the residual rises above production's on most divisions, and pinball loss
  moves by −3% to +1%.

Remedy runs for the other material fits (WAS, SCF, SIPP, ACS, ETB, the PUF
chain, the FRS-trained stages) wait on the runs above.

## 7. Coverage of the production QRF outputs

`harness/coverage.py` classifies every output of every stage whose manifest
declares a QRF operation, plus the QRF fits the manifests do not list (the
SPI channel's FRS-only fill, the housing shell, the UC reporter redraw, the
LCFS road-fuel redraw and the chains' internal targets), and writes each
one's code reference to `results/public/coverage.csv`.

| Status | UK | US |
|---|---|---|
| measured | 120 | 54 |
| measured under the fit's own name | 0 | 1 |
| derived from measured draws | 16 | 4 |
| binary target, selection zero by construction | 1 | 2 |
| another method (carry, rule, classifier, rates) | 15 | 17 |
| declared but absent from the release | 2 | 19 |
| QRF target this analysis cannot rebuild | 0 | 2 |
| run pending | 1 | 0 |
| PUF chain targets awaiting the full run | 0 | 44 |

The UK pending output is the NTS bus-use band. The two US targets this
analysis cannot rebuild are the prior-year income fills: the base pool they
trained on (337,704 households at 6d06410bb) no longer exists, and
`capture_us_puf_support.py` marks them `UNRECONSTRUCTABLE`.

## 8. Caveats

- The test measures selection on forests grown on half a fold's donors (to
  leave an honest half). Production grows on all donors. The sign and rough
  size carry over; the exact percentage need not.
- The PUF-support numbers come from reconstructions (section 3), and the ACS
  rent recipients from a rebuild that matches 54% of draws exactly.
- The UK FRS-trained rebuilds fit at the calibration artifact's initial
  weights, which equal the stage's weights only when no later stage changes
  household weights; each rebuild checks itself against the release's draws.
- The harness drops a target from a fold's fit when the fold leaves one
  donor of a sign (the gate cannot fit); later targets in that chain lose
  that nearly all-zero column as a feature in that fold.
- On near-discrete targets production keeps a 4,096-sample subsample per
  leaf where the harness pools every in-bag row; per the code's own
  docstring that changes leaf quantiles only by multinomial noise.
- Rare targets: held-out shares of the total swing widely when a fold holds
  a few dozen nonzero donors; the adaptive-over-honest ratio and the
  production column give the steadier reading, and this public version
  suppresses every target below 50 nonzero training donors.

## 9. Decision for Max

Whether Microcosm changes how its amount forests fill their leaves, and if
so, how. The options this evidence supports:

1. Keep adaptive fills and document the selection (no code change). The LCFS
   division levelling already absorbs part of it; nothing absorbs it on the
   other chains.
2. Set `min_samples_leaf=20` in `_fit_forest`: a one-argument change that
   removed most selection and improved the held-out fit on every LCFS target
   tested. It moves every QRF output, so it means a rebuild of both
   countries' data.
3. Honest (cross-fitted) fills: no selection by construction and the best
   held-out fit, at the cost of new leaf-filling code in `qrf.py` and the
   same rebuild.

Remedy runs on the other chains, the PUF chain's full results and the
independent review come before any of these lands.

## 10. Reproducing

The harness needs the pinned inputs (survey files under each provider's
licence, and the release artefacts), which this repository does not carry.
With them in place:

```bash
python harness/capture_us.py --out results/capture_us.json
python harness/run_selection.py scf_net_worth --folds 5
zsh harness/run_remedies.sh results/remedies_plan.txt
python harness/aggregate.py && python harness/remedies.py && python harness/coverage.py
python harness/export_public.py && python harness/report_tables.py
```

`harness/vendor/` holds byte-identical copies of two helpers from
PolicyEngine/qrf-moop-shortfall at dba3940.
