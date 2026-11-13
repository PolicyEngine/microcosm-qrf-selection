# Adaptive-split selection in Microcosm's QRF imputations

Microcosm builds PolicyEngine's microdata, and its `RegimeGatedQRF` imputes
hundreds of survey variables onto the population frame. Each amount forest
fills its leaves with the same rows that chose its splits. This repository
compares that adaptive fill with an honest fill of the same trees on the
targets the production build imputes, splits the contrast into selection and
the fills' different smoothing, and measures how four remedies change the
totals and the quantile fit.

Start with [REPORT.md](REPORT.md). `results/public/TABLES.md` lists every
reported target, and the CSVs beside it hold the numbers.

## Layout

| Path | Contents |
|---|---|
| `REPORT.md` | Method, results, remedies, caveats and the decision the evidence informs |
| `harness/selection.py` | The same-tree test: grow Microcosm-style forests, fill them adaptively and honestly, compare |
| `harness/run_selection.py`, `run_*.sh` | Run the test on one captured production fit or a batch |
| `harness/capture*.py`, `puf_chain_twin.py`, `prodpath.py`, `groups.py`, `uc_groups.py`, `validate_cps.py` | Rebuild each production fit's donors, settings and recipients from the pinned inputs |
| `harness/aggregate.py`, `remedies.py`, `remedies_plan.py`, `coverage_by_tau.py`, `scaling.py`, `draw_means.py`, `coverage.py`, `inventory_totals.py`, `resummarize.py`, `empty_check.py`, `make_tables.sh` | Summaries of the runs |
| `harness/compare_runs.py`, `rerun_check.sh` | Compare two runs number by number; rerun finished fits under the current harness |
| `harness/known_answer.py` | The test on synthetic data where the answer is known |
| `harness/smoothing_check.py` | The contrast's split into selection and smoothing on synthetic processes with a known conditional mean |
| `harness/smoothing_plan.py`, `run_smoothing.sh`, `smoothing.py` | The same split on every reported fit, with a conditional mean fitted on held-out donors |
| `harness/run_draw_means.sh` | The contrast in production's expected draws on 13 fits |
| `harness/run_full_size.sh`, `full_size.py` | Each remedy against the forest production would grow on all of a fold's donors |
| `harness/report_rows.py` | Generates every table row `REPORT.md` quotes from the public files; `harness/tests/test_report_numbers.py` checks them |
| `harness/export_public.py`, `report_tables.py` | Write the public result files and their tables |
| `harness/vendor/` | Byte-identical helpers from [qrf-moop-shortfall](https://github.com/PolicyEngine/qrf-moop-shortfall) at dba3940 (MIT) |
| `harness/tests/` | Tests: harness invariants on synthetic data, and rules the public files must meet |
| `inventory/` | Declarative stage manifests exported from the production commits |
| `results/public/` | Aggregates only: every number in the report |

## What this repository leaves out

- **Survey records and releases.** Every donor table, release file and run
  output stays on the machine that ran it. Rebuild them with the capture
  scripts, under each data provider's licence.
- **Anything derived from the HMRC Survey of Personal Incomes public use
  tape.** Its licence sends any analysis to HMRC ten working days before
  anyone distributes or publishes it. That rules out the two fits trained on
  the tape, and every UK production-recipient number, because the UK build
  frame holds the SPI-channel households.
- **UK release statistics and absolute UK totals.** UK rows report percentages
  of held-out donor totals.
- **Results on few donors.** A target whose forests grew on fewer than 50
  nonzero training donors shows no result.

`harness/export_public.py` applies these rules, and
`harness/tests/test_public_files.py` checks them on every push.

## Running it

The measurements ran on Python 3.14.7 with the versions in
`requirements.txt`, which pins Microcosm at 67ee2ddf3.

```bash
uv venv --python 3.14+gil .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest harness/tests
```

The tests use synthetic data and the public result files. Rerunning the
measurements also needs the pinned surveys and releases; REPORT.md section 10
gives the commands, and the path constants at the top of each capture script
say where each input goes.

## License

MIT. See [LICENSE](LICENSE).
