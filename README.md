# Adaptive-split selection in Microcosm's QRF imputations

Microcosm builds PolicyEngine's microdata, and its `RegimeGatedQRF` imputes
hundreds of survey variables onto the population frame. Each amount forest
reads its leaves with the same rows that chose its splits. This repository
measures what that does to the totals the production build imputes, target by
target, and how four remedies change it.

Start with [REPORT.md](REPORT.md). `results/public/TABLES.md` lists every
reported target, and the CSVs beside it hold the numbers.

## Layout

| Path | Contents |
|---|---|
| `REPORT.md` | Method, results, remedies, caveats and the decision the evidence informs |
| `harness/selection.py` | The same-tree test: grow Microcosm-style forests, fill them adaptively and honestly, compare |
| `harness/run_selection.py`, `run_*.sh` | Run the test on one captured production fit or a batch |
| `harness/capture*.py`, `puf_chain_twin.py`, `prodpath.py`, `groups.py`, `uc_groups.py`, `validate_cps.py` | Rebuild each production fit's donors, settings and recipients from the pinned inputs |
| `harness/aggregate.py`, `remedies.py`, `remedies_plan.py`, `scaling.py`, `coverage.py`, `inventory_totals.py`, `resummarize.py`, `empty_check.py` | Summaries of the runs |
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
