#!/bin/zsh
# Rebuild every summary and the public files from the finished runs.
# Usage: make_tables.sh [private-dir]
cd "$(dirname "$0")/.."
private=${1:-$HOME/reviews/microcosm-selection-2026-11-07-private}
py=(.venv/bin/python -I)
set -e
$py harness/aggregate.py
$py harness/remedies.py
$py harness/scaling.py
$py harness/coverage.py
$py harness/export_public.py --private-dir $private
$py harness/report_tables.py --source results/public --out results/public/TABLES.md
$py harness/report_tables.py --source results --out $private/TABLES.md --private
