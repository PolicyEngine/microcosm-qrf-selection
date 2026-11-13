#!/bin/zsh
# Split each fit's adaptive-minus-honest contrast into selection and smoothing
# (selection.contrast_parts), one fit per line of a plan file:
#   <key> [--no-production]
# Writes results/selection_<key>_smoothing.json. A run resumes from its finished folds.
# Usage: run_smoothing.sh results/smoothing_plan.txt
cd "$(dirname "$0")/.."
mkdir -p logs
while read -r key flags; do
  [[ -z "$key" ]] && continue
  echo "== $key"
  .venv/bin/python -I harness/run_selection.py $key --smoothing --tag _smoothing ${=flags} \
    > logs/smoothing_${key}.log 2>&1
  echo "exit $key $?"
done < "$1"
