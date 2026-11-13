#!/bin/zsh
# Each fold's production-size forest at leaf floors 1, 5 and 20, on each line of a plan file:
#   <key> <target> [<target> ...]
# Writes results/selection_<key>_fullsize.json. A run resumes from its finished folds.
# Usage: run_full_size.sh results/remedies_plan.txt
cd "$(dirname "$0")/.."
mkdir -p logs
while read -r key targets; do
  [[ -z "$key" ]] && continue
  echo "== $key: $targets"
  .venv/bin/python -I harness/run_selection.py $key --quantiles --full-variants 1 5 20 \
    --tag _fullsize --only ${=targets} > logs/fullsize_${key}.log 2>&1
  echo "exit $key $?"
done < "$1"
