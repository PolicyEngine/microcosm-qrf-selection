#!/bin/zsh
# Remedy variants (min_samples_leaf 1, 5, 20 with quantile fit) on each line of a plan file:
#   <key> <target> [<target> ...]
# Usage: run_remedies.sh results/remedies_plan.txt
cd "$(dirname "$0")/.."
while read -r key targets; do
  [[ -z "$key" ]] && continue
  if [[ -f results/selection_${key}_remedies.json ]] && \
     .venv/bin/python -I -c "import json,sys; d=json.load(open('results/selection_${key}_remedies.json')); sys.exit(0 if len(d['folds'])==5 else 1)"; then
    echo "skip $key (done)"; continue
  fi
  echo "== $key: $targets"
  .venv/bin/python -I harness/run_selection.py $key --variants 1 5 20 --quantiles --tag _remedies \
    --only ${=targets} > logs/rem_${key}.log 2>&1
  echo "exit $key $?"
done < "$1"
