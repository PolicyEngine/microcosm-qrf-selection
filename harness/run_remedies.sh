#!/bin/zsh
# Remedy variants (min_samples_leaf 1, 5, 20 with quantile fit) on each line of a plan file:
#   <key> <target> [<target> ...]
# A fit whose result has five folds covering every planned target is skipped. A fit whose
# result lacks a planned target moves aside and runs again: a run resumes from its finished
# folds, which would never gain the missing target.
# Usage: run_remedies.sh results/remedies_plan.txt
cd "$(dirname "$0")/.."
mkdir -p results/oldcode
while read -r key targets; do
  [[ -z "$key" ]] && continue
  f=results/selection_${key}_remedies.json
  if [[ -f $f ]]; then
    .venv/bin/python -I -c "
import json, sys
d = json.load(open('$f'))
planned = set('$targets'.split())
covered = len(d['folds']) == 5 and all(planned <= set(x['targets']) for x in d['folds'])
sys.exit(0 if covered else 1)"
    if [[ $? -eq 0 ]]; then
      echo "skip $key (done)"; continue
    fi
    if .venv/bin/python -I -c "
import json, sys
d = json.load(open('$f'))
sys.exit(0 if len(d['folds']) == 5 else 1)"; then
      echo "move aside $key (planned targets changed)"; mv "$f" results/oldcode/
    fi
  fi
  echo "== $key: $targets"
  .venv/bin/python -I harness/run_selection.py $key --variants 1 5 20 --quantiles --tag _remedies \
    --only ${=targets} > logs/rem_${key}.log 2>&1
  echo "exit $key $?"
done < "$1"
