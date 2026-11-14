#!/bin/zsh
# Rerun finished remedy fits under the current harness, compare every stored number,
# and keep the new result. The old one moves to results/before_rerun/.
# Usage: rerun_check.sh <key> [<key> ...]
cd "$(dirname "$0")/.."
mkdir -p results/before_rerun
for key in "$@"; do
  f=results/selection_${key}_remedies.json
  targets=$(.venv/bin/python -I -c "import json; print(' '.join(json.load(open('$f'))['folds'][0]['targets']))")
  new=results/selection_${key}_rerun.json
  if ! .venv/bin/python -I harness/run_selection.py $key --variants 1 5 20 --quantiles --tag _rerun \
      --only ${=targets} > logs/rerun_${key}.log 2>&1; then
    echo "FAILED $key"; rm -f $new results/selection_${key}_rerun_v*.csv; continue
  fi
  .venv/bin/python -I harness/compare_runs.py $f $new
  mv $f results/before_rerun/
  mv $new $f
  for v in 1 5 20; do
    mv results/selection_${key}_rerun_v$v.csv results/selection_${key}_remedies_v$v.csv
  done
done
