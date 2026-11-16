#!/bin/zsh
# Same-tree selection runs (held-out donors) for captured UK stage fits, one at a time.
cd "${0:a:h}/.."
for k in "$@"; do
  [[ -f results/selection_${k}.json ]] && grep -q '"fold": 4' results/selection_${k}.json && continue
  .venv/bin/python -I harness/run_selection.py $k --no-production > logs/sel_${k}.log 2>&1
done
echo batch done
