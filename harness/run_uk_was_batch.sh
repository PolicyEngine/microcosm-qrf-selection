#!/bin/zsh
# Same-tree selection runs for the 11 UK WAS chain segments (held-out donors only), one at a time.
cd "${0:a:h}/.."
for i in 0 1 2 3 4 5 6 7 8 9 10; do
  k=uk_was__$i
  [[ -f results/selection_${k}.json ]] && grep -q '"fold": 4' results/selection_${k}.json && continue
  .venv/bin/python harness/run_selection.py $k --no-production > logs/sel_${k}.log 2>&1
done
echo batch done
