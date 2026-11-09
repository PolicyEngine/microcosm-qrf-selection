#!/bin/zsh
# Same-tree selection runs for every captured US refresh fit, one at a time (4 threads each).
cd "${0:a:h}/.."
for k in scf_financial_assets sipp_financial_assets__0 sipp_financial_assets__1 sipp_financial_assets__2 \
         scf_auto_loans sipp_vehicles sipp_tips org_wages; do
  [[ -f results/selection_${k}.json ]] && grep -q '"fold": 4' results/selection_${k}.json && continue
  .venv/bin/python harness/run_selection.py $k > logs/sel_${k}.log 2>&1
done
for k in ssi_disability_criteria sipp_head_start voluntary_filing; do
  [[ -f donors/${k}.json ]] || continue
  .venv/bin/python harness/run_selection.py $k --no-production > logs/sel_${k}.log 2>&1
done
echo batch done
