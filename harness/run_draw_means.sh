#!/bin/zsh
# Each fill's expected total under production's draw, on the planned targets of the fits
# the scaling check used and the three SCF fits. Writes results/selection_<key>_draws.json.
cd "$(dirname "$0")/.."
for key in acs_rent org_wages sipp_financial_assets__0 sipp_vehicles uk_etb_services uk_lcfs \
           uk_lcfs_road_fuel uk_was__2 uk_was__3 uk_was__4 scf_net_worth scf_financial_assets \
           scf_auto_loans; do
  targets=$(awk -v k=$key '$1==k {for (i=2;i<=NF;i++) print $i}' results/remedies_plan.txt)
  .venv/bin/python -I harness/run_selection.py $key --draw-means --tag _draws --no-production \
    --only ${=targets} > logs/draws_${key}.log 2>&1
  echo "exit $key $?"
done
