#!/bin/zsh
# Same-tree selection runs for the reconstructed US PUF-support fits and ACS rent,
# one at a time (4 threads each), after the UK WAS batch frees its threads.
cd "${0:a:h}/.."
was=$(cat logs/uk_was_batch.pid 2>/dev/null)
while [[ -n "$was" ]] && kill -0 "$was" 2>/dev/null; do sleep 20; done
for k in acs_rent puf_support_retirement_distributions puf_support_other_health_insurance \
         puf_support_retirement_contributions puf_support_childcare puf_support_child_support \
         puf_support_disability_benefits puf_support_workers_compensation \
         puf_support_energy_subsidy puf_support_weeks_unemployed; do
  [[ -f results/selection_${k}.json ]] && grep -q '"fold": 4' results/selection_${k}.json && continue
  .venv/bin/python harness/run_selection.py $k > logs/sel_${k}.log 2>&1
done
echo batch done
