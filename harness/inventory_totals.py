"""Weighted production totals of every QRF-imputed output.

Reads the declarative stage manifests (``inventory/source_stages_*.json``,
exported from the production commits) and the production H5 files the
PolicyEngine bundle 6.16.0 pins, and writes one row per (stage, output):
the entity table that holds the column, its weighted total over all
records, and (US) the split by support channel.

Every entity's weight is its household's ``household_weight``: the US and
UK production files store weights on the household table only.

Usage:
  python inventory_totals.py --out ../results/inventory_totals.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

QRF_KINDS = {
    "fit_weighted_qrf", "fit_weighted_qrf_stage1", "fit_weighted_qrf_stage2",
    "fit_weighted_qrf_chain", "fit_weighted_acs_rent_qrf", "fit_tip_income_model",
    "fit_vehicle_model", "impute_spi_housing_shell", "impute_lifetime_isa_balance",
    "impute_frs_property_finance_costs", "impute_bus_use_band",
}

US = {
    "manifest": ROOT / "inventory/source_stages_us_cae8640.json",
    "h5": ROOT / "data/prod/us/populace_us_2024.h5",
    "entities": {
        "household": "household_id",
        "person": "person_id",
        "tax_unit": "tax_unit_id",
        "spm_unit": "spm_unit_id",
        "family": "family_id",
        "marital_unit": "marital_unit_id",
    },
    "person_link": {
        "tax_unit": "person_tax_unit_id",
        "spm_unit": "person_spm_unit_id",
        "family": "person_family_id",
        "marital_unit": "person_marital_unit_id",
        "household": "person_household_id",
    },
    "channel": "household_support_channel",
}
UK = {
    "manifest": ROOT / "inventory/source_stages_uk_67ee2ddf3.json",
    "h5": ROOT / "data/prod/uk/microcosm_uk_2024_25.h5",
    "entities": {"household": "household_id", "person": "person_id", "benunit": "benunit_id"},
    "person_link": {"benunit": "person_benunit_id", "household": "person_household_id"},
    "channel": None,
}


def qrf_stages(manifest: Path) -> list[dict]:
    stages = json.loads(manifest.read_text())["stages"]
    out = []
    for stage in stages:
        kinds = [op.get("kind", "") for op in stage.get("operations", [])]
        is_qrf = bool(set(kinds) & QRF_KINDS) or any(
            k.startswith("impute_") and k.endswith("_to_puf_support") for k in kinds)
        if is_qrf:
            out.append(stage)
    return out


def entity_weights(country: dict) -> dict[str, pd.DataFrame]:
    """Each entity table with a ``_w`` household weight and ``_channel`` column."""
    h5 = country["h5"]
    hh = pd.read_hdf(h5, key="household")
    person = pd.read_hdf(h5, key="person")
    w_of_hh = pd.Series(hh["household_weight"].to_numpy(), index=hh["household_id"].to_numpy())
    ch_of_hh = (pd.Series(hh[country["channel"]].astype(str).to_numpy(),
                          index=hh["household_id"].to_numpy())
                if country["channel"] else None)
    tables = {}
    p_hh = person[country["person_link"]["household"]].to_numpy()
    person["_w"] = w_of_hh.reindex(p_hh).to_numpy()
    person["_channel"] = ch_of_hh.reindex(p_hh).to_numpy() if ch_of_hh is not None else "all"
    tables["person"] = person
    hh["_w"] = hh["household_weight"].to_numpy()
    hh["_channel"] = hh[country["channel"]].astype(str).to_numpy() if country["channel"] else "all"
    tables["household"] = hh
    for entity, id_col in country["entities"].items():
        if entity in ("person", "household"):
            continue
        table = pd.read_hdf(h5, key=entity)
        link = country["person_link"][entity]
        # The entity's household: take it from any member (all members share one).
        first = person.drop_duplicates(link).set_index(link)
        table["_w"] = first["_w"].reindex(table[id_col].to_numpy()).to_numpy()
        table["_channel"] = first["_channel"].reindex(table[id_col].to_numpy()).to_numpy()
        tables[entity] = table
    return tables


def totals(country_name: str, country: dict) -> list[dict]:
    tables = entity_weights(country)
    rows = []
    for stage in qrf_stages(country["manifest"]):
        for output in stage.get("outputs", []):
            found = [(e, t) for e, t in tables.items() if output in t.columns]
            row = {"country": country_name, "stage": stage["stage"], "survey": stage.get("survey"),
                   "grain": stage.get("grain"), "output": output}
            if not found:
                row["entity"] = None
                rows.append(row)
                continue
            entity, table = found[0]
            values = pd.to_numeric(table[output], errors="coerce")
            row["entity"] = entity
            row["dtype"] = str(table[output].dtype)
            if values.isna().all():
                row["numeric"] = False
                rows.append(row)
                continue
            v = values.fillna(0.0).to_numpy(dtype=np.float64)
            w = table["_w"].to_numpy(dtype=np.float64)
            row["numeric"] = True
            row["n_unique"] = int(len(np.unique(v)))
            row["weighted_total"] = float((v * w).sum())
            row["weighted_abs_total"] = float((np.abs(v) * w).sum())
            row["weighted_nonzero_share"] = float(w[v != 0].sum() / w.sum())
            for channel in np.unique(table["_channel"].astype(str)):
                m = table["_channel"].astype(str).to_numpy() == channel
                row[f"total_{channel}"] = float((v[m] * w[m]).sum())
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = totals("us", US) + totals("uk", UK)
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    with pd.option_context("display.max_rows", 500, "display.width", 250,
                           "display.max_colwidth", 40):
        show = df.copy()
        for c in [c for c in show.columns if c.startswith("total") or c.startswith("weighted_")]:
            if c != "weighted_nonzero_share":
                show[c] = show[c] / 1e9
        print(show.drop(columns=["survey"]).to_string())


if __name__ == "__main__":
    main()
