"""Cradle-to-gate LCA for the formate biorefinery.

Carbon accounting conventions used here
----------------------------------------
1. **CO₂ sourcing burden**: The supply-chain GWP of obtaining the CO₂ feedstock
   depends on its origin (biogenic waste ≈ 0.033, fossil CCS ≈ 0.12, DAC ≈ 0.48
   kg CO₂e per kg CO₂).  For corn-ethanol off-gas the CO₂ itself is biogenic
   (already in the short-term carbon cycle), so only compression / transport counts.

2. **Biogenic carbon credit**: When the CO₂ source is biogenic, the carbon fixed in
   SCP biomass at the cradle-to-gate boundary represents a net removal of biogenic
   CO₂ from the atmosphere.  Following CCU / ISO 14067 temporary-storage accounting,
   this is credited as:
       credit = −scp_kg × scp_carbon_fraction × (44 / 12)  [kg CO₂e]
   This credit is enabled by default when `co2_source == BIOGENIC_WASTE`.

3. **System-expansion protein displacement** (optional, off by default): SCP
   displaces conventional protein (soy meal).  When enabled, the avoided GWP of
   the displaced protein is subtracted:
       credit = −scp_kg × scp_displacement_kgco2e_per_kg

All three contributions are returned as separate named entries so figures can show
the full waterfall breakdown.

Haber–Bosch benchmarks (for reference in figures)
    Natural-gas-fed:  1.8–2.0 kg CO₂e/kg NH₃   (IEA 2021)
    Global average:   2.1 kg CO₂e/kg NH₃
    Coal-fed:         3.7–4.5 kg CO₂e/kg NH₃
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .config import CO2Source, LCAFactors, ScenarioCategory
from .process_blocks import ForegroundResults

# Atomic weights for biogenic carbon → CO₂ conversion
_C_TO_CO2 = 44.0 / 12.0  # kg CO₂ per kg C


@dataclass
class LCAResults:
    contributions_kgco2e_per_y: Dict[str, float]
    """Named annual GWP contributions.  Negative values are credits."""

    metrics: Dict[str, float]
    """Intensity metrics (kg CO₂e / kg product)."""


def evaluate_lca(
    result: ForegroundResults,
    lca_factors: LCAFactors,
) -> LCAResults:
    """Compute cradle-to-gate GWP with correct biogenic CO₂ and carbon accounting."""
    ledger = result.ledger
    cfg = result.scenario

    # --- 1. Electricity (largest contributor in most scenarios) ---
    electricity_factor = lca_factors.electricity_factor(cfg.electricity_case)

    # --- 2. CO₂ supply: burden depends on CO₂ source ---
    co2_supply_factor = lca_factors.co2_source_factor(cfg.co2_source)
    co2_feed_kg = ledger.mass_kg_per_y.get("co2_feed", 0.0)

    # --- 3. Biogenic carbon credit ---
    # Carbon stored in SCP at the gate originally came from atmospheric (biogenic) CO₂.
    # Applying this credit requires that the CO₂ source is biogenic (corn ethanol off-gas).
    biogenic_c_credit = 0.0
    if cfg.use_biogenic_carbon_credit and cfg.co2_source == CO2Source.BIOGENIC_WASTE:
        # scp_carbon_fraction × (44/12) converts kg C to kg CO₂ equivalent
        biogenic_c_credit = -(result.annual_scp_kg * lca_factors.scp_carbon_fraction * _C_TO_CO2)

    # --- 4. System-expansion protein displacement (optional) ---
    scp_displacement_credit = 0.0
    if cfg.use_scp_displacement_credit:
        scp_displacement_credit = -(result.annual_scp_kg * lca_factors.scp_displacement_kgco2e_per_kg)

    # --- 5. Methanol supply chain burden ---
    methanol_feed_kg = ledger.mass_kg_per_y.get("methanol_feed", 0.0)

    contributions: Dict[str, float] = {
        "electricity":               ledger.total_electricity_kwh_per_y * electricity_factor,
        "steam":                     ledger.total_steam_kg_per_y * lca_factors.steam_kgco2e_per_kg,
        "water":                     (ledger.mass_kg_per_y.get("electrolysis_water", 0.0)
                                      + ledger.mass_kg_per_y.get("broth_water", 0.0))
                                     * lca_factors.water_supply_kgco2e_per_kg,
        "wastewater":                ledger.mass_kg_per_y.get("wastewater", 0.0)
                                     * lca_factors.wastewater_treatment_kgco2e_per_kg,
        "naoh":                      ledger.mass_kg_per_y.get("naoh", 0.0)
                                     * lca_factors.naoh_kgco2e_per_kg,
        "h2so4":                     ledger.mass_kg_per_y.get("h2so4", 0.0)
                                     * lca_factors.h2so4_kgco2e_per_kg,
        "h3po4":                     ledger.mass_kg_per_y.get("h3po4", 0.0)
                                     * lca_factors.h3po4_kgco2e_per_kg,
        "mgcl2":                     ledger.mass_kg_per_y.get("mgcl2", 0.0)
                                     * lca_factors.mgcl2_kgco2e_per_kg,
        "co2_supply":                co2_feed_kg * co2_supply_factor,
        "methanol_supply":           methanol_feed_kg * lca_factors.methanol_kgco2e_per_kg,
        "membrane_replacement":      ledger.area_m2.get("recovery_membrane", 0.0) / 3.0
                                     * lca_factors.membrane_kgco2e_per_m2,
        # Credits (negative by convention)
        "biogenic_carbon_credit":    biogenic_c_credit,
        "scp_displacement_credit":   scp_displacement_credit,
    }

    total_gwp = sum(contributions.values())
    total_burden = sum(v for v in contributions.values() if v > 0)  # gross positive only

    primary_kg = max(1e-9, result.sellable_primary_product_kg)
    combined_kg = max(1e-9, result.sellable_primary_product_kg + result.annual_scp_kg)

    metrics: Dict[str, float] = {
        "total_gwp_kgco2e_per_y":                total_gwp,
        "total_burden_kgco2e_per_y":             total_burden,
        # Gross intensity (no credits) — apples-to-apples vs Haber-Bosch without displacement
        "gross_primary_gwp_kgco2e_per_kg":       total_burden / primary_kg,
        # Net intensity — includes all applicable credits
        "primary_product_gwp_kgco2e_per_kg":     total_gwp / primary_kg,
        # Intensity over all outputs (primary + SCP mass, before credits)
        "combined_output_gwp_kgco2e_per_kg":     total_gwp / combined_kg,
        # Electricity share of gross burden (useful for sensitivity interpretation)
        "electricity_fraction_of_burden":        (contributions["electricity"] / total_burden
                                                  if total_burden > 0 else 0.0),
    }
    return LCAResults(contributions_kgco2e_per_y=contributions, metrics=metrics)
