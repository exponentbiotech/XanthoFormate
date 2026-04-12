from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .config import (
    AmmoniaRecoveryMethod,
    EconomicInputs,
    ScenarioCategory,
    ScenarioConfig,
    TechnologyInputs,
    UreaRecoveryMethod,
)

# Molecular weights (kg/mol)
_MW_MGCL2 = 0.0952
_MW_H3PO4 = 0.0980
from .streams import StreamLedger, water_removed_from_scp


NH3_N_FRACTION = 14.0 / 17.0
UREA_N_FRACTION = 28.0 / 60.0
UREA_C_FRACTION = 12.0 / 60.0
FORMATE_C_FRACTION = 12.0 / 46.0
FORMATE_MW_KG_PER_MOL = 0.046
NH3_MW_KG_PER_MOL = 0.017
NAOH_MW_KG_PER_MOL = 0.040


@dataclass
class ForegroundResults:
    scenario: ScenarioConfig
    annual_operating_hours: float
    sellable_primary_product_kg: float
    upstream_primary_product_kg: float
    annual_scp_kg: float
    annual_struvite_kg: float          # Non-zero only when STRUVITE_MAP recovery is used
    annual_map_fert_kg: float          # Non-zero only when MAP_FERTILIZER recovery is used
    working_volume_m3: float
    ledger: StreamLedger
    closure: Dict[str, float]
    equipment_purchase_usd: Dict[str, float]
    notes: List[str]


def _scale_cost(base_cost: float, actual_rate: float, reference_rate: float, exponent: float) -> float:
    if actual_rate <= 0.0:
        return 0.0
    return base_cost * (actual_rate / reference_rate) ** exponent


def _electrolysis(
    annual_formate_kg: float,
    annual_hours: float,
    technology: TechnologyInputs,
    economic: EconomicInputs,
    ledger: StreamLedger,
) -> Dict[str, float]:
    formate_mol = annual_formate_kg / FORMATE_MW_KG_PER_MOL
    electricity_kwh = formate_mol * technology.electrolyzer_energy_kwh_per_mol_formate
    water_kg = formate_mol * technology.water_kg_per_mol_formate
    co2_kg = annual_formate_kg * technology.co2_kg_per_kg_formate
    h2_mol = formate_mol * technology.h2_faradaic_efficiency / max(1e-9, technology.formate_faradaic_efficiency)
    h2_kg = h2_mol * 0.002
    power_kw = electricity_kwh / max(1e-9, annual_hours)
    capex = power_kw * economic.electrolyzer_installed_cost_usd_per_kw

    ledger.add_mass("formate", annual_formate_kg)
    ledger.add_mass("co2_feed", co2_kg)
    ledger.add_mass("electrolysis_water", water_kg)
    ledger.add_mass("h2_byproduct", h2_kg)
    ledger.add_electricity("electrolyzer", electricity_kwh)

    return {
        "electrolyzer_purchase_usd": capex,
        "h2_byproduct_kg": h2_kg,
        "co2_feed_kg": co2_kg,
        "electrolysis_water_kg": water_kg,
    }


def _ammonia_recovery(
    config: ScenarioConfig,
    sellable_nh3_kg: float,
    annual_hours: float,
    technology: TechnologyInputs,
    ledger: StreamLedger,
) -> Dict[str, float]:
    """Return cost-driver quantities for the selected NH3 recovery method.

    Returns keys: naoh_kg, membrane_area_m2, mgcl2_kg, h3po4_kg, struvite_kg.
    """
    nh3_mol = sellable_nh3_kg / NH3_MW_KG_PER_MOL
    recovery_power_kwh = 0.0
    naoh_kg = 0.0
    membrane_area_m2 = 0.0
    mgcl2_kg = 0.0
    h3po4_kg = 0.0
    struvite_kg = 0.0
    map_fert_kg = 0.0

    if config.ammonia_recovery_method == AmmoniaRecoveryMethod.VACUUM_STRIPPING:
        # Vacuum stripping: NaOH raises pH > 10, NH3 stripped under vacuum,
        # captured in H2SO4 acid trap.  Energy-intensive but high purity product.
        recovery_power_kwh = sellable_nh3_kg * technology.vacuum_energy_kwh_per_kg_ammonia
        naoh_kg = nh3_mol * technology.vacuum_naoh_mol_per_mol_nh3 * NAOH_MW_KG_PER_MOL

    elif config.ammonia_recovery_method == AmmoniaRecoveryMethod.MEMBRANE:
        # Hydrophobic hollow-fiber gas-permeable membrane: NH3 diffuses across membrane
        # as gas, absorbed in acid trap. No pH chemicals, moderate electricity.
        recovery_power_kwh = sellable_nh3_kg * technology.membrane_energy_kwh_per_kg_ammonia
        membrane_area_m2 = sellable_nh3_kg / max(1e-9, annual_hours * technology.membrane_flux_kg_per_m2_h)

    elif config.ammonia_recovery_method == AmmoniaRecoveryMethod.STRUVITE_MAP:
        # Struvite / MAP precipitation: Mg2+ + NH4+ + PO4^3- → MgNH4PO4·6H2O
        # at pH 8.5-9.0. Very low energy, produces slow-release fertilizer.
        # Reagents: MgCl2 and H3PO4 (unless P already present in fermentation broth).
        recovery_power_kwh = sellable_nh3_kg * technology.map_electricity_kwh_per_kg_nh3
        mgcl2_kg = nh3_mol * technology.map_mg_mol_per_mol_nh3 * _MW_MGCL2
        h3po4_kg = nh3_mol * technology.map_phosphate_mol_per_mol_nh3 * _MW_H3PO4
        struvite_kg = sellable_nh3_kg * technology.struvite_kg_per_kg_nh3

    elif config.ammonia_recovery_method == AmmoniaRecoveryMethod.MAP_FERTILIZER:
        # Monoammonium phosphate (MAP, 11-52-0) fertilizer route:
        # 1. Gas-permeable membrane strips NH3 from the broth (same as MEMBRANE method).
        # 2. Stripped NH3 is absorbed into a H3PO4 acid trap → NH4H2PO4 in situ.
        #    NH3(g) + H3PO4(l) → NH4H2PO4(l) → spray-dried/prilled MAP granules.
        # No separate pH-adjustment chemicals are needed; H3PO4 acts as both the
        # absorbing acid and the P source.  The product is standard commodity MAP fertilizer.
        recovery_power_kwh = sellable_nh3_kg * technology.membrane_energy_kwh_per_kg_ammonia
        membrane_area_m2 = sellable_nh3_kg / max(1e-9, annual_hours * technology.membrane_flux_kg_per_m2_h)
        h3po4_kg = nh3_mol * technology.map_fert_h3po4_mol_per_mol_nh3 * _MW_H3PO4
        map_fert_kg = sellable_nh3_kg * technology.map_fert_nh4h2po4_kg_per_kg_nh3

    else:  # AIR_STRIPPING
        recovery_power_kwh = sellable_nh3_kg * technology.air_stripping_energy_kwh_per_kg_ammonia
        naoh_kg = nh3_mol * technology.vacuum_naoh_mol_per_mol_nh3 * NAOH_MW_KG_PER_MOL

    ledger.add_mass("naoh", naoh_kg)
    ledger.add_mass("mgcl2", mgcl2_kg)
    ledger.add_mass("h3po4", h3po4_kg)
    ledger.add_electricity("primary_recovery", recovery_power_kwh)
    if membrane_area_m2 > 0.0:
        ledger.add_area("recovery_membrane", membrane_area_m2)

    return {
        "naoh_kg": naoh_kg,
        "membrane_area_m2": membrane_area_m2,
        "mgcl2_kg": mgcl2_kg,
        "h3po4_kg": h3po4_kg,
        "struvite_kg": struvite_kg,
        "map_fert_kg": map_fert_kg,
    }


def _urea_recovery(
    config: ScenarioConfig,
    sellable_urea_kg: float,
    annual_hours: float,
    technology: TechnologyInputs,
    ledger: StreamLedger,
) -> Dict[str, float]:
    """Return cost-driver quantities for the selected urea recovery method."""
    electricity_kwh = 0.0
    steam_kg = 0.0
    membrane_area_m2 = 0.0

    if config.urea_recovery_method == UreaRecoveryMethod.EVAPORATION:
        # Conventional single-effect evaporation + crystallization. Steam-intensive
        # but low CapEx and well-proven at fertilizer scale.
        electricity_kwh = sellable_urea_kg * technology.evaporation_kwh_per_kg_urea
        steam_kg = sellable_urea_kg * technology.evaporation_steam_kg_per_kg_urea

    elif config.urea_recovery_method == UreaRecoveryMethod.MVR_CRYSTALLIZATION:
        # Mechanical vapor recompression (MVR) replaces most steam with electricity.
        # The compressor reuses evaporated vapor as heating medium, cutting LP steam
        # by >90 %.  Standard technology in sugar, ammonium sulfate, and fertilizer.
        electricity_kwh = sellable_urea_kg * technology.mvr_kwh_per_kg_urea
        steam_kg = sellable_urea_kg * technology.mvr_steam_kg_per_kg_urea

    else:  # HYBRID — membrane pre-concentration + reduced evaporation (backward compat)
        electricity_kwh = sellable_urea_kg * technology.evaporation_kwh_per_kg_urea * 0.70
        steam_kg = sellable_urea_kg * technology.evaporation_steam_kg_per_kg_urea * 0.35
        membrane_area_m2 = sellable_urea_kg / max(1e-9, annual_hours * 10.0)

    ledger.add_electricity("primary_recovery", electricity_kwh)
    ledger.add_steam("primary_recovery", steam_kg)
    if membrane_area_m2 > 0.0:
        ledger.add_area("recovery_membrane", membrane_area_m2)

    return {
        "membrane_area_m2": membrane_area_m2,
        "steam_kg": steam_kg,
    }


def _scp_processing(
    broth_water_kg: float,
    annual_scp_kg: float,
    technology: TechnologyInputs,
    ledger: StreamLedger,
) -> Dict[str, float]:
    harvest_kwh = (broth_water_kg / 1000.0) * technology.centrifuge_kwh_per_m3_broth
    drying_water_kg = water_removed_from_scp(
        annual_scp_kg,
        technology.scp_cake_moisture,
        technology.scp_product_moisture,
    )
    drying_kwh = drying_water_kg * technology.scp_drying_kwh_per_kg_water
    wastewater_kg = max(0.0, broth_water_kg - annual_scp_kg)

    ledger.add_electricity("scp_harvest", harvest_kwh)
    ledger.add_electricity("scp_drying", drying_kwh)
    ledger.add_mass("scp_product", annual_scp_kg)
    ledger.add_mass("wastewater", wastewater_kg)

    return {
        "drying_water_kg": drying_water_kg,
        "wastewater_kg": wastewater_kg,
    }


def simulate_foreground(
    config: ScenarioConfig,
    technology: TechnologyInputs,
    economic: EconomicInputs,
) -> ForegroundResults:
    annual_hours = 8760.0 * technology.capacity_factor
    ledger = StreamLedger()
    notes: List[str] = []

    if config.category == ScenarioCategory.AMMONIA_SCP:
        sellable_primary_kg = config.annual_primary_product_kg
        # MAP precipitation has its own recovery efficiency distinct from stripping
        nh3_recovery_eff = (
            technology.map_recovery_efficiency
            if config.ammonia_recovery_method == AmmoniaRecoveryMethod.STRUVITE_MAP
            else technology.ammonia_recovery_efficiency
        )
        upstream_primary_kg = sellable_primary_kg / max(1e-9, nh3_recovery_eff)
        annual_formate_kg = upstream_primary_kg * technology.formate_to_ammonia_kg_per_kg
        annual_scp_kg = upstream_primary_kg * technology.scp_to_ammonia_kg_per_kg
        broth_water_kg = upstream_primary_kg * technology.broth_water_kg_per_kg_ammonia
        working_volume_m3 = upstream_primary_kg / max(1e-9, annual_hours * technology.ammonia_productivity_kg_per_m3_h)
        product_n_fraction = NH3_N_FRACTION
        product_c_fraction = 0.0
        recovery_meta = _ammonia_recovery(config, sellable_primary_kg, annual_hours, technology, ledger)
        if config.ammonia_recovery_method == AmmoniaRecoveryMethod.STRUVITE_MAP:
            equipment_base_cost = technology.map_settler_base_cost_usd
            notes.append("MAP precipitation: MgCl2 + H3PO4 reagents; product is struvite fertilizer.")
        elif config.ammonia_recovery_method == AmmoniaRecoveryMethod.MAP_FERTILIZER:
            # Membrane system (same as MEMBRANE) + granulation equipment
            equipment_base_cost = technology.map_fert_granulation_base_cost_usd
            notes.append("MAP fertilizer route: membrane strip NH3, absorb in H3PO4 → NH4H2PO4 (11-52-0).")
        else:
            equipment_base_cost = technology.recovery_base_cost_usd
            notes.append("Ammonia pathway uses recovery-specific caustic and electricity loads.")
    else:
        sellable_primary_kg = config.annual_primary_product_kg
        upstream_primary_kg = sellable_primary_kg / max(1e-9, technology.urea_recovery_efficiency)
        annual_formate_kg = upstream_primary_kg * technology.formate_to_urea_kg_per_kg
        annual_scp_kg = upstream_primary_kg * technology.scp_to_urea_kg_per_kg
        broth_water_kg = upstream_primary_kg * technology.broth_water_kg_per_kg_urea
        working_volume_m3 = upstream_primary_kg / max(1e-9, annual_hours * technology.urea_productivity_kg_per_m3_h)
        product_n_fraction = UREA_N_FRACTION
        product_c_fraction = UREA_C_FRACTION
        recovery_meta = _urea_recovery(config, sellable_primary_kg, annual_hours, technology, ledger)
        # MVR has higher CapEx but lower steam OPEX; single-effect is cheaper to install
        if config.urea_recovery_method == UreaRecoveryMethod.MVR_CRYSTALLIZATION:
            equipment_base_cost = technology.mvr_base_cost_usd
            notes.append("MVR crystallization: electricity-driven evaporation, minimal steam.")
        else:
            equipment_base_cost = technology.urea_recovery_base_cost_usd
            notes.append("Bio-urea pathway is a screening case with stoichiometric and recovery placeholders.")

    electrolysis_meta = _electrolysis(annual_formate_kg, annual_hours, technology, economic, ledger)
    n2_required_kg = sellable_primary_kg * product_n_fraction + annual_scp_kg * technology.nitrogen_mass_fraction_scp
    air_compression_kwh = n2_required_kg * technology.air_compression_kwh_per_kg_n2
    agitation_kwh = working_volume_m3 * annual_hours * technology.agitation_aeration_kwh_per_m3_h
    ledger.add_mass("nitrogen_feed_equivalent", n2_required_kg)
    ledger.add_mass("broth_water", broth_water_kg)
    ledger.add_electricity("air_compression", air_compression_kwh)
    ledger.add_electricity("agitation_aeration", agitation_kwh)

    scp_meta = _scp_processing(broth_water_kg, annual_scp_kg, technology, ledger)

    actual_primary_rate_kg_per_day = sellable_primary_kg / 365.0
    scp_rate_kg_per_day = annual_scp_kg / 365.0
    equipment_purchase_usd = {
        "electrolyzer": electrolysis_meta["electrolyzer_purchase_usd"],
        "bioreactor": _scale_cost(
            technology.bioreactor_base_cost_usd,
            working_volume_m3,
            technology.bioreactor_reference_m3,
            0.60,
        ),
        "primary_recovery": _scale_cost(
            equipment_base_cost,
            actual_primary_rate_kg_per_day,
            1000.0,
            technology.equipment_scaling_exponent,
        ),
        "scp_processing": _scale_cost(
            technology.scp_processing_base_cost_usd,
            scp_rate_kg_per_day,
            1000.0,
            technology.equipment_scaling_exponent,
        ),
    }
    if recovery_meta.get("membrane_area_m2", 0.0) > 0.0:
        equipment_purchase_usd["recovery_membrane"] = recovery_meta["membrane_area_m2"] * economic.membrane_cost_usd_per_m2

    carbon_in_kg = annual_formate_kg * FORMATE_C_FRACTION
    carbon_to_primary_kg = sellable_primary_kg * product_c_fraction
    carbon_to_scp_kg = annual_scp_kg * technology.carbon_mass_fraction_scp
    carbon_to_offgas_kg = carbon_in_kg - carbon_to_primary_kg - carbon_to_scp_kg

    nitrogen_in_kg = n2_required_kg
    nitrogen_to_primary_kg = sellable_primary_kg * product_n_fraction
    nitrogen_to_scp_kg = annual_scp_kg * technology.nitrogen_mass_fraction_scp
    nitrogen_to_loss_kg = nitrogen_in_kg - nitrogen_to_primary_kg - nitrogen_to_scp_kg

    closure = {
        "carbon_in_kg": carbon_in_kg,
        "carbon_out_primary_kg": carbon_to_primary_kg,
        "carbon_out_scp_kg": carbon_to_scp_kg,
        "carbon_out_offgas_kg": carbon_to_offgas_kg,
        "carbon_balance_error_kg": carbon_in_kg - carbon_to_primary_kg - carbon_to_scp_kg - carbon_to_offgas_kg,
        "nitrogen_in_kg": nitrogen_in_kg,
        "nitrogen_out_primary_kg": nitrogen_to_primary_kg,
        "nitrogen_out_scp_kg": nitrogen_to_scp_kg,
        "nitrogen_out_loss_kg": nitrogen_to_loss_kg,
        "nitrogen_balance_error_kg": nitrogen_in_kg - nitrogen_to_primary_kg - nitrogen_to_scp_kg - nitrogen_to_loss_kg,
        "wastewater_kg": scp_meta["wastewater_kg"],
    }

    notes.append(
        "Element closure is enforced algebraically in this screening model; strain-specific yields remain explicit assumptions."
    )
    is_nh3 = config.category == ScenarioCategory.AMMONIA_SCP
    annual_struvite_kg = recovery_meta.get("struvite_kg", 0.0) if is_nh3 else 0.0
    annual_map_fert_kg = recovery_meta.get("map_fert_kg", 0.0) if is_nh3 else 0.0
    return ForegroundResults(
        scenario=config,
        annual_operating_hours=annual_hours,
        sellable_primary_product_kg=sellable_primary_kg,
        upstream_primary_product_kg=upstream_primary_kg,
        annual_scp_kg=annual_scp_kg,
        annual_struvite_kg=annual_struvite_kg,
        annual_map_fert_kg=annual_map_fert_kg,
        working_volume_m3=working_volume_m3,
        ledger=ledger,
        closure=closure,
        equipment_purchase_usd=equipment_purchase_usd,
        notes=notes,
    )
