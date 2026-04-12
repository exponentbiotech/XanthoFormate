from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from .config import AmmoniaRecoveryMethod, EconomicInputs, ScenarioCategory  # noqa: F401
from .process_blocks import ForegroundResults


@dataclass
class TEAResults:
    variable_opex_usd_per_y: Dict[str, float]
    fixed_opex_usd_per_y: Dict[str, float]
    credits_usd_per_y: Dict[str, float]
    capex_purchase_usd: Dict[str, float]
    metrics: Dict[str, float]


def capital_recovery_factor(discount_rate: float, years: float) -> float:
    if years <= 0.0:
        return 0.0
    if discount_rate == 0.0:
        return 1.0 / years
    numer = discount_rate * (1.0 + discount_rate) ** years
    denom = (1.0 + discount_rate) ** years - 1.0
    return numer / denom


def _direct_labor_cost(result: ForegroundResults, economic: EconomicInputs) -> float:
    primary_tpd = result.sellable_primary_product_kg / 365.0 / 1000.0
    operators_per_shift = max(2, math.ceil(primary_tpd / 25.0))
    total_operator_hours = operators_per_shift * 4.5 * 2080.0
    return total_operator_hours * economic.operator_loaded_wage_usd_per_h


def evaluate_tea(result: ForegroundResults, economic: EconomicInputs) -> TEAResults:
    ledger = result.ledger

    electricity_cost = ledger.total_electricity_kwh_per_y * economic.electricity_price_usd_per_kwh
    steam_cost = ledger.total_steam_kg_per_y * economic.steam_price_usd_per_kg
    water_cost = (
        ledger.mass_kg_per_y.get("electrolysis_water", 0.0)
        + ledger.mass_kg_per_y.get("broth_water", 0.0)
    ) * economic.water_price_usd_per_kg
    wastewater_cost = ledger.mass_kg_per_y.get("wastewater", 0.0) * economic.wastewater_price_usd_per_kg
    chemical_costs = {
        "naoh": ledger.mass_kg_per_y.get("naoh", 0.0) * economic.naoh_price_usd_per_kg,
        "h2so4": ledger.mass_kg_per_y.get("h2so4", 0.0) * economic.h2so4_price_usd_per_kg,
        # h3po4 is a direct cost for MAP precipitation; also used as fermentation nutrient elsewhere
        "h3po4": ledger.mass_kg_per_y.get("h3po4", 0.0) * economic.h3po4_price_usd_per_kg,
        "mgcl2": ledger.mass_kg_per_y.get("mgcl2", 0.0) * economic.mgcl2_price_usd_per_kg,
        "co2": ledger.mass_kg_per_y.get("co2_feed", 0.0) * economic.co2_price_usd_per_kg,
    }
    membrane_replacement = (
        ledger.area_m2.get("recovery_membrane", 0.0) * economic.membrane_cost_usd_per_m2 / 3.0
    )

    variable_opex = {
        "electricity": electricity_cost,
        "steam": steam_cost,
        "water": water_cost,
        "wastewater": wastewater_cost,
        **chemical_costs,
        "membrane_replacement": membrane_replacement,
    }

    direct_labor = _direct_labor_cost(result, economic)
    overhead = direct_labor * economic.admin_overhead_factor
    purchase_total = sum(result.equipment_purchase_usd.values())
    fixed_capital = purchase_total * economic.lang_factor
    working_capital = fixed_capital * economic.working_capital_fraction
    total_capital = fixed_capital + working_capital
    annualized_capex = total_capital * capital_recovery_factor(economic.discount_rate, economic.plant_life_years)
    maintenance = fixed_capital * economic.maintenance_factor
    stack_replacement = (
        result.equipment_purchase_usd.get("electrolyzer", 0.0)
        * economic.electrolyzer_stack_replacement_fraction
        / max(1.0, economic.electrolyzer_stack_life_years)
    )

    fixed_opex = {
        "direct_labor": direct_labor,
        "overhead": overhead,
        "maintenance": maintenance,
        "stack_replacement": stack_replacement,
        "annualized_capex": annualized_capex,
    }

    # Revenue: MAP/struvite pathway sells struvite fertilizer, not bulk ammonia.
    # Struvite revenue is modelled as a PRIMARY PRODUCT CREDIT so the LCOX numerator
    # correctly reflects (total_costs − struvite_revenue − SCP_credit) / kg_NH3_eq.
    # This keeps the denominator as kg NH3 equivalent across all methods for apples-to-apples comparison.
    is_struvite = (
        result.scenario.category == ScenarioCategory.AMMONIA_SCP
        and result.scenario.ammonia_recovery_method == AmmoniaRecoveryMethod.STRUVITE_MAP
    )
    is_map_fert = (
        result.scenario.category == ScenarioCategory.AMMONIA_SCP
        and result.scenario.ammonia_recovery_method == AmmoniaRecoveryMethod.MAP_FERTILIZER
    )

    if is_struvite:
        # Product is struvite fertilizer; revenue replaces NH3 commodity benchmark.
        product_revenue = result.annual_struvite_kg * economic.struvite_market_price_usd_per_kg
        primary_market_price = product_revenue / max(1e-9, result.sellable_primary_product_kg)
    elif is_map_fert:
        # Product is NH4H2PO4 (MAP 11-52-0 fertilizer); revenue replaces NH3 benchmark.
        product_revenue = result.annual_map_fert_kg * economic.map_fert_market_price_usd_per_kg
        primary_market_price = product_revenue / max(1e-9, result.sellable_primary_product_kg)
    elif result.scenario.category == ScenarioCategory.AMMONIA_SCP:
        primary_market_price = economic.ammonia_market_price_usd_per_kg
        product_revenue = result.sellable_primary_product_kg * primary_market_price
    else:
        primary_market_price = economic.urea_market_price_usd_per_kg
        product_revenue = result.sellable_primary_product_kg * primary_market_price

    credits = {
        "scp_credit": result.annual_scp_kg * economic.scp_market_price_usd_per_kg if result.scenario.use_scp_credit else 0.0,
        "h2_credit": ledger.mass_kg_per_y.get("h2_byproduct", 0.0) * result.scenario.h2_credit_usd_per_kg if result.scenario.use_h2_credit else 0.0,
        "co2_credit": ledger.mass_kg_per_y.get("co2_feed", 0.0) * result.scenario.co2_credit_usd_per_kg,
        # Struvite / MAP-fert revenues are treated as credits so LCOX = (costs − product_credit − SCP) / kg_NH3_eq
        "struvite_credit": (
            result.annual_struvite_kg * economic.struvite_market_price_usd_per_kg
            if is_struvite else 0.0
        ),
        "map_fert_credit": (
            result.annual_map_fert_kg * economic.map_fert_market_price_usd_per_kg
            if is_map_fert else 0.0
        ),
    }

    variable_total = sum(variable_opex.values())
    fixed_total = sum(fixed_opex.values())
    credits_total = sum(credits.values())
    total_annual_cost = variable_total + fixed_total
    gross_lcox = total_annual_cost / max(1e-9, result.sellable_primary_product_kg)
    net_lcox = (total_annual_cost - credits_total) / max(1e-9, result.sellable_primary_product_kg)
    benchmark_total_revenue = product_revenue + credits_total
    annual_cash_flow = benchmark_total_revenue - variable_total - (fixed_total - annualized_capex)
    npv = -total_capital
    full_years = int(math.floor(economic.plant_life_years))
    fractional_year = economic.plant_life_years - float(full_years)
    for year in range(1, full_years + 1):
        npv += annual_cash_flow / (1.0 + economic.discount_rate) ** year
    if fractional_year > 1e-9:
        npv += (
            annual_cash_flow
            * fractional_year
            / (1.0 + economic.discount_rate) ** economic.plant_life_years
        )

    metrics = {
        "purchase_capex_usd": purchase_total,
        "fixed_capital_usd": fixed_capital,
        "working_capital_usd": working_capital,
        "total_capital_usd": total_capital,
        "total_annual_cost_usd_per_y": total_annual_cost,
        "gross_primary_lcox_usd_per_kg": gross_lcox,
        "net_primary_lcox_usd_per_kg": net_lcox,
        "benchmark_primary_revenue_usd_per_y": product_revenue,
        "benchmark_total_revenue_usd_per_y": benchmark_total_revenue,
        "annual_cash_flow_usd_per_y": annual_cash_flow,
        "npv_usd": npv,
        "scp_credit_share": credits["scp_credit"] / max(1e-9, credits_total if credits_total else 1.0),
    }

    return TEAResults(
        variable_opex_usd_per_y=variable_opex,
        fixed_opex_usd_per_y=fixed_opex,
        credits_usd_per_y=credits,
        capex_purchase_usd=result.equipment_purchase_usd,
        metrics=metrics,
    )
