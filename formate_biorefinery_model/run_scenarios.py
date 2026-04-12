from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .config import (
    AmmoniaRecoveryMethod,
    CO2Source,
    ElectricityCase,
    ScenarioCategory,
    ScenarioConfig,
    UreaRecoveryMethod,
    build_default_inputs,
    default_scenarios,
    flatten_record_tables,
)
from .lca import LCAResults, evaluate_lca
from .process_blocks import ForegroundResults, simulate_foreground
from .tea import TEAResults, evaluate_tea


@dataclass
class ScenarioEvaluation:
    foreground: ForegroundResults
    tea: TEAResults
    lca: LCAResults
    source_rows: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        scenario = self.foreground.scenario
        primary_name = "ammonia" if scenario.category == ScenarioCategory.AMMONIA_SCP else "urea"
        return {
            "scenario": scenario.category.value,
            "annual_primary_product_tpy": scenario.annual_primary_product_tpy,
            "electricity_case": scenario.electricity_case.value,
            "co2_source": scenario.co2_source.value,
            "ammonia_recovery_method": scenario.ammonia_recovery_method.value,
            "urea_recovery_method": scenario.urea_recovery_method.value,
            "use_biogenic_carbon_credit": scenario.use_biogenic_carbon_credit,
            "use_scp_displacement_credit": scenario.use_scp_displacement_credit,
            "primary_product_name": primary_name,
            "sellable_primary_product_kg_per_y": self.foreground.sellable_primary_product_kg,
            "annual_scp_kg_per_y": self.foreground.annual_scp_kg,
            "working_volume_m3": self.foreground.working_volume_m3,
            "gross_primary_lcox_usd_per_kg": self.tea.metrics["gross_primary_lcox_usd_per_kg"],
            "net_primary_lcox_usd_per_kg": self.tea.metrics["net_primary_lcox_usd_per_kg"],
            "primary_product_gwp_kgco2e_per_kg": self.lca.metrics["primary_product_gwp_kgco2e_per_kg"],
            "combined_output_gwp_kgco2e_per_kg": self.lca.metrics["combined_output_gwp_kgco2e_per_kg"],
            "npv_usd": self.tea.metrics["npv_usd"],
        }


def evaluate_scenario(
    config: ScenarioConfig,
    overrides: Optional[Dict[str, float]] = None,
) -> ScenarioEvaluation:
    economic, lca_factors, technology, records = build_default_inputs(overrides=overrides or config.user_overrides)
    foreground = simulate_foreground(config, technology, economic)
    tea = evaluate_tea(foreground, economic)
    lca = evaluate_lca(foreground, lca_factors)
    return ScenarioEvaluation(
        foreground=foreground,
        tea=tea,
        lca=lca,
        source_rows=flatten_record_tables(records),
    )


def run_baseline_cases(
    scenarios: Optional[Iterable[ScenarioConfig]] = None,
) -> List[ScenarioEvaluation]:
    scenario_list = list(scenarios or default_scenarios())
    return [evaluate_scenario(config) for config in scenario_list]


def run_single_case(
    category: ScenarioCategory,
    annual_primary_product_tpy: float,
    overrides: Optional[Dict[str, float]] = None,
) -> ScenarioEvaluation:
    config = ScenarioConfig(category=category, annual_primary_product_tpy=annual_primary_product_tpy)
    return evaluate_scenario(config, overrides=overrides)


def run_sensitivity_cases(
    config: ScenarioConfig,
    parameter: str,
    delta: float = 0.20,
) -> List[ScenarioEvaluation]:
    base = evaluate_scenario(config)
    source_rows = {row["key"]: row["value"] for row in base.source_rows}
    if parameter not in source_rows:
        raise KeyError(f"Unknown overrideable parameter: {parameter}")
    baseline = float(source_rows[parameter])
    low_overrides = dict(config.user_overrides)
    low_overrides[parameter] = baseline * (1.0 - delta)
    high_overrides = dict(config.user_overrides)
    high_overrides[parameter] = baseline * (1.0 + delta)
    low = evaluate_scenario(config, overrides=low_overrides)
    high = evaluate_scenario(config, overrides=high_overrides)
    return [low, base, high]


_NH3_METHODS = [
    AmmoniaRecoveryMethod.STRUVITE_MAP,
    AmmoniaRecoveryMethod.MEMBRANE,
    AmmoniaRecoveryMethod.MAP_FERTILIZER,
    AmmoniaRecoveryMethod.VACUUM_STRIPPING,
    AmmoniaRecoveryMethod.AIR_STRIPPING,
]

# Methods considered viable for the "best methods" NPV comparison figure
NH3_BEST_METHODS: List[AmmoniaRecoveryMethod] = [
    AmmoniaRecoveryMethod.STRUVITE_MAP,
    AmmoniaRecoveryMethod.MEMBRANE,
    AmmoniaRecoveryMethod.MAP_FERTILIZER,
]
UREA_BEST_METHODS: List[UreaRecoveryMethod] = [
    UreaRecoveryMethod.MVR_CRYSTALLIZATION,
    UreaRecoveryMethod.EVAPORATION,
]

FAVORABLE_LCA_SCENARIO_UPDATES: Dict[str, Any] = {
    "electricity_case": ElectricityCase.RENEWABLE,
    "co2_source": CO2Source.BIOGENIC_WASTE,
    "use_biogenic_carbon_credit": True,
    "use_scp_displacement_credit": True,
}

_UREA_METHODS = [
    UreaRecoveryMethod.MVR_CRYSTALLIZATION,
    UreaRecoveryMethod.EVAPORATION,
    UreaRecoveryMethod.HYBRID,
]

# Human-readable labels for each recovery method
RECOVERY_METHOD_LABELS: Dict[str, str] = {
    AmmoniaRecoveryMethod.STRUVITE_MAP.value:    "Struvite\n(MgNH\u2084PO\u2084\u00b76H\u2082O)",
    AmmoniaRecoveryMethod.MEMBRANE.value:        "Hollow-fiber membrane\n(NH\u2083 gas, ammonium soln.)",
    AmmoniaRecoveryMethod.MAP_FERTILIZER.value:  "MAP fertilizer route\n(NH\u2084H\u2082PO\u2084, 11-52-0)",
    AmmoniaRecoveryMethod.VACUUM_STRIPPING.value: "Vacuum stripping\n(liquid NH\u2083)",
    AmmoniaRecoveryMethod.AIR_STRIPPING.value:   "Air stripping\n(packed column)",
    UreaRecoveryMethod.MVR_CRYSTALLIZATION.value: "MVR + crystallization\n(electricity-driven)",
    UreaRecoveryMethod.EVAPORATION.value:         "Single-effect evap.\n+ crystallization",
    UreaRecoveryMethod.HYBRID.value:              "Membrane pre-conc.\n+ evaporation",
}


def run_best_methods_grid(
    capacities: Optional[List[float]] = None,
    overrides: Optional[Dict[str, float]] = None,
    scenario_updates: Optional[Dict[str, Any]] = None,
) -> List[ScenarioEvaluation]:
    """Evaluate the curated best NH3 and urea recovery methods across multiple capacities.

    Returns one ScenarioEvaluation per (method × capacity) combination, ordered
    by (category, method, capacity) for use in the NPV comparison figure.
    """
    caps = capacities or [100.0, 1_000.0, 10_000.0]
    scenario_updates = scenario_updates or {}
    results: List[ScenarioEvaluation] = []
    for method in NH3_BEST_METHODS:
        for cap in caps:
            cfg_kwargs: Dict[str, Any] = {
                "category": ScenarioCategory.AMMONIA_SCP,
                "annual_primary_product_tpy": cap,
                "ammonia_recovery_method": method,
            }
            cfg_kwargs.update(scenario_updates)
            cfg = ScenarioConfig(
                **cfg_kwargs,
            )
            results.append(evaluate_scenario(cfg, overrides=overrides))
    for method in UREA_BEST_METHODS:
        for cap in caps:
            cfg_kwargs = {
                "category": ScenarioCategory.BIO_UREA_SCP,
                "annual_primary_product_tpy": cap,
                "urea_recovery_method": method,
            }
            cfg_kwargs.update(scenario_updates)
            cfg = ScenarioConfig(**cfg_kwargs)
            results.append(evaluate_scenario(cfg, overrides=overrides))
    return results


def run_best_methods_negative_gwp_grid(
    capacities: Optional[List[float]] = None,
    overrides: Optional[Dict[str, float]] = None,
) -> List[ScenarioEvaluation]:
    """Evaluate the curated best-method set under favorable LCA assumptions.

    This helper mirrors the method families used in the NPV figure but switches to
    the attractive climate case: renewable electricity, biogenic waste CO2, carbon
    storage credit in SCP, and SCP displacement credit. Only scenarios that achieve
    net-negative GWP are returned so the figure focuses on genuinely compelling
    climate outcomes.
    """
    rows = run_best_methods_grid(
        capacities=capacities,
        overrides=overrides,
        scenario_updates=FAVORABLE_LCA_SCENARIO_UPDATES,
    )
    return [
        row for row in rows
        if row.lca.metrics["primary_product_gwp_kgco2e_per_kg"] < 0.0
    ]


def run_lca_sensitivity_grid(
    capacity_tpy: float = 1_000.0,
    overrides: Optional[Dict[str, float]] = None,
) -> List[ScenarioEvaluation]:
    """Evaluate each category across four LCA accounting scenarios at one capacity.

    The four scenarios capture the key axes of LCA uncertainty for this system:

    * **Grid + no biogenic credit**: most conservative — US average electricity,
      fossil-grade CO₂ burden, no biogenic-carbon or displacement credits.
    * **Grid + biogenic credit**: corrects CO₂ sourcing and credits carbon stored in SCP
      (appropriate design basis: corn-ethanol off-gas CO₂, US grid power).
    * **Renewable + biogenic credit**: design-basis electricity decarbonised.
    * **Renewable + biogenic + displacement**: adds system-expansion protein credit.

    Returns one evaluation per (category × scenario) combination for use in the
    GWP waterfall / comparison figure.
    """
    configs = []
    for cat in (ScenarioCategory.AMMONIA_SCP, ScenarioCategory.BIO_UREA_SCP):
        rec = AmmoniaRecoveryMethod.MEMBRANE if cat == ScenarioCategory.AMMONIA_SCP else None
        ure = UreaRecoveryMethod.EVAPORATION if cat == ScenarioCategory.BIO_UREA_SCP else None
        base = dict(
            category=cat,
            annual_primary_product_tpy=capacity_tpy,
            ammonia_recovery_method=rec or AmmoniaRecoveryMethod.MEMBRANE,
            urea_recovery_method=ure or UreaRecoveryMethod.EVAPORATION,
        )
        configs.extend([
            ScenarioConfig(**base, electricity_case=ElectricityCase.US_GRID,
                           co2_source=CO2Source.FOSSIL_CAPTURE,
                           use_biogenic_carbon_credit=False,
                           use_scp_displacement_credit=False),
            ScenarioConfig(**base, electricity_case=ElectricityCase.US_GRID,
                           co2_source=CO2Source.BIOGENIC_WASTE,
                           use_biogenic_carbon_credit=True,
                           use_scp_displacement_credit=False),
            ScenarioConfig(**base, electricity_case=ElectricityCase.RENEWABLE,
                           co2_source=CO2Source.BIOGENIC_WASTE,
                           use_biogenic_carbon_credit=True,
                           use_scp_displacement_credit=False),
            ScenarioConfig(**base, electricity_case=ElectricityCase.RENEWABLE,
                           co2_source=CO2Source.BIOGENIC_WASTE,
                           use_biogenic_carbon_credit=True,
                           use_scp_displacement_credit=True),
        ])
    return [evaluate_scenario(cfg, overrides=overrides) for cfg in configs]


def run_recovery_comparison(
    category: ScenarioCategory,
    annual_primary_product_tpy: float,
    overrides: Optional[Dict[str, float]] = None,
) -> List[ScenarioEvaluation]:
    """Evaluate every recovery method for a given category and capacity.

    Returns one ScenarioEvaluation per method, in the order defined by
    _NH3_METHODS / _UREA_METHODS.
    """
    results: List[ScenarioEvaluation] = []
    if category == ScenarioCategory.AMMONIA_SCP:
        for method in _NH3_METHODS:
            cfg = ScenarioConfig(
                category=category,
                annual_primary_product_tpy=annual_primary_product_tpy,
                ammonia_recovery_method=method,
            )
            results.append(evaluate_scenario(cfg, overrides=overrides))
    else:
        for method in _UREA_METHODS:
            cfg = ScenarioConfig(
                category=category,
                annual_primary_product_tpy=annual_primary_product_tpy,
                urea_recovery_method=method,
            )
            results.append(evaluate_scenario(cfg, overrides=overrides))
    return results


def format_summary(evaluations: Iterable[ScenarioEvaluation]) -> str:
    rows = list(evaluations)
    header = (
        f"{'Scenario':<14} {'Cap_tpy':>9} {'LCOX_gross':>12} "
        f"{'LCOX_net':>12} {'GWP_kgCO2e/kg':>16} {'NPV_MUSD':>10}"
    )
    lines = [header, "-" * len(header)]
    for evaluation in rows:
        data = evaluation.to_dict()
        lines.append(
            f"{data['scenario']:<14} {data['annual_primary_product_tpy']:>9,.0f} "
            f"{data['gross_primary_lcox_usd_per_kg']:>12.3f} "
            f"{data['net_primary_lcox_usd_per_kg']:>12.3f} "
            f"{data['primary_product_gwp_kgco2e_per_kg']:>16.3f} "
            f"{data['npv_usd'] / 1e6:>10.2f}"
        )
    return "\n".join(lines)
