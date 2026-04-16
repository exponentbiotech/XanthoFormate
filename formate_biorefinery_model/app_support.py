from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .config import (
    AmmoniaRecoveryMethod,
    CO2Source,
    ElectricityCase,
    FeedstockType,
    ScenarioCategory,
    ScenarioConfig,
    UreaRecoveryMethod,
    build_default_inputs,
    load_source_metadata,
    records_by_key,
)
from .reporting import load_figure_metadata
from .run_scenarios import ScenarioEvaluation, evaluate_scenario


@dataclass(frozen=True)
class SliderSpec:
    key: str
    label: str
    group: str
    min_value: float
    max_value: float
    step: float
    help_text: str
    categories: Tuple[ScenarioCategory, ...] = ()
    unit: str = ""
    feedstock_types: Tuple[FeedstockType, ...] = ()  # empty = shown for all feedstocks
    is_override_key: bool = True  # False for special keys like major_capex_usd


SLIDER_SPECS: Tuple[SliderSpec, ...] = (
    # ── Formate feedstock biological performance ──
    SliderSpec(
        "formate_to_ammonia_kg_per_kg",
        "Formate required per kg NH\u2083",
        "Biological performance",
        4.0, 20.0, 0.1,
        "Biological and electrochemical efficiency proxy for the ammonia route.",
        (ScenarioCategory.AMMONIA_SCP,),
        "kg formate / kg NH\u2083",
        (FeedstockType.FORMATE,),
    ),
    SliderSpec(
        "scp_to_ammonia_kg_per_kg",
        "SCP produced per kg NH\u2083",
        "Biological performance",
        0.5, 8.0, 0.1,
        "Dry SCP coproduct yield in the ammonia route (formate).",
        (ScenarioCategory.AMMONIA_SCP,),
        "kg SCP / kg NH\u2083",
        (FeedstockType.FORMATE,),
    ),
    SliderSpec(
        "ammonia_productivity_kg_per_m3_h",
        "Ammonia productivity",
        "Biological performance",
        0.05, 2.0, 0.01,
        "Fermenter volumetric productivity for the ammonia case (formate).",
        (ScenarioCategory.AMMONIA_SCP,),
        "kg / m\u00b3\u00b7h",
        (FeedstockType.FORMATE,),
    ),
    SliderSpec(
        "ammonia_recovery_efficiency",
        "NH\u2083 recovery efficiency",
        "Biological performance",
        0.50, 0.99, 0.01,
        "Default recovery yield used in the generic ammonia route.",
        (ScenarioCategory.AMMONIA_SCP,),
        "fraction (0\u20131)",
    ),
    SliderSpec(
        "formate_to_urea_kg_per_kg",
        "Formate required per kg urea",
        "Biological performance",
        3.0, 12.0, 0.1,
        "Biological and electrochemical efficiency proxy for the bio-urea route.",
        (ScenarioCategory.BIO_UREA_SCP,),
        "kg formate / kg urea",
        (FeedstockType.FORMATE,),
    ),
    SliderSpec(
        "scp_to_urea_kg_per_kg",
        "SCP produced per kg urea",
        "Biological performance",
        0.5, 5.0, 0.1,
        "Dry SCP coproduct yield in the urea route (formate).",
        (ScenarioCategory.BIO_UREA_SCP,),
        "kg SCP / kg urea",
        (FeedstockType.FORMATE,),
    ),
    SliderSpec(
        "urea_productivity_kg_per_m3_h",
        "Urea productivity",
        "Biological performance",
        0.05, 1.5, 0.01,
        "Fermenter volumetric productivity for the urea case (formate).",
        (ScenarioCategory.BIO_UREA_SCP,),
        "kg / m\u00b3\u00b7h",
        (FeedstockType.FORMATE,),
    ),
    SliderSpec(
        "urea_recovery_efficiency",
        "Urea recovery efficiency",
        "Biological performance",
        0.50, 0.99, 0.01,
        "Generic urea recovery yield used in the screening model.",
        (ScenarioCategory.BIO_UREA_SCP,),
        "fraction (0\u20131)",
    ),
    # ── H2/CO2 feedstock biological performance ──
    SliderSpec(
        "h2_to_ammonia_kg_per_kg",
        "H\u2082 required per kg NH\u2083",
        "Biological performance",
        1.0, 8.0, 0.1,
        "H2 demand for autotrophic NH3 production (H2/CO2 path).",
        (ScenarioCategory.AMMONIA_SCP,),
        "kg H\u2082 / kg NH\u2083",
        (FeedstockType.H2_CO2,),
    ),
    SliderSpec(
        "h2_to_urea_kg_per_kg",
        "H\u2082 required per kg urea",
        "Biological performance",
        0.5, 5.0, 0.1,
        "H2 demand for autotrophic urea production (H2/CO2 path).",
        (ScenarioCategory.BIO_UREA_SCP,),
        "kg H\u2082 / kg urea",
        (FeedstockType.H2_CO2,),
    ),
    SliderSpec(
        "scp_to_ammonia_h2co2_kg_per_kg",
        "SCP per kg NH\u2083 (H\u2082/CO\u2082)",
        "Biological performance",
        0.5, 8.0, 0.1,
        "Dry SCP coproduct yield on H2/CO2 autotrophic path.",
        (ScenarioCategory.AMMONIA_SCP,),
        "kg SCP / kg NH\u2083",
        (FeedstockType.H2_CO2,),
    ),
    SliderSpec(
        "scp_to_urea_h2co2_kg_per_kg",
        "SCP per kg urea (H\u2082/CO\u2082)",
        "Biological performance",
        0.5, 5.0, 0.1,
        "Dry SCP coproduct yield on H2/CO2 autotrophic path (urea).",
        (ScenarioCategory.BIO_UREA_SCP,),
        "kg SCP / kg urea",
        (FeedstockType.H2_CO2,),
    ),
    SliderSpec(
        "ammonia_productivity_h2co2_kg_per_m3_h",
        "NH\u2083 productivity (H\u2082/CO\u2082)",
        "Biological performance",
        0.05, 2.0, 0.01,
        "Fermenter volumetric productivity on H2/CO2 (ammonia).",
        (ScenarioCategory.AMMONIA_SCP,),
        "kg / m\u00b3\u00b7h",
        (FeedstockType.H2_CO2,),
    ),
    SliderSpec(
        "urea_productivity_h2co2_kg_per_m3_h",
        "Urea productivity (H\u2082/CO\u2082)",
        "Biological performance",
        0.05, 1.5, 0.01,
        "Fermenter volumetric productivity on H2/CO2 (urea).",
        (ScenarioCategory.BIO_UREA_SCP,),
        "kg / m\u00b3\u00b7h",
        (FeedstockType.H2_CO2,),
    ),
    # ── Methanol feedstock biological performance ──
    SliderSpec(
        "methanol_to_ammonia_kg_per_kg",
        "Methanol per kg NH\u2083",
        "Biological performance",
        4.0, 20.0, 0.1,
        "Methanol demand for methylotrophic NH3 production.",
        (ScenarioCategory.AMMONIA_SCP,),
        "kg MeOH / kg NH\u2083",
        (FeedstockType.METHANOL,),
    ),
    SliderSpec(
        "methanol_to_urea_kg_per_kg",
        "Methanol per kg urea",
        "Biological performance",
        2.0, 12.0, 0.1,
        "Methanol demand for methylotrophic urea production.",
        (ScenarioCategory.BIO_UREA_SCP,),
        "kg MeOH / kg urea",
        (FeedstockType.METHANOL,),
    ),
    SliderSpec(
        "scp_to_ammonia_methanol_kg_per_kg",
        "SCP per kg NH\u2083 (methanol)",
        "Biological performance",
        0.5, 8.0, 0.1,
        "Dry SCP coproduct yield on methanol path.",
        (ScenarioCategory.AMMONIA_SCP,),
        "kg SCP / kg NH\u2083",
        (FeedstockType.METHANOL,),
    ),
    SliderSpec(
        "scp_to_urea_methanol_kg_per_kg",
        "SCP per kg urea (methanol)",
        "Biological performance",
        0.5, 5.0, 0.1,
        "Dry SCP coproduct yield on methanol path (urea).",
        (ScenarioCategory.BIO_UREA_SCP,),
        "kg SCP / kg urea",
        (FeedstockType.METHANOL,),
    ),
    SliderSpec(
        "ammonia_productivity_methanol_kg_per_m3_h",
        "NH\u2083 productivity (methanol)",
        "Biological performance",
        0.05, 2.0, 0.01,
        "Fermenter volumetric productivity on methanol (ammonia).",
        (ScenarioCategory.AMMONIA_SCP,),
        "kg / m\u00b3\u00b7h",
        (FeedstockType.METHANOL,),
    ),
    SliderSpec(
        "urea_productivity_methanol_kg_per_m3_h",
        "Urea productivity (methanol)",
        "Biological performance",
        0.05, 1.5, 0.01,
        "Fermenter volumetric productivity on methanol (urea).",
        (ScenarioCategory.BIO_UREA_SCP,),
        "kg / m\u00b3\u00b7h",
        (FeedstockType.METHANOL,),
    ),
    SliderSpec(
        "agitation_aeration_kwh_per_m3_h",
        "Agitation + aeration intensity",
        "Biological performance",
        0.10,
        2.50,
        0.01,
        "Shared fermentation electricity intensity.",
        (),
        "kWh / m\u00b3\u00b7h",
    ),
    SliderSpec(
        "scp_drying_kwh_per_kg_water",
        "SCP drying energy",
        "Biological performance",
        0.25,
        3.00,
        0.05,
        "Drying load for SCP finishing.",
        (),
        "kWh / kg water",
    ),
    SliderSpec(
        "electricity_price_usd_per_kwh",
        "Electricity price",
        "Cost of inputs",
        0.02,
        0.30,
        0.002,
        "Industrial electricity cost used in TEA.",
        (),
        "$ / kWh",
    ),
    SliderSpec(
        "co2_price_usd_per_kg",
        "Delivered CO\u2082 price",
        "Cost of inputs",
        0.00, 0.30, 0.005,
        "Merchant or site-specific CO2 feed price.",
        (), "$ / kg",
        (FeedstockType.FORMATE, FeedstockType.H2_CO2),
    ),
    SliderSpec(
        "methanol_price_usd_per_kg",
        "Methanol price",
        "Cost of inputs",
        0.10, 1.00, 0.01,
        "Industrial methanol commodity price.",
        (), "$ / kg",
        (FeedstockType.METHANOL,),
    ),
    SliderSpec(
        "water_price_usd_per_kg",
        "Process water price",
        "Cost of inputs",
        0.0001,
        0.01,
        0.0001,
        "Water supply cost for TEA.",
        (),
        "$ / kg",
    ),
    SliderSpec(
        "wastewater_price_usd_per_kg",
        "Wastewater handling price",
        "Cost of inputs",
        0.0001,
        0.01,
        0.0001,
        "Wastewater treatment and sewer cost proxy.",
        (),
        "$ / kg",
    ),
    SliderSpec(
        "steam_price_usd_per_kg",
        "Steam price",
        "Cost of inputs",
        0.001,
        0.10,
        0.001,
        "Low-pressure steam price for urea recovery and utilities.",
        (),
        "$ / kg",
    ),
    SliderSpec(
        "naoh_price_usd_per_kg",
        "NaOH price",
        "Cost of inputs",
        0.10,
        2.00,
        0.01,
        "Caustic price used in ammonia recovery scenarios.",
        (),
        "$ / kg",
    ),
    SliderSpec(
        "h3po4_price_usd_per_kg",
        "H\u2083PO\u2084 price",
        "Cost of inputs",
        0.10,
        2.00,
        0.01,
        "Phosphoric acid price used in precipitation and MAP routes.",
        (),
        "$ / kg",
    ),
    SliderSpec(
        "mgcl2_price_usd_per_kg",
        "MgCl\u2082 price",
        "Cost of inputs",
        0.05,
        1.00,
        0.01,
        "Magnesium chloride price for struvite precipitation.",
        (),
        "$ / kg",
    ),
    SliderSpec(
        "membrane_cost_usd_per_m2",
        "Membrane cost",
        "Cost of inputs",
        10.0,
        500.0,
        5.0,
        "Membrane replacement cost used in TEA.",
        (),
        "$ / m\u00b2",
    ),
    SliderSpec(
        "major_capex_usd",
        "Major CapEx (user-defined)",
        "Financing",
        0.0, 10_000_000.0, 25_000.0,
        "Optional lump-sum major equipment cost added to the model. Items >$100K are excluded from minor CapEx by default.",
        (), "USD",
        (), False,
    ),
    SliderSpec(
        "discount_rate",
        "Discount rate",
        "Financing",
        0.02, 0.30, 0.005,
        "Nominal discount rate used for annualization and NPV.",
        (), "fraction (0\u20131)",
    ),
    SliderSpec(
        "plant_life_years",
        "Plant life",
        "Financing",
        5.0,
        40.0,
        0.5,
        "Operating life used for capital recovery and NPV.",
        (),
        "years",
    ),
    SliderSpec(
        "working_capital_fraction",
        "Working capital fraction",
        "Financing",
        0.02,
        0.40,
        0.01,
        "Working capital as a fraction of fixed capital.",
        (),
        "fraction (0\u20131)",
    ),
    SliderSpec(
        "lang_factor",
        "Lang factor",
        "Financing",
        1.5,
        6.0,
        0.1,
        "Installed-cost multiplier for purchased equipment.",
        (),
        "dimensionless",
    ),
    SliderSpec(
        "electrolyzer_installed_cost_usd_per_kw",
        "Electrolyzer installed cost",
        "Financing",
        200.0, 3000.0, 25.0,
        "Installed PEM electrolyzer cost basis.",
        (), "$ / kW",
        (FeedstockType.FORMATE, FeedstockType.H2_CO2),
    ),
    SliderSpec(
        "electrolyzer_stack_replacement_fraction",
        "Stack replacement fraction",
        "Financing",
        0.05, 0.60, 0.01,
        "Fraction of installed electrolyzer cost paid at stack replacement.",
        (), "fraction (0\u20131)",
        (FeedstockType.FORMATE, FeedstockType.H2_CO2),
    ),
    SliderSpec(
        "electrolyzer_stack_life_years",
        "Stack life",
        "Financing",
        1.0, 20.0, 0.5,
        "Replacement interval for the electrolyzer stack.",
        (), "years",
        (FeedstockType.FORMATE, FeedstockType.H2_CO2),
    ),
    SliderSpec(
        "maintenance_factor",
        "Maintenance factor",
        "Labor",
        0.005,
        0.10,
        0.001,
        "Annual maintenance as a fraction of fixed capital.",
        (),
        "fraction of fixed capital",
    ),
    SliderSpec(
        "operator_loaded_wage_usd_per_h",
        "Loaded operator wage",
        "Labor",
        10.0,
        120.0,
        1.0,
        "Fully loaded hourly operator wage.",
        (),
        "$ / hr",
    ),
    SliderSpec(
        "admin_overhead_factor",
        "Admin + overhead factor",
        "Labor",
        0.05,
        0.80,
        0.01,
        "Supervision and overhead fraction of direct labor.",
        (),
        "fraction (0\u20131)",
    ),
)


def slider_specs_for(
    category: ScenarioCategory,
    feedstock_type: FeedstockType = FeedstockType.FORMATE,
) -> Dict[str, List[SliderSpec]]:
    groups: Dict[str, List[SliderSpec]] = {}
    for spec in SLIDER_SPECS:
        if spec.categories and category not in spec.categories:
            continue
        if spec.feedstock_types and feedstock_type not in spec.feedstock_types:
            continue
        groups.setdefault(spec.group, []).append(spec)
    return groups


def all_slider_specs(
    feedstock_type: FeedstockType = FeedstockType.FORMATE,
) -> Dict[str, List[SliderSpec]]:
    groups: Dict[str, List[SliderSpec]] = {}
    for spec in SLIDER_SPECS:
        if spec.feedstock_types and feedstock_type not in spec.feedstock_types:
            continue
        groups.setdefault(spec.group, []).append(spec)
    return groups


def default_input_records() -> Dict[str, Dict[str, object]]:
    _, _, _, datasets = build_default_inputs()
    return records_by_key(datasets)


def slider_defaults(
    category: ScenarioCategory,
    feedstock_type: FeedstockType = FeedstockType.FORMATE,
) -> Dict[str, float]:
    defaults = default_input_records()
    out: Dict[str, float] = {}
    for specs in slider_specs_for(category, feedstock_type=feedstock_type).values():
        for spec in specs:
            if not spec.is_override_key:
                out[spec.key] = spec.min_value
            else:
                out[spec.key] = float(defaults[spec.key]["value"])
    return out


def all_slider_defaults(
    feedstock_type: FeedstockType = FeedstockType.FORMATE,
) -> Dict[str, float]:
    defaults = default_input_records()
    out: Dict[str, float] = {}
    for specs in all_slider_specs(feedstock_type=feedstock_type).values():
        for spec in specs:
            if not spec.is_override_key:
                out[spec.key] = spec.min_value
            else:
                out[spec.key] = float(defaults[spec.key]["value"])
    return out


def normalize_overrides(values: Mapping[str, float]) -> Dict[str, float]:
    return {key: float(value) for key, value in values.items()}


def dashboard_scenarios(
    base_config: ScenarioConfig,
    nh3_method: AmmoniaRecoveryMethod,
    urea_method: UreaRecoveryMethod,
    capacities: Sequence[float] = (100.0, 1_000.0, 10_000.0),
) -> List[ScenarioConfig]:
    shared = {
        "feedstock_type": base_config.feedstock_type,
        "electricity_case": base_config.electricity_case,
        "use_scp_credit": base_config.use_scp_credit,
        "use_h2_credit": base_config.use_h2_credit,
        "h2_credit_usd_per_kg": base_config.h2_credit_usd_per_kg,
        "co2_credit_usd_per_kg": base_config.co2_credit_usd_per_kg,
        "co2_source": base_config.co2_source,
        "use_biogenic_carbon_credit": base_config.use_biogenic_carbon_credit,
        "use_scp_displacement_credit": base_config.use_scp_displacement_credit,
    }
    scenarios: List[ScenarioConfig] = []
    for capacity in capacities:
        scenarios.append(
            ScenarioConfig(
                category=ScenarioCategory.AMMONIA_SCP,
                annual_primary_product_tpy=float(capacity),
                ammonia_recovery_method=nh3_method,
                **shared,
            )
        )
        scenarios.append(
            ScenarioConfig(
                category=ScenarioCategory.BIO_UREA_SCP,
                annual_primary_product_tpy=float(capacity),
                urea_recovery_method=urea_method,
                **shared,
            )
        )
    return scenarios


def evaluate_dashboard_grid(
    base_config: ScenarioConfig,
    overrides: Mapping[str, float],
    nh3_method: AmmoniaRecoveryMethod,
    urea_method: UreaRecoveryMethod,
) -> List[ScenarioEvaluation]:
    scenarios = dashboard_scenarios(base_config, nh3_method=nh3_method, urea_method=urea_method)
    return [evaluate_scenario(cfg, overrides=normalize_overrides(overrides)) for cfg in scenarios]


def current_config_summary(config: ScenarioConfig) -> Dict[str, object]:
    return {
        "category": config.category.value,
        "feedstock_type": config.feedstock_type.value,
        "annual_primary_product_tpy": float(config.annual_primary_product_tpy),
        "electricity_case": config.electricity_case.value,
        "co2_source": config.co2_source.value,
        "ammonia_recovery_method": config.ammonia_recovery_method.value,
        "urea_recovery_method": config.urea_recovery_method.value,
        "use_scp_credit": config.use_scp_credit,
        "use_biogenic_carbon_credit": config.use_biogenic_carbon_credit,
        "use_scp_displacement_credit": config.use_scp_displacement_credit,
    }


def kpi_cards(evaluation: ScenarioEvaluation) -> List[Dict[str, str]]:
    metrics = evaluation.tea.metrics
    lca_metrics = evaluation.lca.metrics
    excluded_total = sum(evaluation.tea.capex_excluded_usd.values())
    excluded_note = f" (excl. ${excluded_total/1e6:.1f}M major)" if excluded_total > 0 else ""
    return [
        {
            "label": "Net LCOX",
            "value": f"${metrics['net_primary_lcox_usd_per_kg']:.2f}/kg",
        },
        {
            "label": "Gross LCOX",
            "value": f"${metrics['gross_primary_lcox_usd_per_kg']:.2f}/kg",
        },
        {
            "label": "Net GWP",
            "value": f"{lca_metrics['primary_product_gwp_kgco2e_per_kg']:.2f} kg CO2e/kg",
        },
        {
            "label": f"NPV{excluded_note}",
            "value": f"${metrics['npv_usd'] / 1e6:.1f}M",
        },
        {
            "label": "Annual SCP",
            "value": f"{evaluation.foreground.annual_scp_kg / 1000.0:,.0f} t/y",
        },
        {
            "label": "Working volume",
            "value": f"{evaluation.foreground.working_volume_m3:,.0f} m3",
        },
    ]


def source_rows_for_display(rows: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    ordered = sorted(rows, key=lambda row: (str(row["dataset"]), str(row["key"])))
    out: List[Dict[str, object]] = []
    for row in ordered:
        out.append(
            {
                "dataset": row["dataset"],
                "parameter": row["key"],
                "value": float(row["value"]),
                "unit": row["unit"],
                "source": row["source"],
                "source_url": row.get("source_url", ""),
                "year": row["year"],
                "confidence": row["confidence"],
                "overridden": bool(row.get("is_override", False)),
                "notes": row["notes"],
            }
        )
    return out


def reference_summary(rows: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in rows:
        key = (str(row["source"]), str(row.get("source_url", "")))
        if key not in grouped:
            grouped[key] = {
                "source": row["source"],
                "source_url": row.get("source_url", ""),
                "datasets": set(),
                "facts": [],
                "overrides": 0,
            }
        grouped[key]["datasets"].add(row["dataset"])
        grouped[key]["facts"].append(f"{row['key']} = {row['value']} {row['unit']}".strip())
        if row.get("is_override", False):
            grouped[key]["overrides"] += 1

    out: List[Dict[str, object]] = []
    for item in grouped.values():
        facts = sorted(item["facts"])
        out.append(
            {
                "source": item["source"],
                "source_url": item["source_url"],
                "datasets": ", ".join(sorted(item["datasets"])),
                "facts_used": "; ".join(facts[:12]) + (" ..." if len(facts) > 12 else ""),
                "fact_count": float(len(facts)),
                "override_count": float(item["overrides"]),
            }
        )
    out.sort(key=lambda row: (row["source"] == "user_override", row["source"]))
    return out


def figure_ids() -> List[str]:
    return list(load_figure_metadata().keys())


def figure_description_rows() -> List[Dict[str, object]]:
    metadata = load_figure_metadata()
    rows: List[Dict[str, object]] = []
    for key, item in metadata.items():
        rows.append(
            {
                "figure_id": key,
                "title": item["title"],
                "what_is_plotted": item["what_is_plotted"],
                "calculation_basis": item["calculation_basis"],
                "important_assumptions": item["important_assumptions"],
                "interpretation_guidance": item["interpretation_guidance"],
            }
        )
    return rows


def app_context_snapshot(
    evaluation: ScenarioEvaluation,
    source_rows: Sequence[Mapping[str, object]],
    active_figure_ids: Sequence[str],
) -> Dict[str, object]:
    metadata = load_figure_metadata()
    return {
        "source_metadata": load_source_metadata(),
        "scenario": current_config_summary(evaluation.foreground.scenario),
        "kpis": evaluation.to_dict(),
        "tea_metrics": evaluation.tea.metrics,
        "lca_metrics": evaluation.lca.metrics,
        "active_figures": {
            figure_id: metadata[figure_id]
            for figure_id in active_figure_ids
            if figure_id in metadata
        },
        "source_rows": source_rows_for_display(source_rows),
    }
