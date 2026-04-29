from __future__ import annotations

import json
from dataclasses import dataclass, replace
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


_METRIC_FIELDS = (
    "gross_lcox_usd_per_kg",
    "net_lcox_usd_per_kg",
    "npv_usd_million",
    "primary_product_gwp_kgco2e_per_kg",
)


# ─────────────────────────────────────────────────────────────────────────────
#  Plain-English translation tables for the chat/model-interpreter snapshot.
#  Earlier LLM chat versions echoed snake_case identifiers (e.g.
#  "npv_usd_million", "ammonia_scp", "struvite_map") back to users. Keeping the
#  snapshot human-readable also makes deterministic interpreter output clearer.
# ─────────────────────────────────────────────────────────────────────────────

# Snake-case categorical IDs → friendly labels. Strings are unique across the
# vocabulary, so a single flat map is sufficient.
_VALUE_LABELS: Dict[str, str] = {
    # Categories / pathways
    "ammonia_scp": "Ammonia + SCP",
    "bio_urea_scp": "Urea + SCP",
    # Feedstocks
    "formate": "Formate (CO2 electrolysis)",
    "h2_co2": "H2/CO2 (water electrolysis)",
    "methanol": "Methanol (purchased)",
    # Electricity cases
    "us_grid": "US grid average",
    "renewable": "Renewable (wind/solar PPA)",
    # CO2 sources
    "biogenic_waste": "Biogenic waste CO2 (corn-ethanol off-gas)",
    "dac": "Direct air capture (DAC)",
    "fossil_capture": "Fossil-fuel CCS",
    # NH3 recovery methods
    "struvite_map": "Struvite (MgNH4PO4)",
    "membrane": "Hollow-fiber membrane",
    "vacuum_stripping": "Vacuum stripping",
    "air_stripping": "Air stripping",
    "map_fertilizer": "MAP fertilizer (NH4H2PO4)",
    # Urea recovery methods
    "mvr_crystallization": "MVR + crystallization",
    "evaporation": "Single-effect evaporation",
    "evaporation_crystallization": "Single-effect evaporation + crystallization",
    "hybrid": "Membrane pre-concentration + evaporation",
    "membrane_polishing": "Membrane pre-concentration + evaporation",
    # Primary product names (from to_dict)
    "ammonia": "Ammonia",
    "urea": "Urea",
}

# Internal field names → friendly labels (with units). Both _compact_eval and
# ScenarioEvaluation.to_dict styles are listed because they coexist in the
# snapshot.
_FIELD_RENAME: Dict[str, str] = {
    "scenario": "Pathway",
    "category": "Pathway",
    "feedstock": "Feedstock",
    "feedstock_type": "Feedstock",
    "capacity_tpy": "Plant capacity (t/y)",
    "annual_primary_product_tpy": "Plant capacity (t/y)",
    "electricity_case": "Electricity case",
    "co2_source": "CO2 source",
    "nh3_recovery_method": "NH3 recovery method",
    "ammonia_recovery_method": "NH3 recovery method",
    "urea_recovery_method": "Urea recovery method",
    "use_scp_credit": "SCP co-product credit enabled",
    "use_biogenic_carbon_credit": "Biogenic carbon credit enabled",
    "use_scp_displacement_credit": "Protein displacement credit enabled",
    "primary_product_name": "Primary product",
    "sellable_primary_product_kg_per_y": "Annual primary product (kg/y)",
    "annual_scp_kg_per_y": "Annual SCP (kg/y)",
    "working_volume_m3": "Working volume (m3)",
    "gross_primary_lcox_usd_per_kg": "Gross LCOX (USD/kg)",
    "net_primary_lcox_usd_per_kg": "Net LCOX (USD/kg)",
    "gross_lcox_usd_per_kg": "Gross LCOX (USD/kg)",
    "net_lcox_usd_per_kg": "Net LCOX (USD/kg)",
    "npv_usd": "NPV (USD)",
    "npv_usd_million": "NPV (million USD)",
    "primary_product_gwp_kgco2e_per_kg": "GWP (kg CO2e per kg product)",
    "combined_output_gwp_kgco2e_per_kg": "Combined output GWP (kg CO2e/kg)",
    "gwp_kgco2e_per_kg": "GWP (kg CO2e per kg product)",
}


def _friendlify_winner_id(id_str: str) -> str:
    """Translate the pipe-separated winner ID into plain English."""
    parts = id_str.split("|")
    if len(parts) < 5:
        return id_str
    cat, feedstock, cap, elec, method = parts[:5]
    cat_lbl = _VALUE_LABELS.get(cat, cat)
    feed_lbl = _VALUE_LABELS.get(feedstock, feedstock)
    elec_lbl = _VALUE_LABELS.get(elec, elec)
    if method.startswith("nh3="):
        m_lbl = "NH3 recovery: " + _VALUE_LABELS.get(method[4:], method[4:])
    elif method.startswith("urea="):
        m_lbl = "Urea recovery: " + _VALUE_LABELS.get(method[5:], method[5:])
    else:
        m_lbl = method
    return (
        f"{cat_lbl} | feedstock: {feed_lbl} | capacity: {cap} | "
        f"electricity: {elec_lbl} | {m_lbl}"
    )


def _friendlify(obj: object) -> object:
    """Recursively rename keys and translate categorical values to plain English."""
    if isinstance(obj, list):
        return [_friendlify(item) for item in obj]
    if isinstance(obj, dict):
        out: Dict[str, object] = {}
        for k, v in obj.items():
            new_key = _FIELD_RENAME.get(k, k)
            if k == "id" and isinstance(v, str):
                v = _friendlify_winner_id(v)
            elif isinstance(v, str) and v in _VALUE_LABELS:
                v = _VALUE_LABELS[v]
            else:
                v = _friendlify(v)
            out[new_key] = v
        return out
    return obj


def _compact_eval(eval_result: ScenarioEvaluation) -> Dict[str, object]:
    """Compact JSON-friendly summary of a scenario evaluation for chat context."""
    cfg = eval_result.foreground.scenario
    m = eval_result.tea.metrics
    l = eval_result.lca.metrics
    return {
        "category": cfg.category.value,
        "feedstock": cfg.feedstock_type.value,
        "capacity_tpy": float(cfg.annual_primary_product_tpy),
        "electricity_case": cfg.electricity_case.value,
        "co2_source": cfg.co2_source.value,
        "nh3_recovery_method": cfg.ammonia_recovery_method.value,
        "urea_recovery_method": cfg.urea_recovery_method.value,
        "use_scp_credit": bool(cfg.use_scp_credit),
        "use_biogenic_carbon_credit": bool(cfg.use_biogenic_carbon_credit),
        "use_scp_displacement_credit": bool(cfg.use_scp_displacement_credit),
        "gross_lcox_usd_per_kg": round(float(m["gross_primary_lcox_usd_per_kg"]), 3),
        "net_lcox_usd_per_kg": round(float(m["net_primary_lcox_usd_per_kg"]), 3),
        "npv_usd_million": round(float(m["npv_usd"]) / 1e6, 2),
        "primary_product_gwp_kgco2e_per_kg": round(float(l["primary_product_gwp_kgco2e_per_kg"]), 3),
    }


def _strip_constant_fields(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Drop fields that are identical across every row.

    The chat interpreter only needs fields that VARY within a comparison list. Stripping
    constants shaves substantial tokens off the prompt (e.g. urea_recovery_method
    is always 'evaporation' inside the NH3 method list, so it's noise).
    Always preserves the metric fields and the scenario category.
    """
    if not rows:
        return rows
    keys = list(rows[0].keys())
    constant_keys = {
        k for k in keys
        if k not in _METRIC_FIELDS and k != "category"
        and len({json.dumps(row.get(k), sort_keys=True, default=str) for row in rows}) <= 1
    }
    return [{k: v for k, v in row.items() if k not in constant_keys} for row in rows]


def _best_row(
    rows: Sequence[Dict[str, object]],
    metric: str,
    *,
    maximize: bool = False,
) -> Dict[str, object]:
    """Return the best row for a metric using deterministic Python ranking."""
    best = max(rows, key=lambda row: float(row[metric])) if maximize else min(rows, key=lambda row: float(row[metric]))
    return dict(best)


def _winner_ref(row: Dict[str, object]) -> str:
    """Compact one-line summary of a winning row for chat context.

    Returning a single string (rather than a nested dict with friendly keys)
    cuts ~60% of the token cost for the rankings block while still giving
    the interpreter everything it needs to cite the configuration and numbers.
    """
    category = str(row.get("category", ""))
    cat_lbl = _VALUE_LABELS.get(category, category)
    feed_lbl = _VALUE_LABELS.get(str(row.get("feedstock", "")), str(row.get("feedstock", "")))
    elec_lbl = _VALUE_LABELS.get(str(row.get("electricity_case", "")), str(row.get("electricity_case", "")))
    if category == ScenarioCategory.AMMONIA_SCP.value:
        rec_id = str(row.get("nh3_recovery_method", ""))
        rec_lbl = "NH3 recovery: " + _VALUE_LABELS.get(rec_id, rec_id)
    else:
        rec_id = str(row.get("urea_recovery_method", ""))
        rec_lbl = "Urea recovery: " + _VALUE_LABELS.get(rec_id, rec_id)
    cap = row.get("capacity_tpy")
    npv = row.get("npv_usd_million")
    lcox = row.get("net_lcox_usd_per_kg")
    gwp = row.get("primary_product_gwp_kgco2e_per_kg")
    return (
        f"{cat_lbl}, {feed_lbl}, {rec_lbl}, {cap:.0f} t/y, {elec_lbl} — "
        f"NPV USD {npv:.1f}M, Net LCOX USD {lcox:.2f}/kg, GWP {gwp:.2f} kgCO2e/kg"
    )


def _winner(
    rows: Sequence[Dict[str, object]],
    metric: str,
    *,
    maximize: bool = False,
) -> Dict[str, object]:
    return _winner_ref(_best_row(rows, metric, maximize=maximize))


def _ranking_summary(
    nh3_methods: Sequence[Dict[str, object]],
    urea_methods: Sequence[Dict[str, object]],
    feedstock_variants: Sequence[Dict[str, object]],
    electricity_variants: Sequence[Dict[str, object]],
    capacity_curve: Sequence[Dict[str, object]],
    credit_sensitivity: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Precompute common "which is best?" answers from Python model rows."""
    all_current_capacity_modes = list(nh3_methods) + list(urea_methods)

    def _block(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
        return {
            "Highest NPV": _winner(rows, "npv_usd_million", maximize=True),
            "Lowest Net LCOX": _winner(rows, "net_lcox_usd_per_kg"),
            "Lowest GWP": _winner(rows, "primary_product_gwp_kgco2e_per_kg"),
        }

    return {
        "Across all production modes (NH3 and Urea, current capacity)": _block(all_current_capacity_modes),
        "Feedstock comparison": _block(feedstock_variants),
        "NH3 recovery method comparison": _block(nh3_methods),
        "Urea recovery method comparison": _block(urea_methods),
        "Capacity scaling": _block(capacity_curve),
        "Electricity case comparison": _block(electricity_variants),
        "LCA credit sensitivity": _block(credit_sensitivity),
    }


def comprehensive_chat_context(
    base_config: ScenarioConfig,
    overrides: Mapping[str, float],
    active_evaluation: ScenarioEvaluation,
    source_rows: Sequence[Mapping[str, object]],
    active_figure_ids: Sequence[str] = (),
) -> Dict[str, object]:
    """Build a snapshot covering the active scenario PLUS broad cross-scenario coverage.

    The chat interpreter uses this to answer questions that are not specific to the
    currently-selected sidebar settings (e.g. "which feedstock is most
    profitable?", "which recovery method has the lowest GWP?", "how does
    LCOX scale with plant capacity?").

    Returned structure (all numbers in compact summary form):
      - active_scenario   — full detail of the currently-selected scenario
      - nh3_recovery_method_comparison   — every NH3 recovery method at active capacity
      - urea_recovery_method_comparison  — every Urea recovery method at active capacity
      - feedstock_comparison             — all feedstock pathways at active scenario
      - electricity_case_comparison      — US grid vs renewable for active scenario
      - capacity_scaling                 — 100 / 1k / 10k t/y for active method
      - lca_credit_sensitivity           — toggle each LCA credit on/off
      - active_figures                   — metadata for figures the user is viewing
    """
    base_overrides = normalize_overrides(overrides)
    cap = float(base_config.annual_primary_product_tpy)

    def _run(updates: Dict[str, object]) -> Dict[str, object]:
        cfg_kwargs = {
            "category": base_config.category,
            "annual_primary_product_tpy": base_config.annual_primary_product_tpy,
            "feedstock_type": base_config.feedstock_type,
            "electricity_case": base_config.electricity_case,
            "ammonia_recovery_method": base_config.ammonia_recovery_method,
            "urea_recovery_method": base_config.urea_recovery_method,
            "use_scp_credit": base_config.use_scp_credit,
            "co2_source": base_config.co2_source,
            "use_biogenic_carbon_credit": base_config.use_biogenic_carbon_credit,
            "use_scp_displacement_credit": base_config.use_scp_displacement_credit,
        }
        cfg_kwargs.update(updates)
        cfg = ScenarioConfig(**cfg_kwargs)
        return _compact_eval(evaluate_scenario(cfg, overrides=base_overrides))

    nh3_methods = [
        _run({
            "category": ScenarioCategory.AMMONIA_SCP,
            "annual_primary_product_tpy": cap,
            "ammonia_recovery_method": method,
        })
        for method in AmmoniaRecoveryMethod
    ]
    urea_methods = [
        _run({
            "category": ScenarioCategory.BIO_UREA_SCP,
            "annual_primary_product_tpy": cap,
            "urea_recovery_method": method,
        })
        for method in UreaRecoveryMethod
    ]
    feedstock_variants = [
        _run({"feedstock_type": feedstock, "annual_primary_product_tpy": cap})
        for feedstock in FeedstockType
    ]
    electricity_variants = [
        _run({"electricity_case": elec, "annual_primary_product_tpy": cap})
        for elec in ElectricityCase
    ]
    capacity_curve = [
        _run({"annual_primary_product_tpy": cap_val})
        for cap_val in (100.0, 1_000.0, 10_000.0)
    ]
    credit_sensitivity = [
        _run({
            "annual_primary_product_tpy": cap,
            "use_biogenic_carbon_credit": False,
            "use_scp_displacement_credit": False,
            "co2_source": CO2Source.FOSSIL_CAPTURE,
            "electricity_case": ElectricityCase.US_GRID,
        }),
        _run({
            "annual_primary_product_tpy": cap,
            "use_biogenic_carbon_credit": True,
            "use_scp_displacement_credit": False,
            "co2_source": CO2Source.BIOGENIC_WASTE,
            "electricity_case": ElectricityCase.US_GRID,
        }),
        _run({
            "annual_primary_product_tpy": cap,
            "use_biogenic_carbon_credit": True,
            "use_scp_displacement_credit": True,
            "co2_source": CO2Source.BIOGENIC_WASTE,
            "electricity_case": ElectricityCase.RENEWABLE,
        }),
    ]

    raw_snapshot: Dict[str, object] = {
        "Notes": (
            "All field names and values are already plain English; use them verbatim in "
            "the reply and never echo any snake_case identifier. Negative 'Net LCOX (USD/kg)' "
            "means SCP / co-product credits exceed variable cost. 'Precomputed best options' "
            "lists Python-ranked winners — consult it FIRST for 'which is best' questions. "
            "Each comparison list varies one axis; constant fields are omitted."
        ),
        "Precomputed best options": _ranking_summary(
            nh3_methods=nh3_methods,
            urea_methods=urea_methods,
            feedstock_variants=feedstock_variants,
            electricity_variants=electricity_variants,
            capacity_curve=capacity_curve,
            credit_sensitivity=credit_sensitivity,
        ),
        "Active scenario": active_evaluation.to_dict(),
        "NH3 recovery method comparison": _strip_constant_fields(nh3_methods),
        "Urea recovery method comparison": _strip_constant_fields(urea_methods),
        "Feedstock comparison": _strip_constant_fields(feedstock_variants),
        "Electricity case comparison": _strip_constant_fields(electricity_variants),
        "Capacity scaling": _strip_constant_fields(capacity_curve),
        "LCA credit sensitivity": _strip_constant_fields(credit_sensitivity),
    }
    return _friendlify(raw_snapshot)  # type: ignore[return-value]
