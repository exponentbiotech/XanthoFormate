from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from formate_biorefinery_model import (
    AmmoniaRecoveryMethod,
    CO2Source,
    ElectricityCase,
    FeedstockType,
    ScenarioCategory,
    ScenarioConfig,
    UreaRecoveryMethod,
)
from formate_biorefinery_model.app_support import (
    all_slider_defaults,
    all_slider_specs,
    app_context_snapshot,
    current_config_summary,
    evaluate_dashboard_grid,
    figure_ids,
    kpi_cards,
    reference_summary,
    source_rows_for_display,
)
from formate_biorefinery_model.groq_chat import DEFAULT_GROQ_MODEL, ask_groq, groq_available
from formate_biorefinery_model.reporting import (
    figure_metadata,
    plot_annual_cashflow,
    plot_cost_structure,
    plot_cost_vs_gwp,
    plot_executive_summary,
    plot_margin_curve,
    plot_market_viability_overview,
    plot_nh3_recovery_comparison,
    plot_process_flow,
    plot_sensitivity_tornado,
    plot_urea_recovery_comparison,
)
from formate_biorefinery_model.run_scenarios import evaluate_scenario


st.set_page_config(
    page_title="Xanthobacter C1 Biorefinery Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _secret(key: str) -> Optional[str]:
    try:
        val = st.secrets.get(key, None)
        return str(val) if val else None
    except Exception:
        return None


def _check_password() -> bool:
    required = _secret("APP_PASSWORD")
    if not required:
        return True
    if st.session_state.get("authenticated"):
        return True

    st.markdown("## Xanthobacter C1 Biorefinery Explorer")
    st.caption("This app is password protected.")
    with st.form("login_form"):
        pw = st.text_input("Password", type="password", key="login_pw")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if pw == required:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f4f7fb 0%, #ffffff 240px);
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            color: #101828;
        }
        [data-testid="stMain"] p,
        [data-testid="stMain"] li,
        [data-testid="stMain"] label,
        [data-testid="stMain"] span,
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4 {
            color: #101828;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #121a2b 0%, #17233a 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        [data-testid="stSidebar"] * {
            color: #eef2ff;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCaption {
            color: #dbe4ff !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] .stTextInput input {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            color: #f7f9ff;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            font-weight: 700;
            font-size: 0.92rem;
        }
        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1.08;
            color: #101828;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            color: #5b6477;
            font-size: 0.98rem;
            margin-bottom: 1.25rem;
        }
        .section-kicker {
            display: inline-block;
            background: #e9f2ff;
            color: #0b63ce;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.28rem 0.6rem;
            border-radius: 999px;
            margin-bottom: 0.55rem;
        }
        .section-title {
            font-size: 1.52rem;
            font-weight: 800;
            color: #101828;
            margin-bottom: 0.10rem;
        }
        .section-subtitle {
            color: #667085;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .mini-card {
            background: #f8fbff;
            border: 1px solid #dbe7f6;
            border-radius: 16px;
            padding: 0.95rem 1rem;
            min-height: 136px;
        }
        .mini-card h4 {
            margin: 0 0 0.65rem 0;
            font-size: 0.92rem;
            color: #0f172a;
        }
        .mini-card p {
            margin: 0.2rem 0;
            color: #334155;
            font-size: 0.92rem;
        }
        .sidebar-chip {
            display: inline-block;
            background: #1d4ed8;
            color: white;
            font-size: 0.70rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.32rem 0.65rem;
            border-radius: 999px;
            margin: 0.35rem 0 0.6rem 0;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            color: #667085 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.28rem !important;
            color: #101828 !important;
        }
        [data-testid="stDataFrame"] * {
            color: #101828 !important;
        }
        [data-testid="stMarkdownContainer"] {
            color: #101828;
        }
        [data-testid="stExpander"] summary {
            font-weight: 700;
            font-size: 0.94rem;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details {
            background: #17233a !important;
            border: 1px solid rgba(219,228,255,0.22) !important;
            border-radius: 14px !important;
            overflow: hidden;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            background: #111827 !important;
            color: #eef2ff !important;
            border-bottom: 1px solid rgba(219,228,255,0.18) !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary *,
        [data-testid="stSidebar"] [data-testid="stExpander"] details * {
            color: #eef2ff !important;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] .stNumberInput [data-baseweb="input"],
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] .stTextInput input {
            background-color: #111827 !important;
            border-color: rgba(219,228,255,0.24) !important;
            color: #eef2ff !important;
            -webkit-text-fill-color: #eef2ff !important;
        }
        [data-testid="stSidebar"] .stNumberInput button {
            background-color: #111827 !important;
            border-color: rgba(219,228,255,0.24) !important;
            color: #eef2ff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_FIG_TITLES = {
    "00_process_flow": "Process Flow Diagram",
    "01_market_viability": "Market Viability Overview",
    "02_cost_structure": "Cost Structure",
    "03_scale_margin": "Scale & Margin Curve",
    "04_cashflow_npv": "Annual Cash-Flow & NPV",
    "05_cost_vs_gwp": "Cost vs. GWP (LCA deep-dive)",
    "06_sensitivity_nh3": "Sensitivity — NH3 route",
    "07_sensitivity_urea": "Sensitivity — Urea route",
    "08_executive_summary": "Executive Summary",
    "09_nh3_recovery_compare": "NH3 Recovery Method Comparison",
    "10_urea_recovery_compare": "Urea Recovery Method Comparison",
}

_NH3_LABELS = {m.value: m.value.replace("_", " ").title() for m in AmmoniaRecoveryMethod}
_NH3_LABELS[AmmoniaRecoveryMethod.STRUVITE_MAP.value] = "Struvite"
_UREA_LABELS = {m.value: m.value.replace("_", " ").title() for m in UreaRecoveryMethod}
_ELEC_LABELS = {m.value: m.value.replace("_", " ").title() for m in ElectricityCase}
_CO2_LABELS = {m.value: m.value.replace("_", " ").title() for m in CO2Source}
_FEED_LABELS = {
    FeedstockType.FORMATE.value: "Formate (CO2 electrolysis)",
    FeedstockType.H2_CO2.value: "H2/CO2 (water electrolysis)",
    FeedstockType.METHANOL.value: "Methanol (purchased)",
}

_MODEL_OPTIONS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "custom",
]


def _display_pathway_label(config: ScenarioConfig) -> str:
    feed_prefix = _FEED_LABELS.get(config.feedstock_type.value, "")
    if config.category == ScenarioCategory.AMMONIA_SCP:
        if config.ammonia_recovery_method == AmmoniaRecoveryMethod.STRUVITE_MAP:
            product = "Struvite + SCP"
        elif config.ammonia_recovery_method == AmmoniaRecoveryMethod.MAP_FERTILIZER:
            product = "MAP fertilizer + SCP"
        else:
            product = "NH3 + SCP"
    else:
        product = "Urea + SCP"
    return f"{product} via {feed_prefix}" if feed_prefix else product


def _data_fingerprint() -> str:
    """Hash the CSV data files so the cache auto-busts when references change."""
    import hashlib
    h = hashlib.md5()
    data_dir = Path(__file__).resolve().parent / "formate_biorefinery_model" / "data"
    for name in sorted(["economic_inputs.csv", "technology_inputs.csv", "lca_factors.csv"]):
        p = data_dir / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


@st.cache_data(show_spinner=False)
def cached_evaluate(config: ScenarioConfig, overrides: Dict[str, float], _data_hash: str = ""):
    return evaluate_scenario(config, overrides=overrides)


@st.cache_data(show_spinner=False)
def cached_grid(
    base_config: ScenarioConfig,
    overrides: Dict[str, float],
    nh3_method: AmmoniaRecoveryMethod,
    urea_method: UreaRecoveryMethod,
    _data_hash: str = "",
):
    return evaluate_dashboard_grid(
        base_config,
        overrides=overrides,
        nh3_method=nh3_method,
        urea_method=urea_method,
    )


def _section_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(f"<div class='section-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def _render_figure_methodology(fig_id: str) -> None:
    meta = figure_metadata(fig_id)
    with st.expander("Methodology & calculation notes", expanded=False):
        st.markdown(f"**What is plotted** — {meta.get('what_is_plotted', '')}")
        st.markdown(f"**How it was calculated** — {meta.get('calculation_basis', '')}")
        assumptions = meta.get("important_assumptions", [])
        if assumptions:
            st.markdown("**Important assumptions:**")
            for item in assumptions:
                st.markdown(f"- {item}")
        guidance = meta.get("interpretation_guidance", "")
        if guidance:
            st.markdown(f"**How to read it** — {guidance}")


def _config_card(title: str, rows: List[tuple[str, str]]) -> None:
    lines = "".join(f"<p><strong>{label}:</strong> {value}</p>" for label, value in rows)
    st.markdown(
        f"<div class='mini-card'><h4>{title}</h4>{lines}</div>",
        unsafe_allow_html=True,
    )


def build_figure(
    fig_id: str,
    grid_rows,
    current_config: ScenarioConfig,
    overrides: Dict[str, float],
    nh3_method: AmmoniaRecoveryMethod,
    urea_method: UreaRecoveryMethod,
):
    cap = float(current_config.annual_primary_product_tpy)
    nh3_cfg = replace(
        current_config,
        category=ScenarioCategory.AMMONIA_SCP,
        annual_primary_product_tpy=cap,
        ammonia_recovery_method=nh3_method,
        user_overrides=overrides,
    )
    urea_cfg = replace(
        current_config,
        category=ScenarioCategory.BIO_UREA_SCP,
        annual_primary_product_tpy=cap,
        urea_recovery_method=urea_method,
        user_overrides=overrides,
    )
    builders = {
        "00_process_flow": lambda: plot_process_flow(),
        "01_market_viability": lambda: plot_market_viability_overview(grid_rows),
        "02_cost_structure": lambda: plot_cost_structure(grid_rows, capacity_tpy=cap),
        "03_scale_margin": lambda: plot_margin_curve(grid_rows),
        "04_cashflow_npv": lambda: plot_annual_cashflow(overrides=overrides),
        "05_cost_vs_gwp": lambda: plot_cost_vs_gwp(grid_rows, capacity_tpy=cap, overrides=overrides),
        "06_sensitivity_nh3": lambda: plot_sensitivity_tornado(nh3_cfg),
        "07_sensitivity_urea": lambda: plot_sensitivity_tornado(urea_cfg),
        "08_executive_summary": lambda: plot_executive_summary(grid_rows),
        "09_nh3_recovery_compare": lambda: plot_nh3_recovery_comparison(capacity_tpy=cap, overrides=overrides),
        "10_urea_recovery_compare": lambda: plot_urea_recovery_comparison(capacity_tpy=cap, overrides=overrides),
    }
    return builders[fig_id]()


def main() -> None:
    if not _check_password():
        st.stop()

    _inject_css()

    category_values = [item.value for item in ScenarioCategory]
    feed_values = [item.value for item in FeedstockType]
    nh3_values = [m.value for m in AmmoniaRecoveryMethod]
    urea_values = [m.value for m in UreaRecoveryMethod]
    elec_values = [item.value for item in ElectricityCase]
    co2_values = [item.value for item in CO2Source]

    with st.sidebar:
        st.markdown("<div class='sidebar-chip'>Focus View</div>", unsafe_allow_html=True)
        st.caption("Choose the default operating point and assumptions that drive the full model.")

        feedstock_type = FeedstockType(
            st.selectbox(
                "Feedstock",
                options=feed_values,
                index=feed_values.index(FeedstockType.FORMATE.value),
                format_func=lambda v: _FEED_LABELS.get(v, v),
                help="Carbon/energy source for the biorefinery. Formate = CO2 electrolysis; H2/CO2 = water electrolysis; Methanol = purchased.",
            )
        )
        category = ScenarioCategory(
            st.selectbox(
                "Primary pathway",
                options=category_values,
                index=category_values.index(ScenarioCategory.AMMONIA_SCP.value),
                format_func=lambda v: "NH3 + SCP" if v == ScenarioCategory.AMMONIA_SCP.value else "Urea + SCP",
                help="Primary product route for the single-scenario cards and figures.",
            )
        )
        capacity_tpy = float(
            st.number_input(
                "Primary product capacity (t / y)",
                min_value=10.0,
                max_value=100_000.0,
                value=1_000.0,
                step=100.0,
                help="Annual nameplate output of the primary product.",
            )
        )

        st.markdown("<div class='sidebar-chip'>Process Basis</div>", unsafe_allow_html=True)
        with st.expander("Recovery methods & grid settings", expanded=True):
            nh3_method = AmmoniaRecoveryMethod(
                st.selectbox(
                    "NH3 recovery route",
                    options=nh3_values,
                    index=nh3_values.index(AmmoniaRecoveryMethod.STRUVITE_MAP.value),
                    format_func=lambda v: _NH3_LABELS[v],
                    help="Used in the current-scenario route and the comparison figures.",
                )
            )
            urea_method = UreaRecoveryMethod(
                st.selectbox(
                    "Urea recovery route",
                    options=urea_values,
                    index=urea_values.index(UreaRecoveryMethod.MVR_CRYSTALLIZATION.value),
                    format_func=lambda v: _UREA_LABELS[v],
                    help="Used in the current-scenario route and the comparison figures.",
                )
            )
            electricity_case = ElectricityCase(
                st.selectbox(
                    "Electricity case",
                    options=elec_values,
                    index=elec_values.index(ElectricityCase.RENEWABLE.value),
                    format_func=lambda v: _ELEC_LABELS[v],
                )
            )
            if feedstock_type != FeedstockType.METHANOL:
                co2_source = CO2Source(
                    st.selectbox(
                        "CO2 source",
                        options=co2_values,
                        index=co2_values.index(CO2Source.BIOGENIC_WASTE.value),
                        format_func=lambda v: _CO2_LABELS[v],
                    )
                )
            else:
                co2_source = CO2Source.BIOGENIC_WASTE
                st.caption("CO2 source not applicable for methanol feedstock.")

        st.markdown("<div class='sidebar-chip'>Climate / Credits</div>", unsafe_allow_html=True)
        with st.expander("LCA credits", expanded=True):
            use_scp_credit = st.checkbox(
                "SCP revenue credit",
                value=True,
                help="Include SCP coproduct revenue in net LCOX.",
            )
            use_biogenic_carbon_credit = st.checkbox(
                "Biogenic carbon credit",
                value=True,
                help="Credit CO2 captured in SCP biomass from a biogenic source.",
            )
            use_scp_displacement_credit = st.checkbox(
                "SCP displacement credit (system expansion)",
                value=False,
                help="System-expansion credit for displacing soy meal protein production.",
            )

        st.markdown("<div class='sidebar-chip'>Assumptions</div>", unsafe_allow_html=True)
        _slider_defaults = all_slider_defaults(feedstock_type=feedstock_type)
        slider_groups = all_slider_specs(feedstock_type=feedstock_type)
        overrides: Dict[str, float] = {}
        for group_name, specs in slider_groups.items():
            with st.expander(group_name, expanded=(group_name == "Biological performance")):
                for spec in specs:
                    label = f"{spec.label}  [{spec.unit}]" if spec.unit else spec.label
                    default_val = _slider_defaults.get(spec.key, spec.min_value)
                    value = float(
                        st.slider(
                            label,
                            min_value=float(spec.min_value),
                            max_value=float(spec.max_value),
                            value=float(default_val),
                            step=float(spec.step),
                            help=spec.help_text,
                            key=f"slider_{spec.key}_{feedstock_type.value}",
                        )
                    )
                    overrides[spec.key] = value

        st.markdown("<div class='sidebar-chip'>Assistant</div>", unsafe_allow_html=True)
        auto_key = _secret("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "")
        if auto_key:
            api_key = auto_key
            st.caption("Groq key loaded from deployment secrets.")
        else:
            api_key = st.text_input(
                "Groq API key",
                value="",
                type="password",
                help="Paste your Groq key here if you did not set `GROQ_API_KEY` in deployment secrets.",
            )

    current_config = ScenarioConfig(
        category=category,
        annual_primary_product_tpy=capacity_tpy,
        feedstock_type=feedstock_type,
        electricity_case=electricity_case,
        ammonia_recovery_method=nh3_method if category == ScenarioCategory.AMMONIA_SCP else AmmoniaRecoveryMethod.VACUUM_STRIPPING,
        urea_recovery_method=urea_method if category == ScenarioCategory.BIO_UREA_SCP else UreaRecoveryMethod.EVAPORATION,
        use_scp_credit=use_scp_credit,
        co2_source=co2_source,
        use_biogenic_carbon_credit=use_biogenic_carbon_credit,
        use_scp_displacement_credit=use_scp_displacement_credit,
        user_overrides=overrides,
    )

    _dhash = _data_fingerprint()
    current_eval = cached_evaluate(current_config, overrides, _data_hash=_dhash)
    comparison_grid = cached_grid(current_config, overrides, nh3_method=nh3_method, urea_method=urea_method, _data_hash=_dhash)
    display_sources = source_rows_for_display(current_eval.source_rows)
    reference_rows = reference_summary(current_eval.source_rows)
    summary = current_config_summary(current_config)

    st.markdown("<div class='hero-title'>Biorefinery TEA Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Integrated ammonia / urea + SCP techno-economics and cradle-to-gate LCA across "
        "three feedstock pathways (formate, H2/CO2, methanol) with transparent assumptions and a grounded Groq assistant.</div>",
        unsafe_allow_html=True,
    )

    _section_header(
        "Scenario Basis",
        "Facility + scenario basis",
        "A clean summary of the active route, recovery configuration, and climate-accounting choices.",
    )
    with st.container(border=True):
        basis_col1, basis_col2 = st.columns(2)
        with basis_col1:
            _config_card(
                "Current pathway basis",
                [
                    ("Feedstock", _FEED_LABELS.get(summary.get("feedstock_type", "formate"), "Formate")),
                    ("Primary pathway", _display_pathway_label(current_config)),
                    ("Primary capacity", f"{summary['annual_primary_product_tpy']:,.0f} t/y"),
                    ("Electricity", _ELEC_LABELS[summary["electricity_case"]]),
                    ("CO2 source", _CO2_LABELS[summary["co2_source"]]),
                ],
            )
        with basis_col2:
            _config_card(
                "Current route settings",
                [
                    ("NH3 route", _NH3_LABELS[summary["ammonia_recovery_method"]]),
                    ("Urea route", _UREA_LABELS[summary["urea_recovery_method"]]),
                    ("SCP credit", "Enabled" if summary["use_scp_credit"] else "Disabled"),
                    ("Biogenic C credit", "Enabled" if summary["use_biogenic_carbon_credit"] else "Disabled"),
                    ("Protein displacement", "Enabled" if summary["use_scp_displacement_credit"] else "Disabled"),
                ],
            )

    _section_header(
        "Operating Point",
        "Selected operating point",
        "The KPI cards below update with every slider and route change.",
    )
    metric_cols = st.columns(3)
    for idx, card in enumerate(kpi_cards(current_eval)):
        metric_cols[idx % 3].metric(card["label"], card["value"])

    with st.expander("Cross-category comparison grid", expanded=False):
        st.caption(
            "Three capacity points (100 / 1 000 / 10 000 t/y) for both routes under the current slider settings."
        )
        summary_rows = [row.to_dict() for row in comparison_grid]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    with st.expander("Active scenario JSON", expanded=False):
        st.json(summary, expanded=False)

    _section_header(
        "Figures",
        "Detailed figures",
        "Every figure below uses the exact same model state shown in the cards above.",
    )
    for fig_id in figure_ids():
        title = _FIG_TITLES.get(fig_id, fig_id)
        with st.expander(title, expanded=False):
            with st.spinner(f"Rendering {title}..."):
                try:
                    fig = build_figure(
                        fig_id,
                        comparison_grid,
                        current_config=current_config,
                        overrides=overrides,
                        nh3_method=nh3_method,
                        urea_method=urea_method,
                    )
                    st.pyplot(fig, use_container_width=True)
                    fig.clf()
                except Exception as exc:
                    st.error(f"Could not render figure: {exc}")
            _render_figure_methodology(fig_id)

    _section_header(
        "Reference Trace",
        "Sources and references",
        "Every active input is mapped to its underlying source, URL, and override status.",
    )
    with st.container(border=True):
        with st.expander("All inputs — flat provenance table", expanded=False):
            st.dataframe(pd.DataFrame(display_sources), use_container_width=True, hide_index=True)

        st.markdown("### By reference")
        ref_df = pd.DataFrame(reference_rows)
        if not ref_df.empty and "source" in ref_df.columns:
            for ref_name, group in ref_df.groupby("source", sort=True):
                with st.expander(str(ref_name), expanded=False):
                    url = group["source_url"].iloc[0] if "source_url" in group.columns else ""
                    if url:
                        st.markdown(f"[{url}]({url})")
                    st.dataframe(
                        group.drop(columns=["source", "source_url"], errors="ignore"),
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.dataframe(ref_df, use_container_width=True, hide_index=True)

    _section_header(
        "Assistant",
        "Ask the biorefinery assistant",
        "Fast Groq chat grounded in the current scenario state. The default model is optimized for responsiveness.",
    )
    with st.container(border=True):
        top_left, top_mid, top_right = st.columns([1.2, 2.6, 1.0])
        with top_left:
            model_choice = st.selectbox(
                "Model",
                options=_MODEL_OPTIONS,
                index=_MODEL_OPTIONS.index(DEFAULT_GROQ_MODEL) if DEFAULT_GROQ_MODEL in _MODEL_OPTIONS else 0,
                key="groq_model_choice",
            )
        with top_mid:
            if model_choice == "custom":
                groq_model = st.text_input(
                    "Custom model",
                    value=DEFAULT_GROQ_MODEL,
                    key="custom_groq_model",
                )
            else:
                groq_model = model_choice
            if auto_key:
                st.caption("Powered by Groq and grounded in the current scenario. API key is loaded from deployment secrets.")
            else:
                st.caption("Powered by Groq and grounded in the current scenario. Enter an API key in the sidebar to enable chat.")
        with top_right:
            if st.session_state.get("groq_messages"):
                if st.button("Clear chat", key="clear_groq"):
                    st.session_state.groq_messages = []
                    st.rerun()

        if "groq_messages" not in st.session_state:
            st.session_state.groq_messages = []

        history_container = st.container(height=320)
        with history_container:
            if not st.session_state.groq_messages:
                st.caption("Ask about the current scenario, which route looks most attractive, or how a slider change would affect cost and GWP.")
            for message in st.session_state.groq_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    chat_available = groq_available() and bool(api_key)
    if chat_available:
        prompt = st.chat_input("Ask about the current scenario, outputs, or a hypothetical...")
        if prompt:
            st.session_state.groq_messages.append({"role": "user", "content": prompt})
            snapshot = app_context_snapshot(
                current_eval,
                current_eval.source_rows,
                active_figure_ids=figure_ids(),
            )
            try:
                reply = ask_groq(
                    api_key=api_key,
                    question=prompt,
                    snapshot=snapshot,
                    history=st.session_state.groq_messages[:-1],
                    model=groq_model,
                )
            except Exception as exc:
                reply = f"Groq request failed: {exc}"

            st.session_state.groq_messages.append({"role": "assistant", "content": reply})
            st.rerun()
    elif not groq_available():
        st.info("The `groq` package is not installed in this environment.")
    else:
        st.info("Add a Groq API key in the sidebar or set `GROQ_API_KEY` in deployment secrets to enable chat.")


if __name__ == "__main__":
    main()
