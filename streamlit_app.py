from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    comprehensive_chat_context,
    current_config_summary,
    evaluate_dashboard_grid,
    figure_ids,
    kpi_cards,
    reference_summary,
    source_rows_for_display,
)
from formate_biorefinery_model.groq_chat import (
    DEFAULT_GROQ_MODEL,
    ask_groq,
    deterministic_answer,
    groq_available,
)
from formate_biorefinery_model.reporting import (
    figure_metadata,
    plot_annual_cashflow,
    plot_cost_structure,
    plot_cost_vs_gwp,
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
        /* Floating chat launcher — visible from anywhere on the page.
           Streamlit ≥1.36 exposes the widget key as a CSS class on the
           element-container, so we target .st-key-float_open_chat.

           bottom is set high enough to clear the Streamlit Community Cloud
           badges (Manage app pill + status indicator) that occupy the
           bottom-right of the viewport on deployed apps. */
        .st-key-float_open_chat {
            position: fixed !important;
            bottom: 90px !important;
            right: 24px !important;
            width: auto !important;
            margin: 0 !important;
            padding: 0 !important;
            z-index: 9999 !important;
        }
        .st-key-float_open_chat button {
            border-radius: 999px !important;
            padding: 14px 22px !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.22) !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
            background: #0b63ce !important;
            color: #ffffff !important;
            border: 1px solid rgba(255,255,255,0.18) !important;
            white-space: nowrap !important;
        }
        .st-key-float_open_chat button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 9px 22px rgba(15, 23, 42, 0.28) !important;
            background: #094fa6 !important;
        }
        .st-key-float_open_chat button:focus,
        .st-key-float_open_chat button:active {
            outline: 3px solid rgba(11, 99, 206, 0.35) !important;
            outline-offset: 2px !important;
        }
        @media (max-width: 768px) {
            .st-key-float_open_chat {
                bottom: 80px !important;
                right: 12px !important;
            }
            .st-key-float_open_chat button {
                padding: 12px 18px !important;
                font-size: 0.9rem !important;
            }
        }
        /* ── Force-light styling for the assistant chat dialog ─────────────
           Streamlit auto-themes some dialog content from the browser's
           prefers-color-scheme. Even with [theme] base="light" pinned in
           config.toml, we explicitly enforce black-on-white inside the
           modal so the chat is always readable regardless of the viewer's
           OS / browser dark-mode setting. */
        [data-testid="stDialog"],
        [data-testid="stModal"],
        div[role="dialog"] {
            color-scheme: light !important;
        }
        [data-testid="stDialog"] [data-testid="stDialogContent"],
        [data-testid="stDialog"] > div > div,
        div[role="dialog"] > div {
            background: #FFFFFF !important;
            color: #101828 !important;
        }
        [data-testid="stDialog"] *,
        div[role="dialog"] * {
            color: #101828 !important;
            border-color: #d7dde7 !important;
        }
        /* Chat-message bubbles inside the dialog */
        [data-testid="stDialog"] [data-testid="stChatMessage"],
        div[role="dialog"] [data-testid="stChatMessage"] {
            background: #F4F7FB !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 0.8rem 1rem !important;
            color: #101828 !important;
        }
        [data-testid="stDialog"] [data-testid="stChatMessage"] *,
        div[role="dialog"] [data-testid="stChatMessage"] * {
            color: #101828 !important;
            background: transparent !important;
        }
        /* Inline code / pre blocks inside chat messages */
        [data-testid="stDialog"] [data-testid="stChatMessage"] code,
        [data-testid="stDialog"] [data-testid="stChatMessage"] pre,
        div[role="dialog"] [data-testid="stChatMessage"] code,
        div[role="dialog"] [data-testid="stChatMessage"] pre {
            background: #EEF2F7 !important;
            color: #0B2447 !important;
            border: 1px solid #DAE1EC !important;
            border-radius: 6px !important;
            padding: 1px 5px !important;
        }
        /* Avatar / icon backdrop */
        [data-testid="stDialog"] [data-testid="stChatMessageAvatarUser"],
        [data-testid="stDialog"] [data-testid="stChatMessageAvatarAssistant"],
        div[role="dialog"] [data-testid="stChatMessageAvatarUser"],
        div[role="dialog"] [data-testid="stChatMessageAvatarAssistant"] {
            background: #0B63CE !important;
            color: #FFFFFF !important;
        }
        [data-testid="stDialog"] [data-testid="stChatMessageAvatarUser"] *,
        [data-testid="stDialog"] [data-testid="stChatMessageAvatarAssistant"] *,
        div[role="dialog"] [data-testid="stChatMessageAvatarUser"] *,
        div[role="dialog"] [data-testid="stChatMessageAvatarAssistant"] * {
            color: #FFFFFF !important;
        }
        /* Scrollable history container */
        [data-testid="stDialog"] [data-testid="stVerticalBlockBorderWrapper"],
        div[role="dialog"] [data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
        }
        /* Chat input */
        [data-testid="stDialog"] [data-testid="stChatInput"],
        div[role="dialog"] [data-testid="stChatInput"] {
            background: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 10px !important;
        }
        [data-testid="stDialog"] [data-testid="stChatInput"] textarea,
        [data-testid="stDialog"] [data-testid="stChatInput"] input,
        div[role="dialog"] [data-testid="stChatInput"] textarea,
        div[role="dialog"] [data-testid="stChatInput"] input {
            background: #FFFFFF !important;
            color: #101828 !important;
            -webkit-text-fill-color: #101828 !important;
            caret-color: #101828 !important;
        }
        [data-testid="stDialog"] [data-testid="stChatInput"] textarea::placeholder,
        [data-testid="stDialog"] [data-testid="stChatInput"] input::placeholder,
        div[role="dialog"] [data-testid="stChatInput"] textarea::placeholder,
        div[role="dialog"] [data-testid="stChatInput"] input::placeholder {
            color: #6B7280 !important;
            -webkit-text-fill-color: #6B7280 !important;
        }
        /* Buttons inside the dialog (Clear / Close) */
        [data-testid="stDialog"] .stButton button,
        div[role="dialog"] .stButton button {
            background: #FFFFFF !important;
            color: #101828 !important;
            border: 1px solid #CBD5E1 !important;
            font-weight: 600 !important;
        }
        [data-testid="stDialog"] .stButton button[kind="primary"],
        div[role="dialog"] .stButton button[kind="primary"] {
            background: #0B63CE !important;
            color: #FFFFFF !important;
            border-color: #0B63CE !important;
        }
        [data-testid="stDialog"] .stButton button[kind="primary"] *,
        div[role="dialog"] .stButton button[kind="primary"] * {
            color: #FFFFFF !important;
        }
        [data-testid="stDialog"] .stButton button:hover,
        div[role="dialog"] .stButton button:hover {
            background: #F4F7FB !important;
        }
        [data-testid="stDialog"] .stButton button[kind="primary"]:hover,
        div[role="dialog"] .stButton button[kind="primary"]:hover {
            background: #094FA6 !important;
        }
        /* Captions inside dialog */
        [data-testid="stDialog"] [data-testid="stCaptionContainer"],
        div[role="dialog"] [data-testid="stCaptionContainer"] {
            color: #4B5563 !important;
        }
        [data-testid="stDialog"] [data-testid="stCaptionContainer"] *,
        div[role="dialog"] [data-testid="stCaptionContainer"] * {
            color: #4B5563 !important;
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
    "09_nh3_recovery_compare": "NH3 Recovery Method Comparison",
    "10_urea_recovery_compare": "Urea Recovery Method Comparison",
}

_FIGURE_DISPLAY_ORDER = [
    "01_market_viability",
    "02_cost_structure",
    "03_scale_margin",
    "04_cashflow_npv",
    "05_cost_vs_gwp",
    "09_nh3_recovery_compare",
    "10_urea_recovery_compare",
    "06_sensitivity_nh3",
    "07_sensitivity_urea",
    "00_process_flow",
]
_EXCLUDED_FIGURES = {"08_executive_summary"}

_COMMON_FORMULA_DEFINITIONS = [
    "M_p = annual sellable primary product mass (kg/y).",
    "C_var = total variable operating cost ($/y).",
    "C_fixed = total fixed operating cost ($/y), including labor, overhead, maintenance, stack replacement, and annualized CapEx.",
    "R_credit = total annual co-product and system credits ($/y), such as SCP, H2, CO2, struvite, or MAP fertilizer credits when enabled.",
    "P_market = benchmark market price for the primary product ($/kg).",
]

_FIGURE_FORMULAS: Dict[str, List[Tuple[str, str, str]]] = {
    "00_process_flow": [
        (
            "Purpose of the schematic",
            r"\text{Process state} = f(\text{feedstock route},\ \text{primary pathway},\ \text{capacity},\ \text{recovery method})",
            "This figure is not a numerical plot. It shows the sequence of unit operations used to build the mass, energy, cost, and GWP ledgers used by the quantitative figures below.",
        ),
        (
            "Mass-balance logic",
            r"\text{Annual flow}_{i} = M_p \times \text{route-specific stoichiometric factor}_{i}",
            "The model scales each material and utility flow from the selected annual primary-product capacity and the chosen NH3 or urea recovery route.",
        ),
    ],
    "01_market_viability": [
        (
            "Gross levelized cost",
            r"\mathrm{Gross\ LCOX} = \frac{C_{\mathrm{var}} + C_{\mathrm{fixed}}}{M_p}",
            "This is the annual cost of making the primary product before co-product credits.",
        ),
        (
            "Net levelized cost",
            r"\mathrm{Net\ LCOX} = \frac{C_{\mathrm{var}} + C_{\mathrm{fixed}} - R_{\mathrm{credit}}}{M_p}",
            "This subtracts enabled credits from the annual cost numerator, then divides by sellable primary-product mass.",
        ),
        (
            "Market benchmark",
            r"P_{\mathrm{market}} = \frac{R_{\mathrm{primary}}}{M_p}",
            "The benchmark line is the assumed product revenue per kg. A case screens favorably when net LCOX is below this line.",
        ),
    ],
    "02_cost_structure": [
        (
            "Positive cost bars",
            r"\mathrm{Cost\ contribution}_{j} = \frac{C_j}{M_p}",
            "Each annual cost component is divided by annual sellable primary-product mass to express it as $/kg product.",
        ),
        (
            "Credit bars",
            r"\mathrm{Credit\ contribution}_{k} = -\frac{R_{\mathrm{credit},k}}{M_p}",
            "Credits are shown as negative bars because they reduce net LCOX.",
        ),
        (
            "Net cost after all bars",
            r"\mathrm{Net\ LCOX} = \sum_j \frac{C_j}{M_p} - \sum_k \frac{R_{\mathrm{credit},k}}{M_p}",
            "The endpoint of the waterfall matches the net LCOX used elsewhere in the app.",
        ),
    ],
    "03_scale_margin": [
        (
            "Net cost at each capacity",
            r"\mathrm{Net\ LCOX}(Q) = \frac{C_{\mathrm{var}}(Q) + C_{\mathrm{fixed}}(Q) - R_{\mathrm{credit}}(Q)}{M_p(Q)}",
            "The same TEA is re-run at each modeled capacity Q.",
        ),
        (
            "Margin to market",
            r"\mathrm{Margin}(Q) = P_{\mathrm{market}} - \mathrm{Net\ LCOX}(Q)",
            "Positive margin means the modeled cost is below the benchmark product price.",
        ),
        (
            "Capital scaling behind the curve",
            r"\mathrm{CapEx}(Q) = \mathrm{CapEx}_{\mathrm{ref}}\left(\frac{Q}{Q_{\mathrm{ref}}}\right)^n",
            "Capacity affects the denominator and also changes scaled capital, labor, and fixed-cost burden.",
        ),
    ],
    "04_cashflow_npv": [
        (
            "Annual cash flow",
            r"\mathrm{Cash\ flow} = R_{\mathrm{primary}} + R_{\mathrm{credit}} - C_{\mathrm{var}} - \left(C_{\mathrm{fixed}} - C_{\mathrm{annualized\ capex}}\right)",
            "CapEx is treated as an upfront investment in the NPV calculation, so annualized CapEx is removed from annual fixed cost when computing cash flow.",
        ),
        (
            "Capital recovery factor",
            r"\mathrm{CRF} = \frac{r(1+r)^N}{(1+r)^N - 1}",
            "The annualized CapEx used in LCOX equals total capital multiplied by CRF, where r is discount rate and N is plant life.",
        ),
        (
            "Net present value",
            r"\mathrm{NPV} = -C_{\mathrm{capital}} + \sum_{t=1}^{N}\frac{\mathrm{Cash\ flow}}{(1+r)^t}",
            "The NPV panel discounts the same annual cash flow over the configured plant life.",
        ),
    ],
    "05_cost_vs_gwp": [
        (
            "Annual GWP contribution",
            r"G_i = A_i \times EF_i",
            "Each activity A_i, such as kWh of electricity or kg of reagent, is multiplied by its emission factor EF_i.",
        ),
        (
            "Biogenic carbon credit",
            r"G_{\mathrm{bioC}} = -M_{\mathrm{SCP}}\times f_C \times \frac{44}{12}",
            "When enabled for biogenic CO2, the carbon stored in SCP is converted from kg C to kg CO2e and applied as a credit.",
        ),
        (
            "Net GWP intensity",
            r"\mathrm{GWP}_{p} = \frac{\sum_i G_i}{M_p}",
            "The plotted carbon intensity is the sum of annual burdens and credits divided by annual sellable primary-product mass.",
        ),
        (
            "Cost-climate screen",
            r"\left(\mathrm{Net\ LCOX},\ \mathrm{GWP}_{p}\right)",
            "The scatter plot pairs the economic result with the carbon-intensity result for each screened case.",
        ),
    ],
    "06_sensitivity_nh3": [
        (
            "One-at-a-time low case",
            r"x_{\mathrm{low}} = x_0(1-\Delta)",
            "For each NH3 parameter, only that parameter is lowered while all other assumptions stay at the base value.",
        ),
        (
            "One-at-a-time high case",
            r"x_{\mathrm{high}} = x_0(1+\Delta)",
            "The same parameter is then raised by the same fractional delta, typically 20%.",
        ),
        (
            "Tornado bar width",
            r"\mathrm{Bar\ width} = \max(\mathrm{Net\ LCOX}_{\mathrm{low}},\mathrm{Net\ LCOX}_{\mathrm{high}}) - \min(\mathrm{Net\ LCOX}_{\mathrm{low}},\mathrm{Net\ LCOX}_{\mathrm{high}})",
            "Longer bars identify assumptions that move NH3 economics the most.",
        ),
    ],
    "07_sensitivity_urea": [
        (
            "One-at-a-time low case",
            r"x_{\mathrm{low}} = x_0(1-\Delta)",
            "For each urea parameter, only that parameter is lowered while all other assumptions stay at the base value.",
        ),
        (
            "One-at-a-time high case",
            r"x_{\mathrm{high}} = x_0(1+\Delta)",
            "The same parameter is then raised by the same fractional delta, typically 20%.",
        ),
        (
            "Tornado bar width",
            r"\mathrm{Bar\ width} = \max(\mathrm{Net\ LCOX}_{\mathrm{low}},\mathrm{Net\ LCOX}_{\mathrm{high}}) - \min(\mathrm{Net\ LCOX}_{\mathrm{low}},\mathrm{Net\ LCOX}_{\mathrm{high}})",
            "Longer bars identify assumptions that move urea economics the most.",
        ),
    ],
    "09_nh3_recovery_compare": [
        (
            "Gross and net recovery-route LCOX",
            r"\mathrm{Gross\ LCOX}_{m} = \frac{C_{\mathrm{var},m}+C_{\mathrm{fixed},m}}{M_p},\qquad \mathrm{Net\ LCOX}_{m} = \frac{C_{\mathrm{var},m}+C_{\mathrm{fixed},m}-R_{\mathrm{credit},m}}{M_p}",
            "Each NH3 recovery method m is evaluated at the same product capacity for an apples-to-apples comparison.",
        ),
        (
            "Recovery cost drivers",
            r"\mathrm{Driver}_{j,m} = \frac{C_{j,m}}{M_p}",
            "Electricity, NaOH, MgCl2, H3PO4, membrane replacement, and annualized CapEx are normalized to $/kg NH3-equivalent product.",
        ),
        (
            "Fertilizer-route crediting",
            r"R_{\mathrm{credit},m} = R_{\mathrm{SCP}} + R_{\mathrm{H2}} + R_{\mathrm{CO2}} + R_{\mathrm{struvite/MAP}}",
            "Struvite and MAP fertilizer revenues are treated as credits while the denominator remains NH3-equivalent mass.",
        ),
    ],
    "10_urea_recovery_compare": [
        (
            "Gross and net recovery-route LCOX",
            r"\mathrm{Gross\ LCOX}_{m} = \frac{C_{\mathrm{var},m}+C_{\mathrm{fixed},m}}{M_p},\qquad \mathrm{Net\ LCOX}_{m} = \frac{C_{\mathrm{var},m}+C_{\mathrm{fixed},m}-R_{\mathrm{credit},m}}{M_p}",
            "Each urea recovery method m is evaluated at the same product capacity.",
        ),
        (
            "Steam cost",
            r"\mathrm{Steam\ OPEX}_{m} = \frac{M_{\mathrm{steam},m}\times P_{\mathrm{steam}}}{M_p}",
            "The center panel isolates steam burden per kg urea.",
        ),
        (
            "Recovery electricity cost",
            r"\mathrm{Recovery\ electricity\ OPEX}_{m} = \frac{\left(E_{\mathrm{total},m}-E_{\mathrm{electrolysis},m}-E_{\mathrm{fermentation},m}-E_{\mathrm{SCP},m}\right)\times P_{\mathrm{electricity}}}{M_p}",
            "The center panel also isolates the incremental electricity associated with urea recovery.",
        ),
        (
            "GWP by recovery route",
            r"\mathrm{GWP}_{p,m} = \frac{\sum_i G_{i,m}}{M_p}",
            "The right panel compares net GWP intensity for each urea recovery method.",
        ),
    ],
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
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
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


_SNAPSHOT_SCHEMA_VERSION = "2026-04-28-v8-deterministic-in-app"


def _data_fingerprint() -> str:
    """Hash the CSV data files + snapshot schema version so the cache auto-busts
    when the references OR the snapshot shape change. Bump
    ``_SNAPSHOT_SCHEMA_VERSION`` whenever the snapshot structure changes so old
    cached payloads don't survive a deploy.
    """
    import hashlib
    h = hashlib.md5()
    h.update(_SNAPSHOT_SCHEMA_VERSION.encode())
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


@st.cache_data(show_spinner=False)
def cached_chat_snapshot(
    base_config: ScenarioConfig,
    overrides: Dict[str, float],
    _data_hash: str = "",
) -> Dict[str, object]:
    """Build the comprehensive cross-scenario snapshot the assistant uses.

    Cached because building it runs ~20 small scenario evaluations covering all
    recovery methods, feedstocks, electricity cases, capacities, and LCA credit
    settings. With caching the dialog opens instantly after the first time the
    user asks a question for a given configuration.
    """
    active = evaluate_scenario(base_config, overrides=overrides)
    return comprehensive_chat_context(
        base_config=base_config,
        overrides=overrides,
        active_evaluation=active,
        source_rows=active.source_rows,
        active_figure_ids=figure_ids(),
    )


def _section_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(f"<div class='section-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def _md_escape(text: str) -> str:
    return str(text).replace("$", r"\$")


def _plain_language_note(text: str) -> str:
    replacements = {
        "run_baseline_cases().": "the baseline scenario set.",
        "run_baseline_cases()": "the baseline scenario set",
        "run_best_methods_grid()": "the curated best-method comparison set",
        "run_lca_sensitivity_grid()": "the climate-accounting sensitivity set",
        "run_best_methods_negative_gwp_grid()": "the negative-GWP screening set",
        "run_recovery_comparison(ScenarioCategory.AMMONIA_SCP, capacity_tpy)": "the NH3 recovery-method comparison at the selected capacity",
        "run_recovery_comparison(ScenarioCategory.BIO_UREA_SCP, capacity_tpy)": "the urea recovery-method comparison at the selected capacity",
        "run_sensitivity_cases()": "the one-at-a-time sensitivity calculation",
        "ScenarioConfig": "the current scenario settings",
        "ScenarioCategory.AMMONIA_SCP": "the NH3 + SCP pathway",
        "ScenarioCategory.BIO_UREA_SCP": "the urea + SCP pathway",
        "ScenarioCategory": "the selected product pathway",
        "SENSITIVITY_PARAMS": "the sensitivity parameter list",
        "TEAResults": "the economic results",
        "LCAResults": "the climate results",
        "process_blocks.py": "the process model",
        "tea.py": "the economic model",
        "net_primary_lcox_usd_per_kg": "net LCOX",
        "benchmark_primary_revenue_usd_per_y": "benchmark annual product revenue",
        "capacity_tpy": "selected annual capacity",
    }
    clean = str(text)
    for old, new in replacements.items():
        clean = clean.replace(old, new)
    return clean


def _render_figure_methodology(fig_id: str) -> None:
    meta = figure_metadata(fig_id)
    formulas = _FIGURE_FORMULAS.get(fig_id, [])
    with st.expander("How was this figure calculated?", expanded=False):
        st.markdown(f"**What is plotted** — {_plain_language_note(meta.get('what_is_plotted', ''))}")
        if fig_id != "00_process_flow":
            st.markdown("**Common terms used below:**")
            for item in _COMMON_FORMULA_DEFINITIONS:
                st.markdown(f"- {_md_escape(item)}")
        for heading, formula, explanation in formulas:
            st.markdown(f"**{heading}**")
            st.latex(formula)
            st.markdown(_md_escape(explanation))
        assumptions = meta.get("important_assumptions", [])
        if assumptions:
            st.markdown("**Important assumptions:**")
            for item in assumptions:
                st.markdown(f"- {_plain_language_note(item)}")
        guidance = meta.get("interpretation_guidance", "")
        if guidance:
            st.markdown(f"**How to read it** — {_plain_language_note(guidance)}")


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
        "09_nh3_recovery_compare": lambda: plot_nh3_recovery_comparison(capacity_tpy=cap, overrides=overrides),
        "10_urea_recovery_compare": lambda: plot_urea_recovery_comparison(capacity_tpy=cap, overrides=overrides),
    }
    return builders[fig_id]()


_DOLLAR_PATTERN = re.compile(r"(?<!\\)\$")


def _sanitize_chat_reply(text: str) -> str:
    r"""Escape bare ``$`` characters so Streamlit does not render currency as LaTeX math.

    Streamlit's ``st.markdown`` interprets ``$...$`` as inline math, which mangles
    common currency strings like ``$13.78, ... $-6.12`` into italicised, space-
    collapsed equations. We escape every unescaped ``$`` to ``\$``. This domain
    has no legitimate inline LaTeX, so the trade-off is safe.
    """
    if not text:
        return text
    return _DOLLAR_PATTERN.sub(r"\\$", text)


@st.dialog("Biorefinery Assistant", width="large")
def _open_chat_dialog(
    api_key: str,
    model: str,
    current_eval,
    snapshot: Dict[str, object],
) -> None:
    """Pop-up chat modal so the assistant's reply is visible without scrolling.

    Uses Streamlit's dialog (which is fragment-like) — when the user submits
    inside ``st.chat_input``, only this dialog re-renders, so the modal stays
    open with the new message visible at the top of the viewport.

    ``snapshot`` is the pre-built comprehensive context (computed once when the
    dialog is opened and reused across fragment reruns), so the LLM sees ALL
    production modes — every recovery method, feedstock, electricity case, and
    capacity — not just the active sidebar scenario.
    """
    if "groq_messages" not in st.session_state:
        st.session_state.groq_messages = []

    scenario = current_eval.foreground.scenario
    metrics = current_eval.tea.metrics
    lca_metrics = current_eval.lca.metrics
    st.caption(
        f"Model: `{model}`. Grounded in the full model "
        f"(active: {_display_pathway_label(scenario)}, "
        f"{int(scenario.annual_primary_product_tpy):,} t/y; the assistant can also "
        f"compare alternative recovery methods, feedstocks, capacities, and electricity cases)."
    )
    st.caption(
        "**Active TEA results from the Python model**  ·  "
        f"NPV {metrics['npv_usd'] / 1e6:.1f}M USD  ·  "
        f"Net LCOX {metrics['net_primary_lcox_usd_per_kg']:.2f} USD/kg NH3-eq  ·  "
        f"Gross LCOX {metrics['gross_primary_lcox_usd_per_kg']:.2f} USD/kg  ·  "
        f"GWP {lca_metrics['primary_product_gwp_kgco2e_per_kg']:.2f} kg CO2e/kg"
    )

    history = st.container(height=420)
    with history:
        if not st.session_state.groq_messages:
            st.caption(
                "Ask about the current scenario, which production mode is most profitable, "
                "which feedstock to use, or how a slider change would affect cost and GWP."
            )
        for message in st.session_state.groq_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    prompt = st.chat_input("Ask about the current scenario, outputs, or a hypothetical...")
    if prompt:
        st.session_state.groq_messages.append({"role": "user", "content": prompt})
        with history:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    raw_reply: str
                    grounded = deterministic_answer(prompt, snapshot)
                    if grounded:
                        raw_reply = (
                            "*Computed directly from the Python TEA model — no LLM was used "
                            "for these numbers.*\n\n" + grounded
                        )
                    else:
                        try:
                            llm_reply = ask_groq(
                                api_key=api_key,
                                question=prompt,
                                snapshot=snapshot,
                                history=st.session_state.groq_messages[:-1],
                                model=model,
                            )
                            raw_reply = (
                                "*LLM interpretation of the Python model results — verify any number "
                                "against the TEA panel above.*\n\n" + llm_reply
                            )
                        except Exception as exc:
                            msg = str(exc)
                            if "rate_limit_exceeded" in msg or "Request too large" in msg or "TPM" in msg:
                                raw_reply = (
                                    "**Groq rate limit hit.**  The free tier caps "
                                    "`llama-3.1-8b-instant` at 6,000 tokens per minute. "
                                    "Wait ~60 seconds and try again, or pick a higher-TPM model "
                                    "(e.g. `llama-3.3-70b-versatile`) from the model selector below."
                                )
                            else:
                                raw_reply = f"Groq request failed: {exc}"
                reply = _sanitize_chat_reply(raw_reply)
                st.markdown(reply)
        st.session_state.groq_messages.append({"role": "assistant", "content": reply})

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("Clear conversation", key="clear_groq_in_dialog", use_container_width=True):
            st.session_state.groq_messages = []
            st.rerun()
    with btn_col2:
        if st.button("Close", key="close_groq_dialog", type="primary", use_container_width=True):
            st.rerun()


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

    # Floating chat launcher — visible from anywhere on the page (CSS-pinned).
    # Resolve the chosen Groq model from session state with a sensible fallback,
    # since the model selector itself renders later in the page.
    _saved_choice = st.session_state.get("groq_model_choice", DEFAULT_GROQ_MODEL)
    if _saved_choice == "custom":
        _floating_model = st.session_state.get("custom_groq_model", DEFAULT_GROQ_MODEL)
    else:
        _floating_model = _saved_choice
    _chat_ready = groq_available() and bool(api_key)
    if st.button(
        "💬 Ask the assistant",
        key="float_open_chat",
        type="primary",
        disabled=not _chat_ready,
        help=("Ask the biorefinery assistant about the current scenario."
              if _chat_ready
              else "Add a Groq API key in the sidebar to enable chat."),
    ):
        _chat_snapshot = cached_chat_snapshot(current_config, overrides, _data_hash=_dhash)
        _open_chat_dialog(
            api_key=api_key,
            model=_floating_model,
            current_eval=current_eval,
            snapshot=_chat_snapshot,
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
        "Choose one figure at a time. The displayed figure uses the exact same model state shown in the cards above.",
    )
    available_figures = [
        fig_id for fig_id in figure_ids()
        if fig_id not in _EXCLUDED_FIGURES
    ]
    ordered_figures = [
        fig_id for fig_id in _FIGURE_DISPLAY_ORDER
        if fig_id in available_figures
    ] + [
        fig_id for fig_id in available_figures
        if fig_id not in _FIGURE_DISPLAY_ORDER
    ]
    selected_fig_id = st.selectbox(
        "Select figure",
        options=ordered_figures,
        index=0,
        format_func=lambda fig_id: _FIG_TITLES.get(fig_id, fig_id),
        key="selected_figure",
    )
    selected_title = _FIG_TITLES.get(selected_fig_id, selected_fig_id)
    st.markdown(f"### {selected_title}")
    _render_figure_methodology(selected_fig_id)
    with st.spinner(f"Rendering {selected_title}..."):
        try:
            fig = build_figure(
                selected_fig_id,
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
        "Fast Groq chat grounded in the current scenario state. Opens in a pop-up so the response stays in view.",
    )

    if "groq_messages" not in st.session_state:
        st.session_state.groq_messages = []
    chat_available = groq_available() and bool(api_key)

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

        # Inline "launcher" row: open-chat button + brief preview of the latest exchange.
        launch_col, preview_col = st.columns([1.0, 3.0])
        with launch_col:
            open_chat = st.button(
                "💬  Open chat",
                key="open_chat_dialog",
                type="primary",
                use_container_width=True,
                disabled=not chat_available,
                help=None if chat_available
                     else "Provide a Groq API key in the sidebar to enable chat.",
            )
        with preview_col:
            msgs = st.session_state.groq_messages
            if not msgs:
                st.caption(
                    "Click *Open chat* to ask about the current scenario, which route looks most attractive, "
                    "or how a slider change would affect cost and GWP. Responses appear in a pop-up window."
                )
            else:
                last_user = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
                last_asst = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")

                def _preview(text: str, n: int = 220) -> str:
                    text = " ".join(text.split())
                    return text if len(text) <= n else text[: n - 1].rstrip() + "…"

                if last_user:
                    st.markdown(f"**You:** {_preview(last_user, 160)}")
                if last_asst:
                    st.markdown(f"**Assistant:** {_preview(last_asst, 280)}")
                st.caption(f"{len(msgs)} message(s) in this session — click *Open chat* to continue.")

    if not groq_available():
        st.info("The `groq` package is not installed in this environment.")
    elif not api_key:
        st.info("Add a Groq API key in the sidebar or set `GROQ_API_KEY` in deployment secrets to enable chat.")

    if open_chat and chat_available:
        _chat_snapshot = cached_chat_snapshot(current_config, overrides, _data_hash=_dhash)
        _open_chat_dialog(
            api_key=api_key,
            model=groq_model,
            current_eval=current_eval,
            snapshot=_chat_snapshot,
        )


if __name__ == "__main__":
    main()
