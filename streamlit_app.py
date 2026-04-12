from __future__ import annotations

import os
from dataclasses import replace
from typing import Dict, List, Optional


def _secret(key: str) -> Optional[str]:
    """Return a Streamlit secret value, or None if not set."""
    try:
        val = st.secrets.get(key, None)
        return str(val) if val else None
    except Exception:
        return None


def _check_password() -> bool:
    """Return True if the user is authenticated (or no password is configured)."""
    required = _secret("APP_PASSWORD")
    if not required:
        return True  # no password configured → open access

    if st.session_state.get("authenticated"):
        return True

    st.markdown("## Formate Biorefinery Explorer")
    st.markdown("This app is password protected.")
    pw = st.text_input("Password", type="password", key="login_pw")
    if st.button("Enter"):
        if pw == required:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

import pandas as pd
import streamlit as st

from formate_biorefinery_model import (
    AmmoniaRecoveryMethod,
    CO2Source,
    ElectricityCase,
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
    page_title="Formate Biorefinery Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a cleaner feel ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Tighten expander headers */
    [data-testid="stExpander"] summary {
        font-weight: 600;
        font-size: 0.92rem;
    }
    /* Slightly smaller metric labels */
    [data-testid="stMetricLabel"] { font-size: 0.78rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.25rem !important; }
    /* Caption style for figure sub-labels */
    .fig-caption { font-size: 0.75rem; color: #888; margin-bottom: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Human-readable labels ──────────────────────────────────────────────────────
_FIG_TITLES = {
    "00_process_flow":        "Process Flow Diagram",
    "01_market_viability":    "Market Viability Overview",
    "02_cost_structure":      "Cost Structure",
    "03_scale_margin":        "Scale & Margin Curve",
    "04_cashflow_npv":        "Annual Cash-Flow & NPV",
    "05_cost_vs_gwp":         "Cost vs. GWP (LCA deep-dive)",
    "06_sensitivity_nh3":     "Sensitivity — NH\u2083 route",
    "07_sensitivity_urea":    "Sensitivity — Urea route",
    "08_executive_summary":   "Executive Summary",
    "09_nh3_recovery_compare":"NH\u2083 Recovery Method Comparison",
    "10_urea_recovery_compare":"Urea Recovery Method Comparison",
}

_NH3_LABELS = {m.value: m.value.replace("_", " ").title() for m in AmmoniaRecoveryMethod}
_UREA_LABELS = {m.value: m.value.replace("_", " ").title() for m in UreaRecoveryMethod}
_ELEC_LABELS = {m.value: m.value.replace("_", " ").title() for m in ElectricityCase}
_CO2_LABELS = {m.value: m.value.replace("_", " ").title() for m in CO2Source}


@st.cache_data(show_spinner=False)
def cached_evaluate(config: ScenarioConfig, overrides: Dict[str, float]):
    return evaluate_scenario(config, overrides=overrides)


@st.cache_data(show_spinner=False)
def cached_grid(
    base_config: ScenarioConfig,
    overrides: Dict[str, float],
    nh3_method: AmmoniaRecoveryMethod,
    urea_method: UreaRecoveryMethod,
):
    return evaluate_dashboard_grid(
        base_config,
        overrides=overrides,
        nh3_method=nh3_method,
        urea_method=urea_method,
    )


def _render_figure_methodology(fig_id: str) -> None:
    """Render a collapsed expander with the figure's calculation methodology."""
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
        "00_process_flow":         lambda: plot_process_flow(),
        "01_market_viability":     lambda: plot_market_viability_overview(grid_rows),
        "02_cost_structure":       lambda: plot_cost_structure(grid_rows, capacity_tpy=cap),
        "03_scale_margin":         lambda: plot_margin_curve(grid_rows),
        "04_cashflow_npv":         lambda: plot_annual_cashflow(overrides=overrides),
        "05_cost_vs_gwp":          lambda: plot_cost_vs_gwp(grid_rows, capacity_tpy=cap, overrides=overrides),
        "06_sensitivity_nh3":      lambda: plot_sensitivity_tornado(nh3_cfg),
        "07_sensitivity_urea":     lambda: plot_sensitivity_tornado(urea_cfg),
        "08_executive_summary":    lambda: plot_executive_summary(grid_rows),
        "09_nh3_recovery_compare": lambda: plot_nh3_recovery_comparison(capacity_tpy=cap, overrides=overrides),
        "10_urea_recovery_compare":lambda: plot_urea_recovery_comparison(capacity_tpy=cap, overrides=overrides),
    }
    return builders[fig_id]()


def main() -> None:
    # ── Password gate ──────────────────────────────────────────────────────────
    if not _check_password():
        st.stop()

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Scenario")

        category = ScenarioCategory(
            st.selectbox(
                "Production pathway",
                options=[item.value for item in ScenarioCategory],
                format_func=lambda v: "NH\u2083 + SCP" if v == ScenarioCategory.AMMONIA_SCP.value else "Urea + SCP",
                help="Primary product route for the single-scenario KPIs.",
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

        with st.expander("Recovery methods & grid settings", expanded=False):
            nh3_method = AmmoniaRecoveryMethod(
                st.selectbox(
                    "NH\u2083 recovery route",
                    options=[m.value for m in AmmoniaRecoveryMethod],
                    format_func=lambda v: _NH3_LABELS[v],
                    help="Used in cross-category comparison figures.",
                )
            )
            urea_method = UreaRecoveryMethod(
                st.selectbox(
                    "Urea recovery route",
                    options=[m.value for m in UreaRecoveryMethod],
                    format_func=lambda v: _UREA_LABELS[v],
                    help="Used in cross-category comparison figures.",
                )
            )
            electricity_case = ElectricityCase(
                st.selectbox(
                    "Electricity case",
                    options=[item.value for item in ElectricityCase],
                    format_func=lambda v: _ELEC_LABELS[v],
                )
            )
            co2_source = CO2Source(
                st.selectbox(
                    "CO\u2082 source",
                    options=[item.value for item in CO2Source],
                    format_func=lambda v: _CO2_LABELS[v],
                )
            )

        with st.expander("LCA credits", expanded=False):
            use_scp_credit = st.checkbox("SCP revenue credit", value=True,
                help="Include SCP co-product revenue in net LCOX.")
            use_biogenic_carbon_credit = st.checkbox("Biogenic carbon credit", value=True,
                help="Credit CO\u2082 captured in SCP biomass from a biogenic source.")
            use_scp_displacement_credit = st.checkbox("SCP displacement credit (system expansion)", value=False,
                help="System-expansion credit for displacing soy-meal protein production.")

        st.markdown("---")
        st.markdown("## Inputs")

        slider_defaults = all_slider_defaults()
        slider_groups = all_slider_specs()
        overrides: Dict[str, float] = {}

        for group_name, specs in slider_groups.items():
            with st.expander(group_name, expanded=(group_name == "Biological performance")):
                for spec in specs:
                    lbl = f"{spec.label}  [{spec.unit}]" if spec.unit else spec.label
                    value = float(
                        st.slider(
                            lbl,
                            min_value=float(spec.min_value),
                            max_value=float(spec.max_value),
                            value=float(slider_defaults[spec.key]),
                            step=float(spec.step),
                            help=spec.help_text,
                            key=f"slider_{spec.key}",
                        )
                    )
                    overrides[spec.key] = value

        st.markdown("---")
        st.markdown("## Groq assistant")
        # Use secret or env var automatically; only show input field as fallback.
        _auto_key = _secret("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "")
        if _auto_key:
            api_key = _auto_key
            st.caption("\U0001F916 Groq key loaded from deployment secrets.")
        else:
            api_key = st.text_input(
                "API key",
                value="",
                type="password",
                help="Paste your Groq API key here, or set GROQ_API_KEY in Streamlit secrets.",
            )
        groq_model = st.text_input("Model", value=DEFAULT_GROQ_MODEL)

    # ── Build scenario ─────────────────────────────────────────────────────────
    current_config = ScenarioConfig(
        category=category,
        annual_primary_product_tpy=capacity_tpy,
        electricity_case=electricity_case,
        ammonia_recovery_method=nh3_method if category == ScenarioCategory.AMMONIA_SCP else AmmoniaRecoveryMethod.VACUUM_STRIPPING,
        urea_recovery_method=urea_method if category == ScenarioCategory.BIO_UREA_SCP else UreaRecoveryMethod.EVAPORATION,
        use_scp_credit=use_scp_credit,
        co2_source=co2_source,
        use_biogenic_carbon_credit=use_biogenic_carbon_credit,
        use_scp_displacement_credit=use_scp_displacement_credit,
        user_overrides=overrides,
    )

    current_eval = cached_evaluate(current_config, overrides)
    comparison_grid = cached_grid(current_config, overrides, nh3_method=nh3_method, urea_method=urea_method)
    display_sources = source_rows_for_display(current_eval.source_rows)
    reference_rows = reference_summary(current_eval.source_rows)

    # ── Page header ────────────────────────────────────────────────────────────
    st.title("Formate Biorefinery TEA / LCA Explorer")
    st.caption(
        "Interactively vary biological performance, input costs, financing, and labor; "
        "inspect every model assumption and its source; ask a grounded Groq model about "
        "the current scenario or plausible alternatives."
    )

    overview_tab, figures_tab, sources_tab = st.tabs(
        ["\u2b50 Overview", "\U0001F4CA Figures", "\U0001F4DA Sources & References"]
    )

    # ── OVERVIEW ───────────────────────────────────────────────────────────────
    with overview_tab:
        with st.expander("Active scenario configuration", expanded=False):
            st.json(current_config_summary(current_config), expanded=True)

        st.markdown("### Key performance indicators")
        metric_cols = st.columns(3)
        for idx, card in enumerate(kpi_cards(current_eval)):
            metric_cols[idx % 3].metric(card["label"], card["value"])

        st.markdown("### Cross-category comparison grid")
        st.caption(
            "Three capacity points (100 / 1 000 / 10 000 t/y) for both routes "
            "under the current slider settings."
        )
        summary_rows = [row.to_dict() for row in comparison_grid]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── FIGURES ────────────────────────────────────────────────────────────────
    with figures_tab:
        st.caption(
            "Each figure is rendered from the current slider values. "
        "Expand any panel to view it, and open the 'Methodology' "
        "section inside for calculation details."
        )
        all_fig_ids = figure_ids()
        for fig_id in all_fig_ids:
            title = _FIG_TITLES.get(fig_id, fig_id)
            with st.expander(f"\U0001F4C8  {title}", expanded=False):
                with st.spinner(f"Rendering {title}\u2026"):
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

    # ── SOURCES & REFERENCES ──────────────────────────────────────────────────
    with sources_tab:
        st.markdown(
            "Every active model input is traced to a specific dataset and publication year. "
            "Inputs overridden by a slider are flagged **is_override = True**."
        )

        with st.expander("All inputs — flat provenance table", expanded=False):
            df_src = pd.DataFrame(display_sources)
            st.dataframe(df_src, use_container_width=True, hide_index=True)

        st.markdown("### By reference")
        st.caption("Expand a reference to see which numbers were drawn from it.")
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

    # ── FLOATING GROQ CHAT ─────────────────────────────────────────────────────
    # Inject CSS that styles the chat drawer to look like a floating widget.
    st.markdown(
        """
        <style>
        /* Chat drawer card */
        div[data-testid="stExpander"].chat-drawer > details {
            border: 1.5px solid #e0e0e0;
            border-radius: 16px 16px 0 0;
            box-shadow: 0 -4px 24px rgba(0,0,0,0.10);
            background: #fafafa;
        }
        div[data-testid="stExpander"].chat-drawer > details > summary {
            background: #1a1a2e;
            color: white;
            border-radius: 12px 12px 0 0;
            padding: 10px 18px;
            font-weight: 700;
            font-size: 0.95rem;
            letter-spacing: 0.01em;
        }
        div[data-testid="stExpander"].chat-drawer > details > summary:hover {
            background: #16213e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sticky spacer so the chat drawer is visually separated from the tabs above
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    chat_available = groq_available() and bool(api_key)
    n_msgs = len(st.session_state.get("groq_messages", []))
    badge = f" · {n_msgs} messages" if n_msgs else ""
    header = f"\U0001F916  Groq AI Assistant{badge}"
    if not chat_available:
        header += "  —  add API key in sidebar to enable"

    with st.expander(header, expanded=False):
        if not groq_available():
            st.warning("`groq` package not installed — run `pip install groq`.")
        elif not api_key:
            st.info("Enter your Groq API key in the sidebar (**Groq assistant** section) to start chatting.")
        else:
            st.caption(
                "Grounded in the current scenario, KPIs, and source records. "
                "Ask about the current state or explore hypotheticals."
            )

            if "groq_messages" not in st.session_state:
                st.session_state.groq_messages = []

            # Scrollable history
            history_container = st.container(height=380)
            with history_container:
                if not st.session_state.groq_messages:
                    st.markdown(
                        "<div style='color:#aaa;text-align:center;padding:40px 0'>"
                        "No messages yet — type below to start.</div>",
                        unsafe_allow_html=True,
                    )
                for message in st.session_state.groq_messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # Clear button
            if st.session_state.groq_messages:
                if st.button("Clear conversation", key="clear_groq"):
                    st.session_state.groq_messages = []
                    st.rerun()

    # chat_input is naturally sticky at the very bottom of the Streamlit page.
    # Only render it when Groq is configured so it doesn't appear when unused.
    if chat_available:
        prompt = st.chat_input(
            "\U0001F916  Ask Groq about the current scenario or explore a hypothetical...",
        )
        if prompt:
            if "groq_messages" not in st.session_state:
                st.session_state.groq_messages = []
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


if __name__ == "__main__":
    main()
