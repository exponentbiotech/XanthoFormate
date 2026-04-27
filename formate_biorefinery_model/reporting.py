from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
    from matplotlib.lines import Line2D
except ModuleNotFoundError:  # pragma: no cover
    plt = None
    mticker = None

from .config import DATA_DIR, AmmoniaRecoveryMethod, ScenarioCategory, ScenarioConfig, UreaRecoveryMethod
from .run_scenarios import (
    FAVORABLE_LCA_SCENARIO_UPDATES,
    NH3_BEST_METHODS,
    RECOVERY_METHOD_LABELS,
    UREA_BEST_METHODS,
    ScenarioEvaluation,
    run_best_methods_grid,
    run_best_methods_negative_gwp_grid,
    run_lca_sensitivity_grid,
    run_recovery_comparison,
    run_sensitivity_cases,
)


# ═══════════════════════════════════════════════════════════════════════════
#  PALETTE  &  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

CAT_COLOR = {
    ScenarioCategory.AMMONIA_SCP: "#0072B2",
    ScenarioCategory.BIO_UREA_SCP: "#009E73",
}
CAT_LABEL = {
    ScenarioCategory.AMMONIA_SCP: "NH\u2083 + SCP",
    ScenarioCategory.BIO_UREA_SCP: "Urea + SCP",
}


def _scenario_label_from_config(config: ScenarioConfig) -> str:
    """Return a route-aware primary product label for displays and legends."""
    if config.category == ScenarioCategory.AMMONIA_SCP:
        if config.ammonia_recovery_method == AmmoniaRecoveryMethod.STRUVITE_MAP:
            return "Struvite + SCP"
        if config.ammonia_recovery_method == AmmoniaRecoveryMethod.MAP_FERTILIZER:
            return "MAP fertilizer + SCP"
        return "NH₃ + SCP"
    return "Urea + SCP"


def _scenario_label(row: ScenarioEvaluation) -> str:
    return _scenario_label_from_config(row.foreground.scenario)


INCUMBENT_BENCHMARKS = {
    "Haber\u2013Bosch NH\u2083 (US Gulf)": 0.60,
    "Industrial urea (granular)": 0.40,
}

SCP_BENCHMARKS = {
    "Soy protein conc.": 1.00,
    "Fishmeal": 1.75,
    "Spirulina (bulk)": 8.00,
}

COST_GROUPS = {
    "Electricity": ("electricity",),
    "Steam": ("steam",),
    "Water / WW": ("water", "wastewater"),
    "Chemicals": ("co2", "naoh", "h2so4", "h3po4", "mgcl2", "membrane_replacement"),
    "Direct labor": ("direct_labor",),
    "Overhead": ("overhead",),
    "Maintenance": ("maintenance",),
    "Stack repl.": ("stack_replacement",),
    "Annualized CapEx": ("annualized_capex",),
}

# Credit keys expected in tea.credits_usd_per_y
_CREDIT_KEYS = ("scp_credit", "h2_credit", "co2_credit", "struvite_credit")
COST_PALETTE = [
    "#4E79A7", "#A0CBE8", "#76B7B2", "#F28E2B",
    "#59A14F", "#8CD17D", "#499894", "#86BCB6", "#79706E",
]
CREDIT_COLOR = "#2E8B57"

# Color + marker scheme for each "best method" series in the NPV comparison
BEST_METHOD_STYLE: Dict[str, Dict[str, object]] = {
    AmmoniaRecoveryMethod.STRUVITE_MAP.value:     {"color": "#2E8B57", "marker": "o",  "ls": "-",  "label": "NH\u2083 | Struvite (MgNH\u2084PO\u2084)"},
    AmmoniaRecoveryMethod.MEMBRANE.value:         {"color": "#0072B2", "marker": "s",  "ls": "-",  "label": "NH\u2083 | Hollow-fiber membrane"},
    AmmoniaRecoveryMethod.MAP_FERTILIZER.value:   {"color": "#56B4E9", "marker": "^",  "ls": "--", "label": "NH\u2083 | MAP fertilizer (11-52-0)"},
    UreaRecoveryMethod.MVR_CRYSTALLIZATION.value: {"color": "#E69F00", "marker": "D",  "ls": "-",  "label": "Urea | MVR + crystallization"},
    UreaRecoveryMethod.EVAPORATION.value:         {"color": "#CC79A7", "marker": "v",  "ls": "--", "label": "Urea | Single-effect evaporation"},
}

_FIGURE_METADATA_PATH = DATA_DIR / "figure_metadata.json"

SENSITIVITY_PARAMS = {
    ScenarioCategory.AMMONIA_SCP: [
        ("Formate : NH\u2083 ratio", "formate_to_ammonia_kg_per_kg"),
        ("SCP co-product yield", "scp_to_ammonia_kg_per_kg"),
        ("NH\u2083 recovery eff.", "ammonia_recovery_efficiency"),
        ("Electricity price", "electricity_price_usd_per_kwh"),
        ("CO\u2082 price", "co2_price_usd_per_kg"),
        ("SCP market price", "scp_market_price_usd_per_kg"),
        ("Electrolyzer CapEx", "electrolyzer_installed_cost_usd_per_kw"),
        ("Aeration intensity", "agitation_aeration_kwh_per_m3_h"),
        ("Operator wage", "operator_loaded_wage_usd_per_h"),
    ],
    ScenarioCategory.BIO_UREA_SCP: [
        ("Formate : urea ratio", "formate_to_urea_kg_per_kg"),
        ("SCP co-product yield", "scp_to_urea_kg_per_kg"),
        ("Urea recovery eff.", "urea_recovery_efficiency"),
        ("Electricity price", "electricity_price_usd_per_kwh"),
        ("Steam price", "steam_price_usd_per_kg"),
        ("SCP market price", "scp_market_price_usd_per_kg"),
        ("Electrolyzer CapEx", "electrolyzer_installed_cost_usd_per_kw"),
        ("Evap. steam demand", "evaporation_steam_kg_per_kg_urea"),
        ("Operator wage", "operator_loaded_wage_usd_per_h"),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
#  STYLE
# ═══════════════════════════════════════════════════════════════════════════

def _require_mpl() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting — pip install matplotlib")


def _style() -> None:
    _require_mpl()
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 320, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "600",
        "axes.labelsize": 13.5, "axes.edgecolor": "#444444", "axes.linewidth": 0.7,
        "axes.grid": True, "axes.axisbelow": True, "axes.facecolor": "#FAFAFA",
        "grid.alpha": 0.25, "grid.linewidth": 0.45, "grid.color": "#CCCCCC",
        "figure.facecolor": "white",
        "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC", "legend.fontsize": 11,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
    })


_STAMP = (
    "Formate biorefinery screening TEA / LCA  \u00b7  "
    "US-grounded defaults (EIA, EPA eGRID, NREL, World Bank)  \u00b7  "
    "X. flavus GJ10 biological assumptions are literature proxies"
)


def _stamp(fig) -> None:
    fig.text(0.5, -0.025, _STAMP, ha="center", fontsize=6.8,
             color="#888888", style="italic")


def load_figure_metadata() -> Dict[str, Dict[str, object]]:
    """Load figure calculation descriptions for app display."""
    with _FIGURE_METADATA_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def figure_metadata(name: str) -> Dict[str, object]:
    """Return metadata for one figure by stable id."""
    return load_figure_metadata().get(name, {})


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _sorted(evals: Iterable[ScenarioEvaluation]) -> List[ScenarioEvaluation]:
    return sorted(evals, key=lambda r: (r.foreground.scenario.category.value,
                                        r.foreground.scenario.annual_primary_product_tpy))


def _by_cat(evals: Iterable[ScenarioEvaluation],
            cat: ScenarioCategory) -> List[ScenarioEvaluation]:
    return [r for r in _sorted(evals) if r.foreground.scenario.category == cat]


def _cap_label(r: ScenarioEvaluation) -> str:
    return f"{int(r.foreground.scenario.annual_primary_product_tpy):,}"


def _market(r: ScenarioEvaluation) -> float:
    return r.tea.metrics["benchmark_primary_revenue_usd_per_y"] / max(
        1e-9, r.foreground.sellable_primary_product_kg)


def _market_product_label(r: ScenarioEvaluation) -> str:
    """Short label describing *what* is sold as the primary product."""
    method = r.foreground.scenario.ammonia_recovery_method
    cat = r.foreground.scenario.category
    if cat == ScenarioCategory.AMMONIA_SCP:
        if method == AmmoniaRecoveryMethod.STRUVITE_MAP:
            return "struvite equiv."
        if method == AmmoniaRecoveryMethod.MAP_FERTILIZER:
            return "MAP fert. equiv."
        return "NH\u2083 commodity"
    return "urea commodity"


def _usd(v: float, decimals: int = 2) -> str:
    return f"${v:,.{decimals}f}"


def _add_benchmarks(ax, benchmarks: Dict[str, float], xmax: float,
                    side: str = "right") -> None:
    for label, price in benchmarks.items():
        ax.axhline(price, ls="--", lw=0.65, color="#AAAAAA", zorder=1)
        ha = "right" if side == "right" else "left"
        xpos = xmax if side == "right" else 0
        ax.text(xpos, price, f"  {label}  ({_usd(price)})",
                fontsize=6.8, color="#777777", ha=ha, va="bottom")


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 0 — PROCESS FLOW DIAGRAM
# ═══════════════════════════════════════════════════════════════════════════

def plot_process_flow() -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(17.5, 7.0))
    ax.set_xlim(-0.5, 18.5)
    ax.set_ylim(0, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")

    C = {"input": "#E0E0E0", "electro": "#D4E6F1", "bio": "#C8E6C9",
         "recovery": "#FFF9C4", "product_n": "#82E0AA", "product_scp": "#F9E79F",
         "arrow": "#555555"}

    def _box(x, y, w, h, txt, color, fs=8.0):
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.10", facecolor=color,
                             edgecolor="#444444", linewidth=1.1, zorder=2)
        ax.add_patch(box)
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
                fontweight="600", zorder=3, linespacing=1.3)

    def _arr(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                            color=C["arrow"], linewidth=1.4,
                            mutation_scale=13, zorder=1)
        ax.add_patch(a)

    bw, bh = 2.3, 1.2
    top = 5.5
    bot = 2.2
    _box(1.5,  top, bw, bh, "CO\u2082 + H\u2082O\n(inputs)", C["input"])
    _box(4.7,  top, bw, bh, "Electrolyzer\n(CO\u2082 to formate\n+ H\u2082 byproduct)", C["electro"], fs=7.5)
    _box(8.0,  top, bw, bh, "Bioreactor\nX. flavus GJ10\n(formate to NH\u2083 or\nurea + biomass)", C["bio"], fs=7.2)
    _box(11.5, top, bw, bh, "N-Product\nRecovery\n(stripping / evap.)", C["recovery"], fs=7.5)
    _box(15.0, top, bw, bh, "NH\u2083 or Urea\n(sellable)", C["product_n"])

    _box(8.0,  bot, bw, bh, "Cell Harvest\n(centrifuge)", C["electro"])
    _box(11.5, bot, bw, bh, "SCP Drying\n(belt / spray)", C["recovery"])
    _box(15.0, bot, bw, bh, "SCP Powder\n(animal feed\n~50\u201355% protein)", C["product_scp"], fs=7.5)

    for x1, x2 in [(2.65, 3.55), (5.85, 6.85), (9.15, 10.35), (12.65, 13.85)]:
        _arr(x1, top, x2, top)
    _arr(8.0, top - bh / 2 - 0.08, 8.0, bot + bh / 2 + 0.08)
    for x1, x2 in [(9.15, 10.35), (12.65, 13.85)]:
        _arr(x1, bot, x2, bot)

    ax.text(7.15, (top + bot) / 2, "Residual\nbiomass",
            ha="center", va="center", fontsize=7.5, color="#666666", style="italic")

    ax.text(9.0, 7.0,
            "Formate Biorefinery \u2014 Electrolytic CO\u2082-to-Formate | NH\u2083 or Urea + SCP",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(9.0, 6.55,
            "Screening process flow for X. flavus GJ10 platform",
            ha="center", fontsize=10, color="#555555")

    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C["input"],     markersize=11, label="Inputs"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C["electro"],   markersize=11, label="Electrolysis"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C["bio"],       markersize=11, label="Fermentation"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C["recovery"],  markersize=11, label="Recovery / drying"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C["product_n"], markersize=11, label="N-product"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C["product_scp"], markersize=11, label="SCP"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", fontsize=8.5,
              framealpha=0.92, ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.06)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 1 — MARKET VIABILITY  (grouped bars + benchmark lines)
# ═══════════════════════════════════════════════════════════════════════════

def plot_market_viability_overview(evaluations: Iterable[ScenarioEvaluation]) -> plt.Figure:
    _style()
    rows_all = _sorted(evaluations)
    categories = [ScenarioCategory.AMMONIA_SCP, ScenarioCategory.BIO_UREA_SCP]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), sharey=False)

    for ax, cat in zip(axes, categories):
        rows = _by_cat(rows_all, cat)
        n = len(rows)
        x = np.arange(n)
        w = 0.30
        gross = [r.tea.metrics["gross_primary_lcox_usd_per_kg"] for r in rows]
        net   = [r.tea.metrics["net_primary_lcox_usd_per_kg"] for r in rows]
        mkt   = [_market(r) for r in rows]

        ax.bar(x - w / 2, gross, w, label="Gross LCOX (before credits)",
               color="#D5D5D5", edgecolor="white", linewidth=0.5, zorder=3)
        ax.bar(x + w / 2, net, w, label="Net LCOX (after SCP credit)",
               color=CAT_COLOR[cat], edgecolor="white", linewidth=0.5, zorder=4)

        ax.plot(x, mkt, color="#E15759", marker="o", markersize=7,
                linewidth=2.0, label="Product revenue / kg primary", zorder=5)
        for i, (m, row) in enumerate(zip(mkt, rows)):
            prod_lbl = _market_product_label(row)
            ax.annotate(
                f"{_usd(m)}\n({prod_lbl})",
                (i, m),
                textcoords="offset points",
                xytext=(0, 7),
                fontsize=7.0,
                color="#E15759",
                ha="center",
                fontweight="600",
                multialignment="center",
            )

        for i in range(n):
            ax.text(i - w / 2, gross[i] + 0.08, _usd(gross[i], 1),
                    ha="center", va="bottom", fontsize=6.5, color="#777777")
            ax.text(i + w / 2, net[i] + 0.08, _usd(net[i], 1),
                    ha="center", va="bottom", fontsize=7, fontweight="600",
                    color=CAT_COLOR[cat])

        ax.set_xticks(x)
        ax.set_xticklabels([f"{_cap_label(r)} t/y" for r in rows])
        ax.set_xlabel("Primary product capacity")
        ax.set_ylabel("USD / kg primary product")
        ax.set_title(_scenario_label(rows[0]) if rows else CAT_LABEL[cat], pad=12)
        ax.legend(fontsize=7.5, loc="upper right")

        ymax = ax.get_ylim()[1]
        relevant = (INCUMBENT_BENCHMARKS if cat == ScenarioCategory.AMMONIA_SCP
                    else {k: v for k, v in INCUMBENT_BENCHMARKS.items() if "urea" in k.lower()})
        _add_benchmarks(ax, relevant, n - 0.3)

    fig.suptitle("Market Viability Overview",
                 fontsize=15, fontweight="bold", y=1.03)
    fig.subplots_adjust(wspace=0.14)
    _stamp(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 2 — COST STRUCTURE  (horizontal waterfall)
# ═══════════════════════════════════════════════════════════════════════════

def plot_cost_structure(evaluations: Iterable[ScenarioEvaluation],
                        capacity_tpy: float = 1_000.0) -> plt.Figure:
    _style()
    subset = [r for r in _sorted(evaluations)
              if math.isclose(r.foreground.scenario.annual_primary_product_tpy,
                              capacity_tpy)]
    if not subset:
        subset = list(_sorted(evaluations))[:2]

    fig, axes = plt.subplots(1, len(subset), figsize=(7.5 * len(subset), 6.5))
    if len(subset) == 1:
        axes = [axes]

    for ax, row in zip(axes, subset):
        kg = row.foreground.sellable_primary_product_kg

        # ── Build (label, value) pairs explicitly ──────────────────────────
        cost_rows: List[tuple] = []
        for group_label, keys in COST_GROUPS.items():
            total = 0.0
            for key in keys:
                total += row.tea.variable_opex_usd_per_y.get(key, 0.0) / kg
                total += row.tea.fixed_opex_usd_per_y.get(key, 0.0) / kg
            if abs(total) > 1e-6:
                cost_rows.append((group_label, total))

        credit_rows: List[tuple] = []
        for credit_key, credit_label in [
            ("struvite_credit", "Struvite credit"),
            ("map_fert_credit", "MAP fert. credit"),
            ("scp_credit", "SCP credit"),
            ("h2_credit", "H2 credit"),
            ("co2_credit", "CO2 credit"),
        ]:
            val = row.tea.credits_usd_per_y.get(credit_key, 0.0) / kg
            if val > 1e-6:
                credit_rows.append((credit_label, -val))

        # Costs sorted largest-first, then credits sorted most-negative-first.
        cost_rows.sort(key=lambda t: t[1], reverse=True)
        credit_rows.sort(key=lambda t: t[1])
        all_rows = cost_rows + credit_rows

        labels = [r[0] for r in all_rows]
        values = [r[1] for r in all_rows]
        y_pos = np.arange(len(labels))

        cost_idx = 0
        colors = []
        for v in values:
            if v < 0:
                colors.append(CREDIT_COLOR)
            else:
                colors.append(COST_PALETTE[cost_idx % len(COST_PALETTE)])
                cost_idx += 1

        bars = ax.barh(y_pos, values, height=0.56, color=colors,
                       edgecolor="white", linewidth=0.5, zorder=3)

        for bar, lbl, val in zip(bars, labels, values):
            cy = bar.get_y() + bar.get_height() / 2
            if val < 0:
                ax.text(val - 0.08, cy,
                        f"\u2212{_usd(abs(val))}",
                        ha="right", va="center", fontsize=8,
                        fontweight="600", color=CREDIT_COLOR)
            else:
                ax.text(val + 0.08, cy,
                        f"{_usd(val)}",
                        ha="left", va="center", fontsize=8,
                        fontweight="600", color="#333333")

        gross = row.tea.metrics["gross_primary_lcox_usd_per_kg"]
        net = row.tea.metrics["net_primary_lcox_usd_per_kg"]
        ax.axvline(gross, color="#AAAAAA", ls=":", lw=1.0, zorder=2)
        ax.axvline(net, color=CREDIT_COLOR, ls="-", lw=1.3, zorder=2)

        xlo, xhi = ax.get_xlim()
        pad_r = (xhi - xlo) * 0.15
        pad_l = (xhi - xlo) * 0.18
        ax.set_xlim(xlo - pad_l, xhi + pad_r)

        ax.text(xhi + pad_r * 0.7, 0.15, f"Gross {_usd(gross)}", fontsize=7.5,
                ha="right", va="bottom", color="#666666")
        ax.text(xhi + pad_r * 0.7, 0.65, f"Net {_usd(net)}", fontsize=8.5,
                ha="right", va="bottom", color=CREDIT_COLOR, fontweight="600")

        ax.axvline(0, color="#555555", lw=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.invert_yaxis()
        ax.set_xlabel("USD / kg primary product")
        ax.set_title(f"{_scenario_label(row)}  "
                     f"({_cap_label(row)} t/y)", pad=12)

    fig.suptitle(f"Cost Structure Waterfall \u2014 {int(capacity_tpy):,} t/y",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.16, wspace=0.08)
    _stamp(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 3 — SCALE CURVE  (LCOX vs capacity with margin shading)
# ═══════════════════════════════════════════════════════════════════════════

def plot_margin_curve(evaluations: Iterable[ScenarioEvaluation]) -> plt.Figure:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), sharey=False)

    for ax, cat in zip(axes, [ScenarioCategory.AMMONIA_SCP, ScenarioCategory.BIO_UREA_SCP]):
        rows = _by_cat(evaluations, cat)
        caps = np.array([r.foreground.scenario.annual_primary_product_tpy for r in rows])
        net = np.array([r.tea.metrics["net_primary_lcox_usd_per_kg"] for r in rows])
        mkt = np.array([_market(r) for r in rows])

        ax.plot(caps, net, marker="o", markersize=8, linewidth=2.5,
                color=CAT_COLOR[cat], label="Net LCOX", zorder=4)
        ax.plot(caps, mkt, marker="s", markersize=6, linewidth=1.6,
                linestyle="--", color="#E15759", label="Market price", zorder=3)

        ax.fill_between(caps, net, mkt, where=mkt >= net,
                        alpha=0.15, color="#2E8B57", label="Profitable region")
        ax.fill_between(caps, net, mkt, where=mkt < net,
                        alpha=0.12, color="#E15759", label="Below market")

        for c, n, m in zip(caps, net, mkt):
            margin = m - n
            color = "#2E8B57" if margin >= 0 else "#E15759"
            ax.annotate(f"{margin:+.2f}/kg", (c, n), textcoords="offset points",
                        xytext=(8, -6), fontsize=7.5, fontweight="600",
                        color=color)

        ax.set_xscale("log")
        if mticker is not None:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{int(x):,}"))
        ax.set_xticks(caps)
        ax.set_xlabel("Primary product capacity (t/y)")
        ax.set_ylabel("USD / kg primary product")
        ax.set_title(
            f"Economies of Scale \u2014 {(_scenario_label(rows[0]) if rows else CAT_LABEL[cat])}",
            pad=12,
        )
        ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle("Scale-Dependent Margin to Market",
                 fontsize=15, fontweight="bold", y=1.02)
    _stamp(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 4 — ANNUAL CASH FLOW & NPV
# ═══════════════════════════════════════════════════════════════════════════

def plot_annual_cashflow(
    evaluations: Optional[Iterable[ScenarioEvaluation]] = None,
    capacities: Optional[List[float]] = None,
    overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """NPV + cash flow comparison across the best NH3 and urea recovery methods.

    Automatically runs the curated method grid if `evaluations` is not provided.
    Three-panel layout:
      a) NPV vs capacity (log scale) — all best methods on one axes for direct comparison
      b) Annual cash flow at mid-scale (1 000 t/y)
      c) Total CapEx vs capacity
    """
    _style()
    caps = capacities or [100.0, 1_000.0, 10_000.0]

    # Build the dataset: one evaluation per (method × capacity)
    if evaluations is not None:
        all_rows = list(evaluations)
    else:
        all_rows = run_best_methods_grid(capacities=caps, overrides=overrides)

    def _method_key(r: ScenarioEvaluation) -> str:
        cat = r.foreground.scenario.category
        if cat == ScenarioCategory.AMMONIA_SCP:
            return r.foreground.scenario.ammonia_recovery_method.value
        return r.foreground.scenario.urea_recovery_method.value

    # Group by method key, sort each group by capacity
    from collections import defaultdict
    groups: Dict[str, List[ScenarioEvaluation]] = defaultdict(list)
    for r in all_rows:
        groups[_method_key(r)].append(r)
    for rows in groups.values():
        rows.sort(key=lambda r: r.foreground.scenario.annual_primary_product_tpy)

    # --- layout ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    ax_npv, ax_cf, ax_capex = axes

    # helper to get style for a method key
    def _sty(key: str) -> Dict[str, object]:
        return BEST_METHOD_STYLE.get(key, {"color": "#AAAAAA", "marker": "o", "ls": "-", "label": key})

    ref_cap = 1_000.0

    # ---- Panel a: NPV vs capacity ----
    for key, rows in groups.items():
        sty = _sty(key)
        cap_x = [r.foreground.scenario.annual_primary_product_tpy for r in rows]
        npv_y = [r.tea.metrics["npv_usd"] / 1e6 for r in rows]
        ax_npv.plot(cap_x, npv_y, color=sty["color"], marker=sty["marker"],
                    ls=sty["ls"], linewidth=2.2, markersize=7,
                    label=sty["label"], zorder=3)

        # Annotate the rightmost point
        ax_npv.annotate(f"{npv_y[-1]:+.0f}M",
                        (cap_x[-1], npv_y[-1]),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=7, color=sty["color"], fontweight="600")

    ax_npv.axhline(0, color="#555555", lw=0.8, ls=":")
    ax_npv.set_xscale("log")
    if mticker:
        ax_npv.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_npv.set_xticks(caps)
    ax_npv.set_xlabel("Primary product capacity (t/y)")
    ax_npv.set_ylabel("NPV (M USD, 20 y, 10%)")
    ax_npv.set_title("a)  NPV by Scale", pad=12)
    ax_npv.legend(fontsize=7.5, loc="lower right")

    # ---- Panel b: Annual cash flow at reference capacity ----
    cf_keys, cf_vals, cf_colors = [], [], []
    for key, rows in groups.items():
        match = [r for r in rows
                 if math.isclose(r.foreground.scenario.annual_primary_product_tpy, ref_cap)]
        if not match:
            continue
        r = match[0]
        cf_keys.append(key)
        cf_vals.append(r.tea.metrics["annual_cash_flow_usd_per_y"] / 1e6)
        cf_colors.append(_sty(key)["color"])

    y_cf = np.arange(len(cf_keys))
    bars = ax_cf.barh(y_cf, cf_vals, height=0.58,
                      color=cf_colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax_cf.axvline(0, color="#555555", lw=0.7)
    for bar, val in zip(bars, cf_vals):
        ha = "left" if val >= 0 else "right"
        offset = 0.08 if val >= 0 else -0.08
        ax_cf.text(val + offset, bar.get_y() + bar.get_height() / 2,
                   f"{val:+.1f}M", ha=ha, va="center", fontsize=8, fontweight="700",
                   color=_sty(cf_keys[y_cf.tolist().index(bar.get_y() + bar.get_height() / 2 - 0.0001)]
                              if False else cf_keys[int(bar.get_y() + bar.get_height() / 2 + 0.01)])
                   if False else "#333333")
    short_labels = [RECOVERY_METHOD_LABELS.get(k, k).split("\n")[0] for k in cf_keys]
    ax_cf.set_yticks(y_cf)
    ax_cf.set_yticklabels(short_labels, fontsize=9)
    ax_cf.invert_yaxis()
    ax_cf.set_xlabel("Annual cash flow (M USD/y)")
    ax_cf.set_title(f"b)  Cash Flow at {int(ref_cap):,} t/y", pad=12)

    # ---- Panel c: CapEx vs capacity ----
    for key, rows in groups.items():
        sty = _sty(key)
        cap_x = [r.foreground.scenario.annual_primary_product_tpy for r in rows]
        capex_y = [r.tea.metrics["total_capital_usd"] / 1e6 for r in rows]
        ax_capex.plot(cap_x, capex_y, color=sty["color"], marker=sty["marker"],
                      ls=sty["ls"], linewidth=2.0, markersize=6, zorder=3)

    ax_capex.set_xscale("log")
    ax_capex.set_yscale("log")
    if mticker:
        ax_capex.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax_capex.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}M"))
    ax_capex.set_xticks(caps)
    ax_capex.set_xlabel("Primary product capacity (t/y)")
    ax_capex.set_ylabel("Total capital (M USD)")
    ax_capex.set_title("c)  Capital Investment", pad=12)

    fig.suptitle("Best-Method NPV & Capital Comparison \u2014 NH\u2083 vs Urea Recovery Routes",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    _stamp(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 5 — COST vs GWP  (bubble chart)
# ═══════════════════════════════════════════════════════════════════════════

def plot_cost_vs_gwp(
    evaluations: Optional[Iterable[ScenarioEvaluation]] = None,
    capacity_tpy: float = 1_000.0,
    overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """LCA deep-dive: waterfall by accounting scenario + cost vs GWP scatter.

    Three-panel layout
    ------------------
    a) GWP waterfall bars for NH₃ and urea pathways across four LCA scenarios:
         1. Grid power / fossil CO₂ / no biogenic credits  (worst case)
         2. Grid power / biogenic CO₂ / biogenic-C credit  (correct design basis)
         3. Renewable power / biogenic CO₂ / biogenic-C credit
         4. Renewable + biogenic + protein displacement credit (system expansion)
       Each bar is split into burden components and credit components.
    b) Cost vs GWP scatter — uses the same curated best-method families as fig 04,
       but only retains the most attractive net-negative-GWP case for each method.
    c) GWP contribution stacked bar at design-basis (1 000 t/y, grid, biogenic CO₂)
       showing which inputs drive the carbon footprint.
    """
    from .run_scenarios import run_lca_sensitivity_grid  # avoid circular at module level

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(24, 9.5))
    ax_wf, ax_sc, ax_stk = axes

    # ---------- data ----------
    lca_rows = run_lca_sensitivity_grid(capacity_tpy=capacity_tpy, overrides=overrides)

    scenario_labels = [
        "Grid\nFossil CO\u2082\nno credits",
        "Grid\nBiogenic CO\u2082\n+ C credit",
        "Renewable\nBiogenic CO\u2082\n+ C credit",
        "Renewable\nBiogenic CO\u2082\n+ C + protein\ncredit",
    ]

    # Separate NH3 and urea rows (4 each)
    nh3_rows = lca_rows[:4]
    urea_rows = lca_rows[4:]

    # ================================================================
    # Panel a — GWP waterfall across four LCA scenarios
    # ================================================================
    # Haber-Bosch reference lines
    HB_GWP_NG   = 1.8   # Natural gas, IEA 2021
    HB_GWP_AVG  = 2.1   # Global average
    HB_GWP_COAL = 3.9   # Coal-based

    n_scen = len(scenario_labels)
    x = np.arange(n_scen)
    width = 0.35

    def _net_gwp(r: ScenarioEvaluation) -> float:
        return r.lca.metrics["primary_product_gwp_kgco2e_per_kg"]

    nh3_gwp = np.array([_net_gwp(r) for r in nh3_rows])
    urea_gwp = np.array([_net_gwp(r) for r in urea_rows])

    bars_nh3 = ax_wf.bar(x - width / 2, nh3_gwp, width, label="NH\u2083 + SCP",
                          color=CAT_COLOR[ScenarioCategory.AMMONIA_SCP],
                          edgecolor="white", alpha=0.88, zorder=3)
    bars_urea = ax_wf.bar(x + width / 2, urea_gwp, width, label="Urea + SCP",
                           color=CAT_COLOR[ScenarioCategory.BIO_UREA_SCP],
                           edgecolor="white", alpha=0.88, zorder=3)

    for bars, vals in [(bars_nh3, nh3_gwp), (bars_urea, urea_gwp)]:
        for bar, v in zip(bars, vals):
            ha_x = bar.get_x() + bar.get_width() / 2
            va = "bottom" if v >= 0 else "top"
            y_off = 0.12 if v >= 0 else -0.12
            ax_wf.text(ha_x, v + y_off, f"{v:.1f}", ha="center", va=va,
                       fontsize=7.5, fontweight="700")

    # Haber-Bosch reference bands
    ax_wf.axhline(HB_GWP_NG,   color="#E15759", lw=1.4, ls="--", zorder=4)
    ax_wf.axhline(HB_GWP_COAL, color="#E15759", lw=0.8, ls=":",  zorder=4)
    ax_wf.text(n_scen - 0.6, HB_GWP_NG + 0.08,
               f"H-B (nat. gas) {HB_GWP_NG} kg",
               fontsize=7, color="#E15759", fontweight="600", va="bottom")
    ax_wf.text(n_scen - 0.6, HB_GWP_COAL + 0.08,
               f"H-B (coal) {HB_GWP_COAL} kg",
               fontsize=7, color="#E15759", va="bottom")
    ax_wf.axhline(0, color="#555555", lw=0.7)

    ax_wf.set_xticks(x)
    ax_wf.set_xticklabels(scenario_labels, fontsize=8)
    ax_wf.set_ylabel("Net GWP (kg CO\u2082e / kg primary product)")
    ax_wf.set_title("a)  Net GWP by Accounting Scenario", pad=12)
    ax_wf.legend(fontsize=8, loc="upper right")

    # ================================================================
    # Panel b — Cost vs GWP scatter under favorable LCA (renewable + biogenic)
    # All curated NH3 and urea recovery methods are shown regardless of GWP
    # sign so the cost-vs-climate trade-off is visible (e.g. struvite has the
    # lowest LCOX but a positive GWP; the membrane route has the lowest GWP).
    # ================================================================
    attractive_rows = run_best_methods_grid(
        capacities=[100.0, 1_000.0, 10_000.0],
        overrides=overrides,
        scenario_updates=FAVORABLE_LCA_SCENARIO_UPDATES,
    )

    def _method_key(row: ScenarioEvaluation) -> str:
        if row.foreground.scenario.category == ScenarioCategory.AMMONIA_SCP:
            return row.foreground.scenario.ammonia_recovery_method.value
        return row.foreground.scenario.urea_recovery_method.value

    method_groups: Dict[str, List[ScenarioEvaluation]] = {}
    for row in attractive_rows:
        method_groups.setdefault(_method_key(row), []).append(row)
    best_rows: List[ScenarioEvaluation] = []
    for rows in method_groups.values():
        rows.sort(key=lambda row: row.foreground.scenario.annual_primary_product_tpy)
        best_rows.append(
            min(
                rows,
                key=lambda row: row.tea.metrics["net_primary_lcox_usd_per_kg"],
            )
        )

    # ── Plot scatter points (one per curated method, only the largest capacity
    #    that achieves net-negative GWP, so labels don't collide) ────────────
    plotted_xy: List[tuple[float, float, str, dict]] = []
    seen_method_keys: set[str] = set()
    for row in best_rows:
        method_key = _method_key(row)
        sty = BEST_METHOD_STYLE.get(method_key, {"color": "#777777", "marker": "o", "ls": "-", "label": method_key})
        gwp_x = row.lca.metrics["primary_product_gwp_kgco2e_per_kg"]
        lcox_y = row.tea.metrics["net_primary_lcox_usd_per_kg"]
        # Log-scaled marker; small (110) at 100 t/y, medium (200) at 1 000, large (290) at 10 000
        size = 110.0 + 90.0 * math.log10(max(1.0, row.foreground.scenario.annual_primary_product_tpy) / 100.0)
        legend_label = sty["label"] if method_key not in seen_method_keys else "_nolegend_"
        seen_method_keys.add(method_key)
        ax_sc.scatter(
            [gwp_x],
            [lcox_y],
            s=[size],
            color=sty["color"],
            marker=sty["marker"],
            edgecolors="white",
            linewidths=1.0,
            alpha=0.92,
            zorder=3,
            label=legend_label,
        )
        plotted_xy.append((gwp_x, lcox_y, str(int(row.foreground.scenario.annual_primary_product_tpy)), sty))

    # ── Anti-collision label placement ─────────────────────────────────────
    # Group points that are within a small bounding box and stagger their
    # leader-line callouts vertically so capacity labels never overlap.
    def _is_close(a, b, dx=0.6, dy=0.18) -> bool:
        return abs(a[0] - b[0]) < dx and abs(a[1] - b[1]) < dy

    clusters: List[List[int]] = []
    for i, p in enumerate(plotted_xy):
        placed = False
        for cl in clusters:
            if _is_close(p, plotted_xy[cl[0]]):
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    LABEL_OFFSETS_Y = [22, -22, 36, -36, 54, -54]
    for cl in clusters:
        for rank, idx in enumerate(cl):
            gwp_x, lcox_y, cap_str, sty = plotted_xy[idx]
            dy = LABEL_OFFSETS_Y[rank % len(LABEL_OFFSETS_Y)]
            dx = 12 if dy >= 0 else -12
            ha = "left" if dx >= 0 else "right"
            ax_sc.annotate(
                f"{int(cap_str):,} t/y",
                (gwp_x, lcox_y),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=10,
                color=sty["color"],
                fontweight="600",
                ha=ha,
                arrowprops=dict(arrowstyle="-", color=sty["color"],
                                 lw=0.7, alpha=0.6, shrinkA=2, shrinkB=4),
            )

    # ── Haber-Bosch reference (placed inside axes with breathing room) ─────
    ax_sc.scatter(HB_GWP_NG, 0.60, s=190, color="#E15759", marker="X",
                  edgecolors="white", linewidths=1.0, zorder=5)
    ax_sc.annotate("Haber-Bosch\n(nat. gas)",
                   (HB_GWP_NG, 0.60), textcoords="offset points",
                   xytext=(-22, 30), fontsize=10.5, color="#E15759", fontweight="700",
                   ha="right",
                   arrowprops=dict(arrowstyle="-", color="#E15759", lw=0.7, alpha=0.8))

    handles, labels = ax_sc.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="X", color="w",
                          markerfacecolor="#E15759", markersize=11,
                          label="Haber-Bosch (nat. gas)"))
    labels.append("Haber-Bosch (nat. gas)")
    ax_sc.legend(handles, labels, fontsize=10, loc="lower right",
                  framealpha=0.95, edgecolor="#CCCCCC")

    if best_rows:
        gwp_vals = [row.lca.metrics["primary_product_gwp_kgco2e_per_kg"] for row in best_rows]
        gwp_min = min(gwp_vals)
        gwp_max = max(gwp_vals + [HB_GWP_NG])
        # Extra right-side room (HB label + legend) and small left margin.
        ax_sc.set_xlim(gwp_min - 0.8, gwp_max + 1.8)
        # Vertical headroom so leader lines don't clip.
        y_min = min(row.tea.metrics["net_primary_lcox_usd_per_kg"] for row in best_rows)
        y_max = max(row.tea.metrics["net_primary_lcox_usd_per_kg"] for row in best_rows)
        y_span = max(0.4, y_max - y_min)
        ax_sc.set_ylim(y_min - 0.35 * y_span, max(y_max + 0.45 * y_span, 0.95))
    ax_sc.axvline(0.0, color="#777777", lw=0.7, ls=":")
    # Subtle GWP=0 marker at the bottom of the axes.
    ax_sc.text(
        0.0, ax_sc.get_ylim()[0],
        " GWP = 0",
        fontsize=8.5, color="#777777", ha="left", va="bottom",
    )
    ax_sc.set_xlabel("Net GWP (kg CO\u2082e / kg product) — favorable LCA case")
    ax_sc.set_ylabel("Net LCOX (USD / kg)")
    ax_sc.set_title("b)  Cost vs Climate Intensity", pad=12)
    # Caption below the axes (NOT inside it) so it never overlaps data.
    # Plain (non-italic) Helvetica/Arial because italic fallbacks drop subscripts.
    ax_sc.text(
        0.5,
        -0.18,
        "All curated recovery methods shown (lowest-LCOX capacity per method, labelled in t/y).\n"
        "Assumptions: renewable electricity, biogenic CO$_2$, SCP C credit, protein displacement.",
        transform=ax_sc.transAxes,
        fontsize=10,
        color="#666666",
        ha="center",
        va="top",
    )

    # ================================================================
    # Panel c — GWP contribution stacked bar (design-basis: grid, biogenic CO2)
    # One bar per category, showing which inputs matter
    # ================================================================
    CONTRIB_ORDER = [
        ("electricity",            "Electricity",              "#4E79A7"),
        ("naoh",                   "NaOH",                     "#A0CBE8"),
        ("h3po4",                  "H\u2083PO\u2084",          "#F28E2B"),
        ("mgcl2",                  "MgCl\u2082",               "#FFBE7D"),
        ("co2_supply",             "CO\u2082 supply",          "#59A14F"),
        ("water",                  "Water/WW",                 "#B6992D"),
        ("wastewater",             "Wastewater",               "#B6992D"),
        ("steam",                  "Steam",                    "#D37295"),
        ("membrane_replacement",   "Membrane repl.",           "#9D7660"),
        ("biogenic_carbon_credit", "Biogenic C credit",        "#2E8B57"),
        ("scp_displacement_credit","Protein displacement",     "#1B6535"),
    ]

    # Design-basis contribution analysis: renewable + biogenic CO2 + biogenic-C
    # credit (index 2 of run_lca_sensitivity_grid). The grid case still appears
    # in panel (a) as the downside comparator.
    stk_nh3  = nh3_rows[2]
    stk_urea = urea_rows[2]
    stk_cats = [stk_nh3, stk_urea]
    stk_labels_ax = ["NH\u2083 + SCP", "Urea + SCP"]
    stk_x = np.arange(len(stk_cats))

    for ci, (key, label, color) in enumerate(CONTRIB_ORDER):
        vals = np.array([
            r.lca.contributions_kgco2e_per_y.get(key, 0.0)
            / max(1e-9, r.foreground.sellable_primary_product_kg)
            for r in stk_cats
        ])
        # Separate positive (burden) and negative (credit) for correct stacking
        pos_vals = np.where(vals > 0, vals, 0.0)
        neg_vals = np.where(vals < 0, vals, 0.0)
        if ci == 0:
            pos_bottoms = np.zeros(len(stk_cats))
            neg_bottoms = np.zeros(len(stk_cats))
        # Track running bottoms
        if ci == 0:
            _pos_b = [0.0] * len(stk_cats)
            _neg_b = [0.0] * len(stk_cats)
        if any(abs(v) > 1e-6 for v in vals):
            ax_stk.bar(stk_x, pos_vals, bottom=_pos_b, color=color,
                       edgecolor="white", linewidth=0.5, width=0.52, label=label)
            ax_stk.bar(stk_x, neg_vals, bottom=_neg_b, color=color,
                       edgecolor="white", linewidth=0.5, width=0.52)
            _pos_b = [_pos_b[i] + pos_vals[i] for i in range(len(stk_cats))]
            _neg_b = [_neg_b[i] + neg_vals[i] for i in range(len(stk_cats))]

    # Net line
    for i, r in enumerate(stk_cats):
        net = r.lca.metrics["primary_product_gwp_kgco2e_per_kg"]
        ax_stk.scatter(i, net, s=60, color="#333333", marker="D", zorder=5)
        ax_stk.text(i + 0.28, net, f"Net: {net:.1f}", fontsize=8, va="center", fontweight="700")

    ax_stk.axhline(HB_GWP_NG, color="#E15759", lw=1.3, ls="--", zorder=4,
                   label=f"H-B nat. gas ({HB_GWP_NG} kg)")
    ax_stk.axhline(0, color="#555555", lw=0.7)
    ax_stk.set_xticks(stk_x)
    ax_stk.set_xticklabels(stk_labels_ax, fontsize=10)
    ax_stk.set_ylabel("GWP contribution (kg CO\u2082e / kg product)")
    ax_stk.set_title("c)  Contribution Analysis\n(renewable power, biogenic CO\u2082)", pad=12)
    ax_stk.legend(fontsize=10, loc="upper right", ncol=1, framealpha=0.95)

    fig.suptitle(
        "Life Cycle Assessment \u2014 GWP by Accounting Scenario and Contribution\n"
        "Design basis: renewable electricity (PPA wind/solar, IPCC AR5 lifecycle), "
        "biogenic CO\u2082 from corn-ethanol off-gas. "
        "US grid kept as a downside comparator in panel (a).",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    _stamp(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 6 — SENSITIVITY TORNADO  (dual-color)
# ═══════════════════════════════════════════════════════════════════════════

def _sensitivity_rows(config: ScenarioConfig,
                      delta: float) -> List[Tuple[str, float, float, float]]:
    out: List[Tuple[str, float, float, float]] = []
    for label, param in SENSITIVITY_PARAMS[config.category]:
        try:
            low, base, high = run_sensitivity_cases(config, param, delta=delta)
        except KeyError:
            continue
        out.append((label,
                    low.tea.metrics["net_primary_lcox_usd_per_kg"],
                    base.tea.metrics["net_primary_lcox_usd_per_kg"],
                    high.tea.metrics["net_primary_lcox_usd_per_kg"]))
    out.sort(key=lambda t: abs(t[3] - t[1]), reverse=True)
    return out


def plot_sensitivity_tornado(config: ScenarioConfig,
                             delta: float = 0.20) -> plt.Figure:
    _style()
    rows = _sensitivity_rows(config, delta)
    base = rows[0][2]

    fig, ax = plt.subplots(figsize=(11.5, max(4.0, 0.7 * len(rows) + 1.8)))
    y = np.arange(len(rows))

    for i, (label, lo, _, hi) in enumerate(rows):
        lo_bar = min(lo, hi) - base
        hi_bar = max(lo, hi) - base
        ax.barh(i, lo_bar, left=base, height=0.55,
                color="#E15759", edgecolor="white", linewidth=0.4, zorder=3)
        ax.barh(i, hi_bar, left=base, height=0.55,
                color="#4E79A7", edgecolor="white", linewidth=0.4, zorder=3)

        ax.text(min(lo, hi) - 0.015, i, _usd(min(lo, hi)),
                ha="right", va="center", fontsize=7.5)
        ax.text(max(lo, hi) + 0.015, i, _usd(max(lo, hi)),
                ha="left", va="center", fontsize=7.5)

    ax.axvline(base, color="#333333", lw=1.3, zorder=4)
    ax.text(base, -0.8, f"Base case: {_usd(base)}/kg",
            ha="center", fontsize=9.5, fontweight="700")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Net levelized cost (USD / kg primary product)")
    ax.set_title(
        f"Sensitivity (\u00b1{int(delta * 100)}%) \u2014 "
        f"{_scenario_label_from_config(config)} at {int(config.annual_primary_product_tpy):,} t/y",
        pad=14)

    legend_elements = [
        Patch(facecolor="#E15759", label=f"\u2212{int(delta * 100)}% parameter"),
        Patch(facecolor="#4E79A7", label=f"+{int(delta * 100)}% parameter"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    _stamp(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 7 — FOUR-PANEL SUMMARY  (executive overview)
# ═══════════════════════════════════════════════════════════════════════════

def plot_executive_summary(evaluations: Iterable[ScenarioEvaluation]) -> plt.Figure:
    _style()
    rows = _sorted(evaluations)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    ax_lcox = axes[0, 0]
    for cat in [ScenarioCategory.AMMONIA_SCP, ScenarioCategory.BIO_UREA_SCP]:
        sub = _by_cat(rows, cat)
        caps = [r.foreground.scenario.annual_primary_product_tpy for r in sub]
        net = [r.tea.metrics["net_primary_lcox_usd_per_kg"] for r in sub]
        ax_lcox.plot(caps, net, marker="o", markersize=7, linewidth=2.2,
                     color=CAT_COLOR[cat], label=_scenario_label(sub[0]) if sub else CAT_LABEL[cat])
    ax_lcox.set_xscale("log")
    if mticker is not None:
        ax_lcox.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"))
    ax_lcox.set_xlabel("Capacity (t/y)")
    ax_lcox.set_ylabel("Net LCOX (USD/kg)")
    ax_lcox.set_title("a)  Levelized Cost by Scale", pad=10)
    ax_lcox.legend(fontsize=8)

    ax_gwp = axes[0, 1]
    # Baseline GWP from the comparison grid (user's current LCA settings)
    for cat in [ScenarioCategory.AMMONIA_SCP, ScenarioCategory.BIO_UREA_SCP]:
        sub = _by_cat(rows, cat)
        caps = [r.foreground.scenario.annual_primary_product_tpy for r in sub]
        gwp = [r.lca.metrics["primary_product_gwp_kgco2e_per_kg"] for r in sub]
        ax_gwp.plot(
            caps,
            gwp,
            marker="s",
            markersize=7,
            linewidth=2.2,
            color=CAT_COLOR[cat],
            label=f"{(_scenario_label(sub[0]) if sub else CAT_LABEL[cat])} (current settings)",
        )
    # Favorable-LCA overlay: renewable power + biogenic CO2 + biogenic-C credit
    try:
        favorable_rows = run_best_methods_negative_gwp_grid(
            capacities=[100.0, 1_000.0, 10_000.0],
        )
        for cat in [ScenarioCategory.AMMONIA_SCP, ScenarioCategory.BIO_UREA_SCP]:
            fav_sub = [r for r in favorable_rows if r.foreground.scenario.category == cat]
            fav_sub.sort(key=lambda r: r.foreground.scenario.annual_primary_product_tpy)
            if fav_sub:
                fcaps = [r.foreground.scenario.annual_primary_product_tpy for r in fav_sub]
                fgwp = [r.lca.metrics["primary_product_gwp_kgco2e_per_kg"] for r in fav_sub]
                ax_gwp.plot(
                    fcaps,
                    fgwp,
                    marker="^",
                    markersize=6,
                    linewidth=1.6,
                    linestyle="--",
                    color=CAT_COLOR[cat],
                    alpha=0.6,
                    label=f"{(_scenario_label(fav_sub[0]) if fav_sub else CAT_LABEL[cat])} (renew. + biogenic)",
                )
    except Exception:
        pass  # don't break the figure if favorable grid fails
    _HB_NG = 1.8  # kg CO2e/kg, IEA 2021 Haber-Bosch nat. gas
    ax_gwp.axhline(_HB_NG, ls="--", lw=0.9, color="#E15759")
    ax_gwp.axhline(0, ls="-", lw=0.5, color="#555555")
    ax_gwp.text(110, _HB_NG + 0.15, f"H\u2013B nat. gas ({_HB_NG} kg CO\u2082e/kg)",
                fontsize=7, color="#E15759")
    ax_gwp.set_xscale("log")
    if mticker is not None:
        ax_gwp.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"))
    ax_gwp.set_xlabel("Capacity (t/y)")
    ax_gwp.set_ylabel("GWP (kg CO\u2082e/kg)")
    ax_gwp.set_title("b)  Carbon Intensity by Scale", pad=10)
    ax_gwp.legend(fontsize=8)

    ax_cap = axes[1, 0]
    for cat in [ScenarioCategory.AMMONIA_SCP, ScenarioCategory.BIO_UREA_SCP]:
        sub = _by_cat(rows, cat)
        caps = [r.foreground.scenario.annual_primary_product_tpy for r in sub]
        total_cap = [r.tea.metrics["total_capital_usd"] / 1e6 for r in sub]
        ax_cap.plot(caps, total_cap, marker="D", markersize=6, linewidth=2.2,
                    color=CAT_COLOR[cat], label=_scenario_label(sub[0]) if sub else CAT_LABEL[cat])
    ax_cap.set_xscale("log")
    ax_cap.set_yscale("log")
    if mticker is not None:
        ax_cap.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"))
        ax_cap.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x:,.0f}M"))
    ax_cap.set_xlabel("Capacity (t/y)")
    ax_cap.set_ylabel("Total capital (M USD)")
    ax_cap.set_title("c)  Capital Investment", pad=10)
    ax_cap.legend(fontsize=8)

    ax_npv = axes[1, 1]
    for cat in [ScenarioCategory.AMMONIA_SCP, ScenarioCategory.BIO_UREA_SCP]:
        sub = _by_cat(rows, cat)
        caps = [r.foreground.scenario.annual_primary_product_tpy for r in sub]
        npv = [r.tea.metrics["npv_usd"] / 1e6 for r in sub]
        ax_npv.plot(caps, npv, marker="^", markersize=7, linewidth=2.2,
                    color=CAT_COLOR[cat], label=_scenario_label(sub[0]) if sub else CAT_LABEL[cat])
    ax_npv.axhline(0, color="#555555", lw=0.7)
    ax_npv.set_xscale("log")
    if mticker is not None:
        ax_npv.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x):,}"))
    ax_npv.set_xlabel("Capacity (t/y)")
    ax_npv.set_ylabel("NPV (M USD)")
    ax_npv.set_title("d)  Net Present Value (20 y, 10%)", pad=10)
    ax_npv.legend(fontsize=8)

    fig.suptitle("Executive Summary \u2014 Formate Biorefinery TEA / LCA",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    _stamp(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  BACKWARD COMPAT
# ═══════════════════════════════════════════════════════════════════════════

def _recovery_method_label(row: ScenarioEvaluation) -> str:
    """Short display label for the recovery method used in a given evaluation."""
    cat = row.foreground.scenario.category
    if cat == ScenarioCategory.AMMONIA_SCP:
        key = row.foreground.scenario.ammonia_recovery_method.value
    else:
        key = row.foreground.scenario.urea_recovery_method.value
    return RECOVERY_METHOD_LABELS.get(key, key)


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 9 — NH3 RECOVERY METHOD COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def plot_nh3_recovery_comparison(
    capacity_tpy: float = 1_000.0,
    overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """Compare all NH3 recovery methods at a fixed capacity.

    Panel layout:
    - Left: LCOX comparison bars with market price benchmarks
    - Right: Energy + reagent cost breakdown per method
    """
    _style()
    rows = run_recovery_comparison(ScenarioCategory.AMMONIA_SCP, capacity_tpy, overrides=overrides)

    # --- colors for each method
    METHOD_COLORS = {
        AmmoniaRecoveryMethod.STRUVITE_MAP: "#2E8B57",
        AmmoniaRecoveryMethod.MEMBRANE: "#4E79A7",
        AmmoniaRecoveryMethod.VACUUM_STRIPPING: "#F28E2B",
        AmmoniaRecoveryMethod.AIR_STRIPPING: "#E15759",
    }
    method_colors = [METHOD_COLORS.get(r.foreground.scenario.ammonia_recovery_method, "#AAAAAA")
                     for r in rows]
    labels = [_recovery_method_label(r) for r in rows]
    y = np.arange(len(rows))

    fig, (ax_lcox, ax_cost) = plt.subplots(1, 2, figsize=(16, 6.5))

    # ---- Left: LCOX bars ----
    gross = np.array([r.tea.metrics["gross_primary_lcox_usd_per_kg"] for r in rows])
    net = np.array([r.tea.metrics["net_primary_lcox_usd_per_kg"] for r in rows])
    mkt = np.array([_market(r) for r in rows])

    w = 0.32
    ax_lcox.barh(y - w / 2, gross, w, color="#DDDDDD", edgecolor="white",
                 linewidth=0.5, label="Gross LCOX (before SCP credit)", zorder=3)
    ax_lcox.barh(y + w / 2, net, w, color=method_colors, edgecolor="white",
                 linewidth=0.5, label="Net LCOX (after SCP credit)", zorder=4)

    for i, (g, n, m) in enumerate(zip(gross, net, mkt)):
        ax_lcox.text(g + 0.06, i - w / 2, _usd(g, 1), va="center", fontsize=7.5, color="#666666")
        ax_lcox.text(n + 0.06, i + w / 2, _usd(n, 1), va="center", fontsize=8,
                     fontweight="700", color=method_colors[i])
        ax_lcox.scatter(m, i, marker="|", s=120, color="#E15759", linewidths=2.0, zorder=5)

    ax_lcox.axvline(0, color="#555555", lw=0.6)

    # NH3 market price reference line
    nh3_price = rows[0].tea.metrics["benchmark_primary_revenue_usd_per_y"] / max(
        1e-9, rows[0].foreground.sellable_primary_product_kg)
    # We want the standard ammonia price, not struvite-adjusted; read first non-MAP row
    for r in rows:
        if r.foreground.scenario.ammonia_recovery_method != AmmoniaRecoveryMethod.STRUVITE_MAP:
            nh3_ref = _market(r)
            break
    else:
        nh3_ref = 0.60

    ax_lcox.axvline(nh3_ref, color="#E15759", ls="--", lw=1.0, zorder=2)
    ax_lcox.text(nh3_ref + 0.05, len(rows) - 0.2, f"NH\u2083 market\n{_usd(nh3_ref)}/kg",
                 fontsize=7.5, color="#E15759", va="top")

    ax_lcox.set_yticks(y)
    ax_lcox.set_yticklabels(labels, fontsize=9.5)
    ax_lcox.set_xlabel("USD / kg NH\u2083 equivalent")
    ax_lcox.set_title(f"a)  LCOX by Recovery Method  ({int(capacity_tpy):,} t/y)", pad=12)
    ax_lcox.invert_yaxis()
    ax_lcox.legend(fontsize=7.5, loc="lower right")

    # ---- Right: cost driver breakdown ----
    # Show electricity, reagent (NaOH or MgCl2+H3PO4), membrane for each method
    kg_ref = np.array([r.foreground.sellable_primary_product_kg for r in rows])
    elec_cost = np.array([
        r.tea.variable_opex_usd_per_y.get("electricity", 0.0) / max(1e-9, kg_ref[i])
        for i, r in enumerate(rows)])
    naoh_cost = np.array([
        r.tea.variable_opex_usd_per_y.get("naoh", 0.0) / max(1e-9, kg_ref[i])
        for i, r in enumerate(rows)])
    mgcl2_cost = np.array([
        r.tea.variable_opex_usd_per_y.get("mgcl2", 0.0) / max(1e-9, kg_ref[i])
        for i, r in enumerate(rows)])
    h3po4_cost = np.array([
        r.tea.variable_opex_usd_per_y.get("h3po4", 0.0) / max(1e-9, kg_ref[i])
        for i, r in enumerate(rows)])
    membrane_cost = np.array([
        r.tea.variable_opex_usd_per_y.get("membrane_replacement", 0.0) / max(1e-9, kg_ref[i])
        for i, r in enumerate(rows)])
    capex_cost = np.array([
        r.tea.fixed_opex_usd_per_y.get("annualized_capex", 0.0) / max(1e-9, kg_ref[i])
        for i, r in enumerate(rows)])

    drivers = [
        ("Electricity",   elec_cost,     "#4E79A7"),
        ("NaOH (pH adj.)", naoh_cost,    "#F28E2B"),
        ("MgCl\u2082 (MAP)", mgcl2_cost, "#59A14F"),
        ("H\u2083PO\u2084 (MAP)", h3po4_cost, "#8CD17D"),
        ("Membrane repl.", membrane_cost, "#A0CBE8"),
        ("Annualized CapEx", capex_cost, "#79706E"),
    ]

    left = np.zeros(len(rows))
    for label, vals, color in drivers:
        if not np.any(vals > 1e-6):
            continue
        ax_cost.barh(y, vals, left=left, height=0.55, color=color, edgecolor="white",
                     linewidth=0.4, label=label, zorder=3)
        for i, v in enumerate(vals):
            if v > 0.12:
                ax_cost.text(left[i] + v / 2, i, _usd(v, 2), ha="center", va="center",
                             fontsize=7, color="white", fontweight="600")
        left += vals

    ax_cost.set_yticks(y)
    ax_cost.set_yticklabels(labels, fontsize=9.5)
    ax_cost.set_xlabel("Recovery OPEX (USD / kg NH\u2083 eq.)")
    ax_cost.set_title("b)  Recovery Cost Drivers", pad=12)
    ax_cost.invert_yaxis()
    ax_cost.legend(fontsize=7.5, loc="lower right")

    # Annotation table: energy intensity + product form
    table_data = []
    for r in rows:
        method = r.foreground.scenario.ammonia_recovery_method
        if method == AmmoniaRecoveryMethod.STRUVITE_MAP:
            product_form = "Struvite fertilizer"
            energy = "0.10 kWh/kg"
        elif method == AmmoniaRecoveryMethod.MEMBRANE:
            product_form = "Ammonium solution"
            energy = "0.30 kWh/kg"
        elif method == AmmoniaRecoveryMethod.VACUUM_STRIPPING:
            product_form = "Liquid NH\u2083 / AS"
            energy = "0.50 kWh/kg"
        else:
            product_form = "Liquid NH\u2083 / AS"
            energy = "0.80 kWh/kg"
        table_data.append(f"{energy} \u2502 {product_form}")

    for i, note in enumerate(table_data):
        ax_cost.text(ax_cost.get_xlim()[1] * 0.98, i, note,
                     ha="right", va="center", fontsize=6.5, color="#555555",
                     style="italic")

    fig.suptitle(f"NH\u2083 Recovery Method Comparison \u2014 {int(capacity_tpy):,} t/y",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.subplots_adjust(wspace=0.35)
    _stamp(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  FIG 10 — UREA RECOVERY METHOD COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def plot_urea_recovery_comparison(
    capacity_tpy: float = 1_000.0,
    overrides: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """Compare urea recovery methods (single-effect evap vs MVR) at a fixed capacity."""
    _style()
    rows = run_recovery_comparison(ScenarioCategory.BIO_UREA_SCP, capacity_tpy, overrides=overrides)

    METHOD_COLORS_U = {
        UreaRecoveryMethod.MVR_CRYSTALLIZATION: "#009E73",
        UreaRecoveryMethod.EVAPORATION:         "#F28E2B",
        UreaRecoveryMethod.HYBRID:              "#76B7B2",
    }
    method_colors = [METHOD_COLORS_U.get(r.foreground.scenario.urea_recovery_method, "#AAAAAA")
                     for r in rows]
    labels = [_recovery_method_label(r) for r in rows]
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0))
    ax_lcox, ax_steam, ax_gwp = axes

    # ---- Left: LCOX bars ----
    gross = np.array([r.tea.metrics["gross_primary_lcox_usd_per_kg"] for r in rows])
    net   = np.array([r.tea.metrics["net_primary_lcox_usd_per_kg"] for r in rows])
    mkt_ref = 0.40  # standard urea market price

    w = 0.30
    ax_lcox.barh(y - w / 2, gross, w, color="#DDDDDD", edgecolor="white", linewidth=0.5,
                 label="Gross LCOX", zorder=3)
    ax_lcox.barh(y + w / 2, net, w, color=method_colors, edgecolor="white", linewidth=0.5,
                 label="Net LCOX (after SCP credit)", zorder=4)

    for i, (g, n) in enumerate(zip(gross, net)):
        ax_lcox.text(g + 0.04, i - w / 2, _usd(g, 2), va="center", fontsize=8, color="#666666")
        ax_lcox.text(n + 0.04, i + w / 2, _usd(n, 2), va="center", fontsize=8.5,
                     fontweight="700", color=method_colors[i])

    ax_lcox.axvline(mkt_ref, color="#E15759", ls="--", lw=1.0, zorder=2)
    ax_lcox.text(mkt_ref + 0.02, len(rows) - 0.15, f"Urea market\n{_usd(mkt_ref)}/kg",
                 fontsize=7.5, color="#E15759", va="top")
    ax_lcox.axvline(0, color="#555555", lw=0.5)
    ax_lcox.set_yticks(y)
    ax_lcox.set_yticklabels(labels, fontsize=10)
    ax_lcox.invert_yaxis()
    ax_lcox.set_xlabel("USD / kg urea")
    ax_lcox.set_title(f"a)  LCOX Comparison  ({int(capacity_tpy):,} t/y)", pad=12)
    ax_lcox.legend(fontsize=7.5, loc="lower right")

    # ---- Center: Steam vs electricity trade-off ----
    kg_ref = np.array([r.foreground.sellable_primary_product_kg for r in rows])
    steam_cost = np.array([
        r.tea.variable_opex_usd_per_y.get("steam", 0.0) / max(1e-9, kg_ref[i])
        for i, r in enumerate(rows)])
    elec_cost_rec = np.array([
        # Isolate just the recovery electricity: total minus fermentation/scp/electrolysis
        (r.tea.variable_opex_usd_per_y.get("electricity", 0.0)
         - r.foreground.ledger.electricity_kwh_per_y.get("electrolyzer", 0.0)
           * 0.082  # electricity price proxy
         - r.foreground.ledger.electricity_kwh_per_y.get("agitation_aeration", 0.0)
           * 0.082
         - r.foreground.ledger.electricity_kwh_per_y.get("scp_harvest", 0.0)
           * 0.082
         - r.foreground.ledger.electricity_kwh_per_y.get("scp_drying", 0.0)
           * 0.082) / max(1e-9, kg_ref[i])
        for i, r in enumerate(rows)])

    xpos = np.arange(len(rows))
    bw = 0.32
    ax_steam.bar(xpos - bw / 2, steam_cost, bw, color="#A0CBE8",
                 edgecolor="white", linewidth=0.5, label="Steam OPEX", zorder=3)
    ax_steam.bar(xpos + bw / 2, np.clip(elec_cost_rec, 0, None), bw, color="#4E79A7",
                 edgecolor="white", linewidth=0.5, label="Recovery electricity OPEX", zorder=3)
    ax_steam.set_xticks(xpos)
    ax_steam.set_xticklabels([r.split("\n")[0] for r in labels], fontsize=9)
    ax_steam.set_ylabel("USD / kg urea")
    ax_steam.set_title("b)  Steam vs Electricity Trade-off", pad=12)
    ax_steam.legend(fontsize=8)
    ymax_s = max(max(steam_cost), max(np.clip(elec_cost_rec, 0, None))) * 1.35
    ax_steam.set_ylim(0, max(ymax_s, 1e-4))

    for i, (s, e) in enumerate(zip(steam_cost, np.clip(elec_cost_rec, 0, None))):
        ax_steam.text(i - bw / 2, s + 0.01, _usd(s, 3), ha="center", fontsize=7.5)
        ax_steam.text(i + bw / 2, e + 0.01, _usd(e, 3), ha="center", fontsize=7.5)

    # ---- Right: GWP comparison ----
    gwp = np.array([r.lca.metrics["primary_product_gwp_kgco2e_per_kg"] for r in rows])
    bars = ax_gwp.barh(y, gwp, height=0.55, color=method_colors, edgecolor="white",
                       linewidth=0.5, zorder=3)
    for bar, val in zip(bars, gwp):
        ax_gwp.text(val + 0.15, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=9, fontweight="600")
    ax_gwp.set_yticks(y)
    ax_gwp.set_yticklabels(labels, fontsize=10)
    ax_gwp.invert_yaxis()
    ax_gwp.set_xlabel("kg CO\u2082e / kg urea")
    ax_gwp.set_title("c)  GWP by Recovery Route", pad=12)

    fig.suptitle(f"Urea Recovery Method Comparison \u2014 {int(capacity_tpy):,} t/y",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    _stamp(fig)
    return fig


def plot_cost_and_gwp(evaluations: Iterable[ScenarioEvaluation]) -> plt.Figure:
    return plot_cost_vs_gwp(evaluations)


def plot_sensitivity(evaluations: Sequence[ScenarioEvaluation],
                     parameter_name: str) -> plt.Figure:
    _style()
    labels = ["\u221220%", "Base", "+20%"]
    lcox = [r.tea.metrics["net_primary_lcox_usd_per_kg"] for r in evaluations]
    gwp = [r.lca.metrics["primary_product_gwp_kgco2e_per_kg"] for r in evaluations]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(labels, lcox, marker="o", color="#E15759", linewidth=2, label="Net LCOX")
    ax2.plot(labels, gwp, marker="s", color="#76B7B2", linewidth=2, label="GWP")
    ax1.set_ylabel("Net LCOX (USD/kg)")
    ax2.set_ylabel("GWP (kg CO\u2082e/kg)")
    ax1.set_title(f"Sensitivity to {parameter_name}")
    fig.tight_layout()
    _stamp(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def create_all_figures(
    evaluations: Iterable[ScenarioEvaluation],
    sensitivity_config: Optional[ScenarioConfig] = None,
    sensitivity_delta: float = 0.20,
    overrides: Optional[Dict[str, float]] = None,
    save_dir: Optional[str | Path] = None,
    show: bool = False,
    verbose: bool = True,
) -> Dict[str, plt.Figure]:
    _style()
    rows = list(evaluations)
    sensitivity_config = sensitivity_config or ScenarioConfig(
        category=ScenarioCategory.BIO_UREA_SCP,
        annual_primary_product_tpy=1_000.0,
    )

    def _log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    cap = sensitivity_config.annual_primary_product_tpy
    specs = [
        ("00_process_flow",         lambda: plot_process_flow()),
        ("01_market_viability",     lambda: plot_market_viability_overview(rows)),
        ("02_cost_structure",       lambda: plot_cost_structure(rows, capacity_tpy=cap)),
        ("03_scale_margin",         lambda: plot_margin_curve(rows)),
        ("04_cashflow_npv",         lambda: plot_annual_cashflow(overrides=overrides)),  # runs its own best-method grid
        ("05_cost_vs_gwp",          lambda: plot_cost_vs_gwp(rows, overrides=overrides)),
        ("06_sensitivity_nh3",      lambda: plot_sensitivity_tornado(
            ScenarioConfig(category=ScenarioCategory.AMMONIA_SCP,
                           annual_primary_product_tpy=cap),
            delta=sensitivity_delta)),
        ("07_sensitivity_urea",     lambda: plot_sensitivity_tornado(
            sensitivity_config, delta=sensitivity_delta)),
        ("08_executive_summary",    lambda: plot_executive_summary(rows)),
        ("09_nh3_recovery_compare", lambda: plot_nh3_recovery_comparison(capacity_tpy=cap, overrides=overrides)),
        ("10_urea_recovery_compare", lambda: plot_urea_recovery_comparison(capacity_tpy=cap, overrides=overrides)),
    ]

    figures: Dict[str, plt.Figure] = {}
    for name, builder in specs:
        _log(f"  [fig] {name} ...")
        figures[name] = builder()

    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight",
                        facecolor="white")
            _log(f"  [save] {name}.png")

    if show:
        plt.show()

    _log(f"  [done] {len(figures)} figures generated")
    return figures


def save_figure(fig, path: str | Path) -> None:
    _require_mpl()
    fig.savefig(Path(path), dpi=300, bbox_inches="tight", facecolor="white")
