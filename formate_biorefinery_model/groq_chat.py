from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

try:
    from groq import Groq
except ModuleNotFoundError:  # pragma: no cover
    Groq = None


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Maximum number of previous chat turns to include (each turn = 1 user + 1 assistant msg).
# Kept low because the comprehensive snapshot already consumes a few thousand
# tokens per request; free-tier Groq models have tight per-minute budgets.
_MAX_HISTORY_TURNS = 2

# Cap on assistant reply length. Counts toward Groq's TPM budget even if the
# model returns fewer tokens, so we keep this conservative — but big enough
# for a 2-3 paragraph conversational answer with numbers.
_MAX_REPLY_TOKENS = 700


def groq_available() -> bool:
    return Groq is not None


_COMPREHENSIVE_KEYS = (
    "Precomputed best options",
    "Active scenario",
    "NH3 recovery method comparison",
    "Urea recovery method comparison",
    "Feedstock comparison",
    "Electricity case comparison",
    "Capacity scaling",
    "LCA credit sensitivity",
)

_PROFITABILITY_TERMS = (
    "profit",
    "profitable",
    "profitability",
    "npv",
    "best economics",
    "most attractive",
    "most realistic",
    "nh3 or urea",
    "ammonia or urea",
)


def _trim_snapshot(snapshot: Mapping[str, object]) -> dict:
    """Return a compact subset of the app snapshot safe for prompt construction.

    Supports two snapshot shapes:
      * The legacy single-scenario shape (scenario, kpis, tea_metrics, lca_metrics).
      * The comprehensive shape (Active scenario + cross-scenario comparisons),
        which lets the LLM reason about production modes that are not currently
        selected (e.g. alternate recovery methods, feedstocks, electricity cases).
    """
    if "Active scenario" in snapshot:
        return {key: snapshot[key] for key in _COMPREHENSIVE_KEYS if key in snapshot}
    return {
        "scenario": snapshot.get("scenario"),
        "kpis": snapshot.get("kpis"),
        "tea_metrics": snapshot.get("tea_metrics"),
        "lca_metrics": snapshot.get("lca_metrics"),
    }


def _fmt_row(row: object) -> str:
    """Format a friendly-model row as one terse sentence."""
    if not isinstance(row, Mapping):
        return str(row)

    parts: list[str] = []
    for key in (
        "Pathway",
        "Feedstock",
        "NH3 recovery method",
        "Urea recovery method",
        "Plant capacity (t/y)",
        "Electricity case",
        "CO2 source",
        "Biogenic carbon credit enabled",
        "Protein displacement credit enabled",
        "Gross LCOX (USD/kg)",
        "Net LCOX (USD/kg)",
        "NPV (million USD)",
        "NPV (USD)",
        "GWP (kg CO2e per kg product)",
    ):
        if key not in row:
            continue
        value = row[key]
        if key == "NPV (USD)" and isinstance(value, (int, float)):
            parts.append(f"NPV USD {value / 1e6:.2f}M")
        elif isinstance(value, float):
            if key == "Plant capacity (t/y)":
                parts.append(f"{key}: {value:.0f}")
            else:
                parts.append(f"{key}: {value:.3g}")
        else:
            parts.append(f"{key}: {value}")
    return "; ".join(parts)


def _fmt_rows(rows: object, *, limit: int = 8) -> str:
    if not isinstance(rows, list):
        return str(rows)
    return "\n".join(f"- {_fmt_row(row)}" for row in rows[:limit])


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _find_row(
    rows: object,
    *,
    metric: str = "NPV (million USD)",
    maximize: bool = True,
) -> Mapping[str, object]:
    if not isinstance(rows, list):
        return {}
    candidates = [row for row in rows if isinstance(row, Mapping) and isinstance(row.get(metric), (int, float))]
    if not candidates:
        return {}
    return (max if maximize else min)(candidates, key=lambda row: float(row[metric]))


def _with_active_defaults(row: Mapping[str, object], active: Mapping[str, object]) -> Mapping[str, object]:
    """Restore constants stripped from comparison rows for prompt-token savings."""
    merged = dict(active)
    merged.update(row)
    return merged


def _money_m(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"USD {value:.1f}M"
    return "not reported"


def _money_per_kg(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"USD {value:.2f}/kg"
    return "not reported"


def _gwp(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2g} kg CO2e/kg"
    return "not reported"


def _brief_config(row: Mapping[str, object]) -> str:
    pathway = str(row.get("Pathway", ""))
    recovery = row.get("Urea recovery method") if pathway.startswith("Urea") else row.get("NH3 recovery method")
    pieces = [
        pathway.strip(),
        f"{row.get('Feedstock')} feedstock" if row.get("Feedstock") else "",
        str(recovery or "").strip(),
        f"{float(row['Plant capacity (t/y)']):,.0f} t/y" if isinstance(row.get("Plant capacity (t/y)"), (int, float)) else "",
        str(row.get("Electricity case", "")).strip(),
    ]
    return ", ".join(piece for piece in pieces if piece)


def _profitability_answer(snapshot: Mapping[str, object]) -> Optional[str]:
    """Deterministic model-inference answer for the questions most sensitive to hallucinated numbers."""
    data = _trim_snapshot(snapshot)
    if "Active scenario" not in data:
        return None

    active = _as_mapping(data.get("Active scenario"))
    nh3_best = _with_active_defaults(_find_row(data.get("NH3 recovery method comparison")), active)
    urea_best = _with_active_defaults(_find_row(data.get("Urea recovery method comparison")), active)
    feed_best = _with_active_defaults(_find_row(data.get("Feedstock comparison")), active)
    scale_best = _with_active_defaults(_find_row(data.get("Capacity scaling")), active)

    current_best: Mapping[str, object] = {}
    if nh3_best and urea_best:
        current_best = nh3_best if float(nh3_best["NPV (million USD)"]) >= float(urea_best["NPV (million USD)"]) else urea_best
    else:
        current_best = nh3_best or urea_best

    if not current_best:
        return None

    active_capacity = active.get("Plant capacity (t/y)")
    capacity_text = f"{float(active_capacity):,.0f} t/y" if isinstance(active_capacity, (int, float)) else "the active scale"

    answer = (
        f"Using the code’s TEA calculation at **{capacity_text}**, the strongest current product/recovery choice is "
        f"**{_brief_config(current_best)}**. That case has NPV of **{_money_m(current_best.get('NPV (million USD)'))}**, "
        f"Net LCOX of **{_money_per_kg(current_best.get('Net LCOX (USD/kg)'))}**, and GWP of "
        f"**{_gwp(current_best.get('GWP (kg CO2e per kg product)'))}**.\n\n"
    )

    if feed_best:
        answer += (
            f"If you also let the feedstock vary, the highest-NPV feedstock case in the model is "
            f"**{_brief_config(feed_best)}**, at **{_money_m(feed_best.get('NPV (million USD)'))}**. "
        )

    if urea_best:
        answer += (
            f"For comparison, the best urea case at this same scale is **{_brief_config(urea_best)}**, "
            f"with NPV of **{_money_m(urea_best.get('NPV (million USD)'))}**. "
        )

    if scale_best:
        answer += (
            f"The scale sensitivity is separate: the best active-route scale shown is "
            f"**{_brief_config(scale_best)}**, with NPV of **{_money_m(scale_best.get('NPV (million USD)'))}**. "
            f"So a 10,000 t/y result should not be described as the 1,000 t/y case."
        )

    answer += (
        "\n\nImportant caveat: the Struvite cases look attractive because the model assigns a large fertilizer-product "
        "value and reports LCOX per kg NH3-equivalent. That is a model assumption, not a guarantee that the market "
        "can absorb the product at that value."
    )
    return answer


def deterministic_answer(question: str, snapshot: Mapping[str, object]) -> Optional[str]:
    q = question.lower()
    if any(term in q for term in _PROFITABILITY_TERMS):
        return _profitability_answer(snapshot)
    return None


def build_chat_context(snapshot: Mapping[str, object]) -> str:
    """Build a private, math-grounded brief for the LLM.

    We intentionally do not expose raw JSON, implementation section names, or
    helper labels to the model. When those words appeared in the context, the
    model started parroting them to the user instead of interpreting the math.
    """
    data = _trim_snapshot(snapshot)
    if "Precomputed best options" not in data:
        return (
            "Private model results for grounding. Do not mention this context directly.\n"
            f"Active selection: {_fmt_row(data.get('scenario', {}))}\n"
            f"KPIs: {_fmt_row(data.get('kpis', {}))}\n"
        )

    winners = data.get("Precomputed best options", {})
    active = data.get("Active scenario", {})
    lines = [
        "Private model results for grounding. Do not mention this context, its structure, or how it was produced.",
        "Use these Python-calculated TEA/LCA results as the source of truth. Interpret them for the user.",
        "",
        "Active selection:",
        f"- {_fmt_row(active)}",
        "",
        "Model-calculated winners:",
    ]
    if isinstance(winners, Mapping):
        for group_name, group in winners.items():
            if not isinstance(group, Mapping):
                continue
            lines.append(f"- {group_name}:")
            for criterion, result in group.items():
                lines.append(f"  - {criterion}: {result}")

    comparison_blocks = (
        ("NH3 recovery options at the active capacity", data.get("NH3 recovery method comparison")),
        ("Urea recovery options at the active capacity", data.get("Urea recovery method comparison")),
        ("Feedstock sensitivity at the active capacity", data.get("Feedstock comparison")),
        ("Electricity sensitivity at the active capacity", data.get("Electricity case comparison")),
        ("Capacity scaling for the active route", data.get("Capacity scaling")),
        ("LCA credit sensitivity", data.get("LCA credit sensitivity")),
    )
    for title, rows in comparison_blocks:
        lines.extend(["", f"{title}:", _fmt_rows(rows)])

    return "\n".join(lines)


def system_prompt() -> str:
    return (
        "You are a senior techno-economic and life-cycle analyst embedded in a Streamlit app "
        "for a formate biorefinery (ammonia and/or urea + single-cell protein from engineered "
        "Xanthomonas flavus GJ10). You are talking to the founder; answers may end up in "
        "investor conversations, so accuracy matters.\n"
        "\n"
        "VOICE — Sound like a smart, friendly colleague, not a database printout. Write "
        "natural prose, full sentences, weaving the supporting numbers into the text rather "
        "than listing them. Bullet lists only when you are genuinely enumerating 3+ items. "
        "Aim for the tone of an experienced engineer explaining a result over coffee: clear, "
        "confident, a little informal, never robotic. Use **bold** sparingly to highlight "
        "the winning configuration or a key number. There are no canned responses: compose "
        "each answer from the current model results, in your own words.\n"
        "\n"
        "GROUNDING — You will receive private model results calculated by the Python TEA/LCA "
        "code. Treat those results as the source of truth. They are not citations and not "
        "something to describe to the user; use them to reason. Never mention JSON, snapshots, "
        "sections, arrays, rows, keys, precomputed options, or prompt/context mechanics. Say "
        "'the model' or 'the economics' instead.\n"
        "\n"
        "ANSWERING DISCIPLINE:\n"
        "- **Singular question = singular answer.** 'NH3 or urea?' → pick one; do not "
        "report numbers for the loser. 'Which feedstock, product, recovery method?' → one "
        "winning configuration, not a menu. Omit irrelevant fields (don't mention urea "
        "recovery if you picked NH3).\n"
        "- **Always answer from the model results.** If the question is at all answerable, "
        "answer it — do not stall, ask for more inputs, or claim data is missing. Vague "
        "phrasing ('most realistic', 'best', 'most attractive') means the user wants your "
        "best interpretation of the model-calculated economics.\n"
        "- **Look up data when challenged.** If the user pushes back ('Isn't that for 10 000 "
        "t/y?', 'That seems too high'), do NOT capitulate or invent a number. Find the "
        "relevant row in the snapshot and answer with the real figure. If your earlier "
        "answer was right, say so and cite the row that proves it; if it was wrong, correct "
        "it explicitly. Never silently change a number to match the user's framing.\n"
        "- **Cite real numbers only.** Every number must come from the provided model results; never "
        "extrapolate or invent values. If a specific cross-product (e.g. Struvite NPV at "
        "10 000 t/y) is not directly in the snapshot, name the closest available point and "
        "give that number — do not refuse the question.\n"
        "\n"
        "WORDING — Use only friendly business/science terms. Never paste "
        "snake_case identifiers (npv_usd_million, ammonia_scp, struvite_map, h2_co2, "
        "us_grid, etc.). Use 'USD X million' or 'USD X/kg' for money, never '$X'. No "
        "LaTeX, no equations. Round to 2–3 significant figures. Do not use any number "
        "unless it appears in the current model results."
    )


def build_messages(
    question: str,
    snapshot: Mapping[str, object],
    history: Optional[Iterable[Mapping[str, str]]] = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt()},
        {
            "role": "system",
            "content": build_chat_context(snapshot),
        },
    ]
    # Cap history to avoid blowing the context window on long conversations
    recent = list(history or [])[-(_MAX_HISTORY_TURNS * 2):]
    for item in recent:
        role = str(item.get("role", "user"))
        if role not in {"user", "assistant"}:
            continue
        messages.append({"role": role, "content": str(item.get("content", ""))})
    messages.append({"role": "user", "content": question})
    return messages


def ask_groq(
    api_key: str,
    question: str,
    snapshot: Mapping[str, object],
    history: Optional[Iterable[Mapping[str, str]]] = None,
    model: str = DEFAULT_GROQ_MODEL,
) -> str:
    if not api_key:
        raise ValueError("A Groq API key is required.")
    if Groq is None:
        raise RuntimeError("The groq package is not installed.")

    grounded = deterministic_answer(question, snapshot)
    if grounded:
        return grounded

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=build_messages(question, snapshot=snapshot, history=history),
        temperature=0.25,
        max_tokens=_MAX_REPLY_TOKENS,
    )
    message = completion.choices[0].message.content
    return message if message is not None else ""
