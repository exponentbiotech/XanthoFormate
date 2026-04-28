from __future__ import annotations

import json
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
# model returns fewer tokens, so we keep this conservative.
_MAX_REPLY_TOKENS = 600


def groq_available() -> bool:
    return Groq is not None


_COMPREHENSIVE_KEYS = (
    "notes",
    "computed_rankings",
    "active_scenario",
    "nh3_recovery_method_comparison",
    "urea_recovery_method_comparison",
    "feedstock_comparison",
    "electricity_case_comparison",
    "capacity_scaling",
    "lca_credit_sensitivity",
)


def _trim_snapshot(snapshot: Mapping[str, object]) -> dict:
    """Return a compact subset of the app snapshot safe for the LLM context budget.

    Supports two snapshot shapes:
      * The legacy single-scenario shape (scenario, kpis, tea_metrics, lca_metrics).
      * The comprehensive shape (active_scenario + cross-scenario comparisons),
        which lets the LLM reason about production modes that are not currently
        selected (e.g. alternate recovery methods, feedstocks, electricity cases).
    """
    if "active_scenario" in snapshot:
        return {key: snapshot[key] for key in _COMPREHENSIVE_KEYS if key in snapshot}
    return {
        "scenario": snapshot.get("scenario"),
        "kpis": snapshot.get("kpis"),
        "tea_metrics": snapshot.get("tea_metrics"),
        "lca_metrics": snapshot.get("lca_metrics"),
    }


def build_chat_context(snapshot: Mapping[str, object]) -> str:
    """Serialize a trimmed app state snapshot for a grounded LLM prompt."""
    return json.dumps(_trim_snapshot(snapshot), indent=2, ensure_ascii=True, default=str)


def system_prompt() -> str:
    return (
        "You are an expert assistant embedded inside a techno-economic analysis (TEA) and "
        "life-cycle-assessment (LCA) Streamlit app for a formate biorefinery producing ammonia "
        "and/or urea plus single-cell protein (SCP) using engineered Xanthomonas flavus GJ10. "
        "\n\n"
        "GROUNDING — The user-message attachment titled 'Current app state (JSON)' contains "
        "either a single active scenario or, when available, a comprehensive comparison "
        "covering ALL production modes plus a computed_rankings object with deterministic "
        "Python-ranked winners. The comparisons cover every NH3 recovery method, every urea recovery "
        "method, every feedstock pathway (formate, H2/CO2, methanol), every electricity "
        "case (US grid vs renewable), capacity scaling (100 / 1 000 / 10 000 t/y), and "
        "LCA credit sensitivity. "
        "\n"
        "When the user asks open-ended questions like 'which is most profitable?', 'which "
        "feedstock should we use?', or 'which recovery method has the lowest GWP?', you MUST "
        "consult computed_rankings first, then use the cross-scenario arrays for supporting "
        "detail. Do NOT restrict your answer to the active_scenario only. Compare net_lcox_usd_per_kg, npv_usd_million, and "
        "primary_product_gwp_kgco2e_per_kg across the relevant comparison list, name the "
        "winning configuration explicitly (category, feedstock, recovery method, capacity, "
        "electricity case), and quote the supporting numbers."
        "\n\n"
        "FORMATTING — IMPORTANT:"
        "\n  * Write currency as 'USD 13.78' or 'USD 13.78/kg', NOT '$13.78'."
        "\n  * NEVER use LaTeX math notation (no $...$, no $$...$$, no \\( \\), no \\[ \\])."
        "\n  * Use plain text and bullet lists. No equations."
        "\n  * Be concise and quantitative; round to 2-3 significant figures."
        "\n  * Do not invent references or numbers — every number you cite must come from the "
        "provided JSON."
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
            "content": "Current app state (JSON):\n" + build_chat_context(snapshot),
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

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=build_messages(question, snapshot=snapshot, history=history),
        temperature=0.2,
        max_tokens=_MAX_REPLY_TOKENS,
    )
    message = completion.choices[0].message.content
    return message if message is not None else ""
