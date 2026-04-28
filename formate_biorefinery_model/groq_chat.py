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
    "Notes",
    "Precomputed best options",
    "Active scenario",
    "NH3 recovery method comparison",
    "Urea recovery method comparison",
    "Feedstock comparison",
    "Electricity case comparison",
    "Capacity scaling",
    "LCA credit sensitivity",
)


def _trim_snapshot(snapshot: Mapping[str, object]) -> dict:
    """Return a compact subset of the app snapshot safe for the LLM context budget.

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


def build_chat_context(snapshot: Mapping[str, object]) -> str:
    """Serialize a trimmed app state snapshot for a grounded LLM prompt."""
    return json.dumps(_trim_snapshot(snapshot), indent=2, ensure_ascii=True, default=str)


def system_prompt() -> str:
    return (
        "You are an assistant inside a TEA + LCA app for a formate biorefinery producing "
        "ammonia and/or urea plus single-cell protein (SCP) with Xanthomonas flavus GJ10.\n"
        "\n"
        "DATA: A JSON snapshot follows. All keys and values are ALREADY plain English "
        "(e.g. 'NPV (million USD)', 'Net LCOX (USD/kg)', 'Ammonia + SCP', 'Struvite "
        "(MgNH4PO4)'). The 'Precomputed best options' section holds Python-computed "
        "winners — consult it FIRST for any 'which is best / most profitable / lowest cost "
        "/ lowest GWP' question. Use the comparison arrays only for supporting detail.\n"
        "\n"
        "ANSWERING RULES:\n"
        "1. SINGULAR question = SINGULAR answer. Asked 'NH3 or urea?' — pick ONE; do not "
        "report numbers for the loser. Asked 'which feedstock, product, recovery method?' — "
        "give ONE winning configuration, not a list.\n"
        "2. OMIT irrelevant fields. If you picked NH3 do not mention urea recovery method, "
        "and vice versa.\n"
        "3. Use the computed winner — do not try to re-rank rows yourself.\n"
        "4. State the winning configuration (pathway, feedstock, recovery method, capacity, "
        "electricity) and quote the supporting NPV / Net LCOX / GWP numbers from the JSON.\n"
        "\n"
        "WORDING (STRICT):\n"
        "- PLAIN ENGLISH ONLY. NEVER paste any snake_case identifier (e.g. npv_usd_million, "
        "net_lcox_usd_per_kg, ammonia_scp, struvite_map, mvr_crystallization, us_grid, "
        "h2_co2). Use the friendly labels that appear in the JSON.\n"
        "- Refer to concepts in natural English; do not quote raw JSON section names.\n"
        "- Currency: write 'USD 144 million' or 'USD -6.12/kg'. NEVER write '$144M'.\n"
        "- NEVER use LaTeX (no $...$, no $$...$$, no \\(...\\), no \\[...\\]).\n"
        "- Plain text and bullet lists only; under 150 words; round to 2-3 sig figs.\n"
        "- Cite only numbers that are present in the provided JSON.\n"
        "\n"
        "GOOD answer style: '**Ammonia + SCP** with **Formate (CO2 electrolysis)** feedstock, "
        "**Struvite (MgNH4PO4)** recovery, 1,000 t/y, renewable electricity. NPV: USD 144 "
        "million. Net LCOX: USD -6.12/kg. GWP: 4.5 kg CO2e/kg.'"
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
        temperature=0.1,
        max_tokens=_MAX_REPLY_TOKENS,
    )
    message = completion.choices[0].message.content
    return message if message is not None else ""
