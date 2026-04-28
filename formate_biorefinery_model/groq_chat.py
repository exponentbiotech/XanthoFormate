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
# model returns fewer tokens, so we keep this conservative — but big enough
# for a 2-3 paragraph conversational answer with numbers.
_MAX_REPLY_TOKENS = 700


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
        "the winning configuration or a key number.\n"
        "\n"
        "DATA — The JSON snapshot is ALWAYS fully populated when you receive it; it is "
        "built deterministically from the live Python model. NEVER claim the snapshot is "
        "empty, missing, loading, or 'waiting for data'. If you are unsure, scan the JSON "
        "again — every section ('Precomputed best options', 'Active scenario', the various "
        "comparison arrays) contains real numbers. The 'Precomputed best options' section "
        "holds Python-ranked winners as compact one-line summaries — consult it FIRST for "
        "any 'which is best / most profitable / lowest cost / lowest GWP / most realistic' "
        "question and quote the winner directly.\n"
        "\n"
        "ANSWERING DISCIPLINE:\n"
        "- **Singular question = singular answer.** 'NH3 or urea?' → pick one; do not "
        "report numbers for the loser. 'Which feedstock, product, recovery method?' → one "
        "winning configuration, not a menu. Omit irrelevant fields (don't mention urea "
        "recovery if you picked NH3).\n"
        "- **Always answer from the data.** If the question is at all answerable from the "
        "snapshot, answer it — do not stall, ask for more inputs, or claim data is missing. "
        "Vague phrasing ('most realistic', 'best', 'most attractive') means the user wants "
        "your best-pick from 'Precomputed best options'.\n"
        "- **Look up data when challenged.** If the user pushes back ('Isn't that for 10 000 "
        "t/y?', 'That seems too high'), do NOT capitulate or invent a number. Find the "
        "relevant row in the snapshot and answer with the real figure. If your earlier "
        "answer was right, say so and cite the row that proves it; if it was wrong, correct "
        "it explicitly. Never silently change a number to match the user's framing.\n"
        "- **Cite real numbers only.** Every number must come from the JSON; never "
        "extrapolate or invent values. If a specific cross-product (e.g. Struvite NPV at "
        "10 000 t/y) is not directly in the snapshot, name the closest available point and "
        "give that number — do not refuse the question.\n"
        "\n"
        "WORDING — Use only the friendly labels that appear in the JSON. Never paste "
        "snake_case identifiers (npv_usd_million, ammonia_scp, struvite_map, h2_co2, "
        "us_grid, etc.). Currency: 'USD 144 million', 'USD -6.12/kg' — never '$144M'. No "
        "LaTeX, no equations. Round to 2–3 significant figures.\n"
        "\n"
        "STYLE EXAMPLE for 'NH3 or urea, which is more profitable?':\n"
        "  'Ammonia wins comfortably. The strongest configuration is **Ammonia + SCP** with "
        "formate feedstock and Struvite (MgNH4PO4) recovery at 1,000 t/y on renewable "
        "electricity — NPV around USD 144 million and a Net LCOX of USD -6.12/kg (the SCP "
        "credit pushes it negative). The best urea route lands well below that; happy to "
        "walk through that comparison if useful.'"
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
