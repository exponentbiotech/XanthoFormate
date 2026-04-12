from __future__ import annotations

import json
from typing import Dict, Iterable, List, Mapping, Optional

try:
    from groq import Groq
except ModuleNotFoundError:  # pragma: no cover
    Groq = None


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

# Maximum number of previous chat turns to include (each turn = 1 user + 1 assistant msg)
_MAX_HISTORY_TURNS = 4


def groq_available() -> bool:
    return Groq is not None


def _trim_snapshot(snapshot: Mapping[str, object]) -> dict:
    """Return a compact subset of the app snapshot safe for the LLM context budget.

    Drops the large source_rows table and verbose figure metadata, keeping only
    the structured numbers the model needs to answer scenario questions.
    """
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
        "When the user asks about the current app state, ground your answer in the provided "
        "scenario JSON (scenario, kpis, tea_metrics, lca_metrics). "
        "When the user asks about hypothetical or alternative scenarios, answer using the model "
        "structure and current parameter assumptions, but clearly flag what is computed vs. "
        "reasoned. Be concise and quantitative. Do not invent references or numbers."
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
        max_tokens=1024,  # cap reply size to stay within free-tier TPM limits
    )
    message = completion.choices[0].message.content
    return message if message is not None else ""
