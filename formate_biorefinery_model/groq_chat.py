from __future__ import annotations

import json
from typing import Dict, Iterable, List, Mapping, Optional

try:
    from groq import Groq
except ModuleNotFoundError:  # pragma: no cover
    Groq = None


DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def groq_available() -> bool:
    return Groq is not None


def build_chat_context(snapshot: Mapping[str, object]) -> str:
    """Serialize the current app state for a grounded LLM prompt."""
    return json.dumps(snapshot, indent=2, ensure_ascii=True, default=str)


def system_prompt() -> str:
    return (
        "You are an assistant embedded inside a techno-economic and life-cycle-analysis "
        "Streamlit app for a formate biorefinery. Be grounded in the current app state "
        "when the user asks about what is currently shown. If the user asks about another "
        "scenario, answer using the model structure and current assumptions, but be explicit "
        "about what is directly computed in the current app state versus what is a reasoned "
        "extension or hypothetical. Cite parameter names, values, and source rows when helpful. "
        "Do not invent references. If information is not in the app state, say so clearly."
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
            "content": "Current app context:\n" + build_chat_context(snapshot),
        },
    ]
    for item in history or []:
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
    )
    message = completion.choices[0].message.content
    return message if message is not None else ""
