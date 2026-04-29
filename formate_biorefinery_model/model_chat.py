"""Code-aware chat for the biorefinery TEA/LCA app.

The chat module sends the user's question to Groq's hosted Llama 3.3 along with:

  1. ``LIVE_RESULTS`` — the locked numeric snapshot (active scenario plus every
     recovery-method, feedstock, electricity, capacity, and LCA-credit
     comparison row). All numeric claims must be quoted from this section.
  2. ``SOURCE_CODE`` — the full Python source for ``tea.py`` and ``lca.py`` so
     the model can answer methodology questions ("how is NPV computed?",
     "where does the biogenic-carbon credit come from?") by referring to the
     actual code rather than guessing.

If a Groq API key is not configured (or the call fails) we fall back to the
deterministic ``answer_model_question`` interpreter so users still get a
useful, accurate reply rather than an error toast.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from .model_interpreter import answer_model_question


DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
"""Groq's most capable free-tier model with a 128k context window."""

# How many of the most recent chat turns to forward to the LLM. Old turns are
# dropped to stay well under Groq's free-tier TPM ceiling and to make sure each
# answer is grounded in the latest snapshot, not stale numbers from previous
# scenarios.
_MAX_HISTORY_TURNS = 6


# ──────────────────────────────────────────────────────────────────────────────
#  Source code excerpts the LLM uses to answer methodology questions.
# ──────────────────────────────────────────────────────────────────────────────


_THIS_DIR = Path(__file__).resolve().parent


def _read_source(name: str) -> str:
    """Read one of the model's Python source files.

    Returns an empty string if the file is missing rather than raising; the
    chat module has to keep working even in stripped-down deployments.
    """
    path = _THIS_DIR / name
    try:
        return path.read_text()
    except OSError:
        return ""


def _build_source_excerpt() -> str:
    """Concatenate the TEA + LCA source so the LLM can quote and explain it.

    The two files are small (≈325 lines combined) and contain the canonical
    NPV / LCOX / GWP formulas. Sending them verbatim is much more reliable
    than trying to summarise the math in the system prompt.
    """
    sections = [
        ("formate_biorefinery_model/tea.py", _read_source("tea.py")),
        ("formate_biorefinery_model/lca.py", _read_source("lca.py")),
    ]
    parts: List[str] = []
    for label, body in sections:
        if not body.strip():
            continue
        parts.append(f"### {label}\n```python\n{body.rstrip()}\n```")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
#  System prompt
# ──────────────────────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """You are the senior TEA/LCA analyst for a Xanthobacter C1
biorefinery model. You're briefing the project lead and investors.

You have two grounded sources you can use in every answer:

1. LIVE_RESULTS — the actual Python model output for the user's currently
   selected sidebar scenario, plus comparison grids for every NH3 recovery
   method, urea recovery method, feedstock, electricity case, capacity, and
   LCA-credit setting. This is the ONLY source of numeric truth.

2. SOURCE_CODE — the full Python source for `tea.py` (NPV, LCOX, capital
   recovery factor, cash-flow construction) and `lca.py` (cradle-to-gate
   GWP, biogenic carbon credit, displacement credit). Use this to explain
   methodology. You may quote short snippets in fenced code blocks.

Hard rules:

- Every number you cite (USD/kg, M USD, NPV, GWP, CapEx, etc.) must come
  verbatim from LIVE_RESULTS. Never compute on your own. Never invent.
  Never make up rounded versions of values that don't appear there.
- Read the user's question carefully. If they ask for "least", "worst",
  "lowest", "most expensive", or another inverse, do NOT default to the
  best-NPV / lowest-LCOX answer. Look in LIVE_RESULTS for the row that
  actually matches the direction of the question.
- For "how does X work?" / "what does this code do?" / "where is Y
  defined?" questions, refer to SOURCE_CODE. Quote short snippets when
  helpful and explain in plain English.
- Always be specific about the configuration you're describing
  (pathway + feedstock + recovery method + capacity + electricity case),
  using the plain-English labels already in LIVE_RESULTS. Never echo
  snake_case identifiers like `npv_usd_million` or `struvite_map`.
- Be conversational, expert, and concise — like a senior analyst, not a
  templated report. Use short paragraphs and bullet lists when it
  improves clarity.
- If the user asks about a configuration that isn't in LIVE_RESULTS, say
  what's available and briefly suggest how to extend the model.
- When discussing Struvite or MAP fertilizer routes, remember LCOX is
  reported per kg NH3-equivalent (see `tea.py`), and that the fertilizer
  revenue is treated as a credit for LCOX accounting but is excluded from
  the NPV cash flow to avoid double-counting product revenue.
"""


# ──────────────────────────────────────────────────────────────────────────────
#  LIVE_RESULTS payload formatting
# ──────────────────────────────────────────────────────────────────────────────


def _format_live_results(snapshot: Mapping[str, object]) -> str:
    """Serialize the deterministic snapshot as the LIVE_RESULTS section.

    JSON is used because the snapshot is already plain-English (every key and
    categorical value has been translated by ``app_support._friendlify``) and
    JSON keeps the structure unambiguous for the LLM. We only pretty-print with
    a small indent to save tokens.
    """
    return json.dumps(snapshot, indent=1, default=str, sort_keys=False)


# ──────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ──────────────────────────────────────────────────────────────────────────────


def _truncate_history(
    history: Sequence[Mapping[str, str]],
) -> List[Mapping[str, str]]:
    """Keep only the most recent ``_MAX_HISTORY_TURNS`` user/assistant turns."""
    pairs: List[Mapping[str, str]] = []
    for entry in history:
        role = entry.get("role")
        content = entry.get("content", "")
        if role in ("user", "assistant") and content:
            pairs.append({"role": role, "content": str(content)})
    if len(pairs) > _MAX_HISTORY_TURNS * 2:
        pairs = pairs[-_MAX_HISTORY_TURNS * 2 :]
    return pairs


def _build_messages(
    question: str,
    snapshot: Mapping[str, object],
    history: Sequence[Mapping[str, str]],
) -> List[Mapping[str, str]]:
    """Construct the message list for the Groq chat completion call."""
    grounding = (
        "Below are the two grounded sources for this conversation.\n\n"
        "===== LIVE_RESULTS (use ONLY these for numeric values) =====\n"
        f"{_format_live_results(snapshot)}\n"
        "===== END LIVE_RESULTS =====\n\n"
        "===== SOURCE_CODE (use to explain methodology) =====\n"
        f"{_build_source_excerpt()}\n"
        "===== END SOURCE_CODE =====\n"
    )
    messages: List[Mapping[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "system", "content": grounding},
    ]
    messages.extend(_truncate_history(history))
    messages.append({"role": "user", "content": question})
    return messages


def _resolve_api_key(explicit_key: Optional[str]) -> Optional[str]:
    """Pick the first available Groq API key from explicit arg / env var."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    return env_key or None


def is_llm_available(api_key: Optional[str] = None) -> bool:
    """True iff a Groq API key is configured for this session."""
    return bool(_resolve_api_key(api_key))


def answer_question(
    question: str,
    snapshot: Mapping[str, object],
    history: Iterable[Mapping[str, str]] = (),
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_GROQ_MODEL,
    temperature: float = 0.4,
    max_tokens: int = 900,
) -> str:
    """Answer ``question`` using Groq with code + locked numbers in context.

    Falls back to the deterministic ``answer_model_question`` interpreter when
    no API key is configured or the LLM call raises.
    """
    history_list = list(history)
    resolved_key = _resolve_api_key(api_key)
    if not resolved_key:
        return answer_model_question(question, snapshot)

    try:
        from groq import Groq  # noqa: WPS433  (lazy import keeps cold-start fast)
    except Exception:  # pragma: no cover - groq package missing in some envs
        return answer_model_question(question, snapshot)

    try:
        client = Groq(api_key=resolved_key)
        completion = client.chat.completions.create(
            model=model,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            messages=list(_build_messages(question, snapshot, history_list)),
        )
        reply = (completion.choices[0].message.content or "").strip()
        if not reply:
            return answer_model_question(question, snapshot)
        return reply
    except Exception as exc:  # pragma: no cover - network/quota errors
        deterministic = answer_model_question(question, snapshot)
        return (
            f"_The hosted LLM call failed ({exc.__class__.__name__}). "
            "Falling back to the deterministic model interpreter:_\n\n"
            f"{deterministic}"
        )


__all__ = [
    "DEFAULT_GROQ_MODEL",
    "answer_question",
    "is_llm_available",
]
