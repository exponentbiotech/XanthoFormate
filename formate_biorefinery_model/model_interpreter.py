from __future__ import annotations

from typing import Iterable, Mapping, Optional


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _num(row: Mapping[str, object], key: str) -> Optional[float]:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _fmt_musd(value: object) -> str:
    return f"USD {float(value):.1f}M" if isinstance(value, (int, float)) else "not available"


def _fmt_usdkg(value: object) -> str:
    return f"USD {float(value):.2f}/kg" if isinstance(value, (int, float)) else "not available"


def _fmt_gwp(value: object) -> str:
    return f"{float(value):.2f} kg CO2e/kg" if isinstance(value, (int, float)) else "not available"


def _fmt_capacity(value: object) -> str:
    return f"{float(value):,.0f} t/y" if isinstance(value, (int, float)) else "active scale"


def _active_npv_musd(active: Mapping[str, object]) -> Optional[float]:
    if isinstance(active.get("NPV (million USD)"), (int, float)):
        return float(active["NPV (million USD)"])
    if isinstance(active.get("NPV (USD)"), (int, float)):
        return float(active["NPV (USD)"]) / 1e6
    return None


def _with_active_defaults(row: Mapping[str, object], active: Mapping[str, object]) -> Mapping[str, object]:
    merged = dict(active)
    merged.update(row)
    return merged


def _best(rows: Iterable[Mapping[str, object]], key: str = "NPV (million USD)") -> Mapping[str, object]:
    candidates = [row for row in rows if _num(row, key) is not None]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: float(row[key]))


def _lowest(rows: Iterable[Mapping[str, object]], key: str) -> Mapping[str, object]:
    candidates = [row for row in rows if _num(row, key) is not None]
    if not candidates:
        return {}
    return min(candidates, key=lambda row: float(row[key]))


def _row_at_capacity(rows: Iterable[Mapping[str, object]], capacity: float) -> Mapping[str, object]:
    for row in rows:
        if _num(row, "Plant capacity (t/y)") == capacity:
            return row
    return {}


def _config(row: Mapping[str, object]) -> str:
    pathway = str(row.get("Pathway", "")).strip()
    recovery = row.get("Urea recovery method") if pathway.startswith("Urea") else row.get("NH3 recovery method")
    pieces = [
        pathway,
        f"{row.get('Feedstock')} feedstock" if row.get("Feedstock") else "",
        str(recovery or "").strip(),
        _fmt_capacity(row.get("Plant capacity (t/y)")) if row.get("Plant capacity (t/y)") is not None else "",
        str(row.get("Electricity case", "")).strip(),
    ]
    return ", ".join(piece for piece in pieces if piece)


def _metrics_sentence(row: Mapping[str, object]) -> str:
    npv = row.get("NPV (million USD)")
    if npv is None and isinstance(row.get("NPV (USD)"), (int, float)):
        npv = float(row["NPV (USD)"]) / 1e6
    return (
        f"NPV **{_fmt_musd(npv)}**, Net LCOX **{_fmt_usdkg(row.get('Net LCOX (USD/kg)'))}**, "
        f"Gross LCOX **{_fmt_usdkg(row.get('Gross LCOX (USD/kg)'))}**, and GWP "
        f"**{_fmt_gwp(row.get('GWP (kg CO2e per kg product)'))}**"
    )


def _comparison_line(label: str, row: Mapping[str, object]) -> str:
    return f"- {label}: **{_config(row)}** — {_metrics_sentence(row)}"


def _profitability_answer(data: Mapping[str, object]) -> str:
    active = _as_mapping(data.get("Active scenario"))
    nh3 = [_with_active_defaults(row, active) for row in _rows(data.get("NH3 recovery method comparison"))]
    urea = [_with_active_defaults(row, active) for row in _rows(data.get("Urea recovery method comparison"))]
    feedstock = [_with_active_defaults(row, active) for row in _rows(data.get("Feedstock comparison"))]
    scale = [_with_active_defaults(row, active) for row in _rows(data.get("Capacity scaling"))]

    best_nh3 = _best(nh3)
    best_urea = _best(urea)
    best_current = _best([row for row in (best_nh3, best_urea) if row])
    best_feed = _best(feedstock)
    best_scale = _best(scale)

    lines = [
        "Here is what the Python model actually says, without an LLM choosing the numbers.",
        "",
    ]
    if active:
        lines.append(_comparison_line("Active scenario", active))
    if best_current:
        lines.append(_comparison_line("Best product/recovery at the active scale", best_current))
    if best_feed:
        lines.append(_comparison_line("Best feedstock case at the active scale", best_feed))
    if best_urea:
        lines.append(_comparison_line("Best urea case at the active scale", best_urea))
    if best_scale:
        lines.append(_comparison_line("Best active-route scale shown", best_scale))

    lines.extend([
        "",
        "The scale result is separate from the active-scale result. A 10,000 t/y NPV should not be described as the 1,000 t/y NPV.",
        "Important caveat: Struvite looks economically strong here because the model assigns fertilizer-product value and reports economics per kg NH3-equivalent. That market assumption needs separate validation.",
    ])
    return "\n".join(lines)


def _scale_answer(data: Mapping[str, object], question: str) -> str:
    active = _as_mapping(data.get("Active scenario"))
    scale = [_with_active_defaults(row, active) for row in _rows(data.get("Capacity scaling"))]
    q = question.replace(",", "").lower()
    requested = 10_000.0 if "10000" in q or "10 000" in q else 1_000.0 if "1000" in q or "1 000" in q else None

    if requested is not None:
        row = _row_at_capacity(scale, requested)
        if row:
            return (
                f"For **{_fmt_capacity(requested)}**, the active-route model case is **{_config(row)}**. "
                f"It gives {_metrics_sentence(row)}."
            )

    lines = ["Capacity scaling for the active route:"]
    for row in sorted(scale, key=lambda r: _num(r, "Plant capacity (t/y)") or 0.0):
        lines.append(_comparison_line(_fmt_capacity(row.get("Plant capacity (t/y)")), row))
    return "\n".join(lines)


def _feedstock_answer(data: Mapping[str, object]) -> str:
    active = _as_mapping(data.get("Active scenario"))
    rows = [_with_active_defaults(row, active) for row in _rows(data.get("Feedstock comparison"))]
    best = _best(rows)
    lines = ["Feedstock sensitivity at the active scale:"]
    for row in rows:
        lines.append(_comparison_line(str(row.get("Feedstock", "Feedstock")), row))
    if best:
        lines.append("")
        lines.append(f"Highest NPV in this feedstock sweep: **{_config(best)}**.")
    return "\n".join(lines)


def _recovery_answer(data: Mapping[str, object], question: str) -> str:
    active = _as_mapping(data.get("Active scenario"))
    q = question.lower()
    if "urea" in q and "nh3" not in q and "ammonia" not in q:
        rows = [_with_active_defaults(row, active) for row in _rows(data.get("Urea recovery method comparison"))]
        title = "Urea recovery methods at the active scale:"
    else:
        rows = [_with_active_defaults(row, active) for row in _rows(data.get("NH3 recovery method comparison"))]
        title = "NH3 recovery methods at the active scale:"
    best_npv = _best(rows)
    lowest_gwp = _lowest(rows, "GWP (kg CO2e per kg product)")
    lines = [title]
    for row in rows:
        method = row.get("Urea recovery method") if str(row.get("Pathway", "")).startswith("Urea") else row.get("NH3 recovery method")
        lines.append(_comparison_line(str(method), row))
    if best_npv:
        lines.extend(["", f"Highest NPV: **{_config(best_npv)}**."])
    if lowest_gwp:
        lines.append(f"Lowest GWP: **{_config(lowest_gwp)}**.")
    return "\n".join(lines)


def _lca_answer(data: Mapping[str, object], question: str) -> str:
    active = _as_mapping(data.get("Active scenario"))
    q = question.lower()
    if "electric" in q or "grid" in q or "renewable" in q:
        rows = [_with_active_defaults(row, active) for row in _rows(data.get("Electricity case comparison"))]
        title = "Electricity sensitivity for the active route:"
    else:
        rows = [_with_active_defaults(row, active) for row in _rows(data.get("LCA credit sensitivity"))]
        title = "LCA credit sensitivity:"
    lines = [title]
    for row in rows:
        label = str(row.get("Electricity case") or row.get("CO2 source") or row.get("Pathway") or "Case")
        lines.append(_comparison_line(label, row))
    return "\n".join(lines)


def _active_answer(data: Mapping[str, object]) -> str:
    active = _as_mapping(data.get("Active scenario"))
    if not active:
        return "I could not find an active scenario in the model output."
    return f"The active scenario is **{_config(active)}**. It gives {_metrics_sentence(active)}."


def answer_model_question(question: str, snapshot: Mapping[str, object]) -> str:
    """Answer from Python model outputs only. No LLM calls, no invented values."""
    q = question.lower()
    data = snapshot

    if any(term in q for term in ("scale", "capacity", "1000", "1,000", "10000", "10,000", "10 000")):
        return _scale_answer(data, question)
    if any(term in q for term in ("feedstock", "formate", "methanol", "h2/co2", "h2 co2")):
        return _feedstock_answer(data)
    if any(term in q for term in ("recovery", "struvite", "membrane", "stripping", "crystallization")):
        return _recovery_answer(data, question)
    if any(term in q for term in ("gwp", "lca", "emission", "carbon", "grid", "renewable", "electricity")):
        return _lca_answer(data, question)
    if any(term in q for term in ("profit", "profitable", "profitability", "npv", "reasonable", "realistic", "attractive", "best", "nh3 or urea", "ammonia or urea")):
        return _profitability_answer(data)
    if any(term in q for term in ("current", "active", "selected", "scenario")):
        return _active_answer(data)

    return (
        "I can answer this from the Python model outputs, but I need to map it to a model view. "
        "Try asking about profitability, NPV, scale/capacity, feedstock, recovery method, electricity, LCA/GWP, or the active scenario.\n\n"
        + _active_answer(data)
    )
