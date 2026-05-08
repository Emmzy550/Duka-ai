from __future__ import annotations

from typing import Any

from agents import request_llm
from agents.advisor_agent import analyze_financial_advice
from agents.cashflow_agent import analyze_cash_flow
from agents.loan_agent import analyze_loan_readiness
from agents.market_intelligence_agent import analyze_market_intelligence
from tools.financial_calculator import calculate_business_health_score, format_currency
from tools.text_parser import parse_business_input


def _business_identity_snippet(business_profile: dict[str, Any] | None, parsed_data: dict[str, Any]) -> str:
    """Human-readable business line for prompts — keeps summaries grounded."""
    bp = business_profile or {}
    bt = (bp.get("business_type") or "").strip()
    loc = (bp.get("location") or "").strip()
    ps = (bp.get("products_services") or "").strip()
    parts: list[str] = []
    if bt:
        parts.append(bt)
    elif parsed_data.get("business_type"):
        parts.append(str(parsed_data["business_type"]).strip())
    if loc:
        parts.append(f"in {loc}")
    if ps:
        parts.append(f"selling/offering: {ps[:120]}")
    return ", ".join(parts) if parts else "this SME (type not specified)"


def _numeric_anchor_block(
    *,
    parsed_data: dict[str, Any],
    cashflow: dict[str, Any],
    business_health: dict[str, Any],
    loan: dict[str, Any],
) -> str:
    """Exact figures the executive summary MUST quote — reduces generic hedging."""
    rev = float(cashflow.get("revenue") or parsed_data.get("revenue") or 0)
    exp = float(cashflow.get("expenses") or parsed_data.get("expenses") or 0)
    profit = float(cashflow.get("profit") or 0)
    margin = float(cashflow.get("profit_margin") or 0)
    er = float(cashflow.get("expense_ratio") or 0)
    debt = float(parsed_data.get("debt") or 0)
    return (
        f"Monthly revenue (use this label): {format_currency(rev)}\n"
        f"Monthly expenses: {format_currency(exp)}\n"
        f"Monthly profit: {format_currency(profit)}\n"
        f"Profit margin: {margin:.1f}%\n"
        f"Expense ratio (expenses/revenue): {er:.1f}%\n"
        f"Outstanding debt: {format_currency(debt)}\n"
        f"Business health score: {business_health.get('score')}/100 — {business_health.get('status')}\n"
        f"Loan readiness score: {loan.get('loan_readiness_score')}/100\n"
    )


def _deduplicate_actions(actions: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_actions: list[str] = []
    for action in actions:
        normalized = action.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_actions.append(action.strip())
    return unique_actions


_INSIGHT_TOPIC_KEYWORDS: list[tuple[str, list[str]]] = [
    ("expenses", ["expense", "cost", "spending", "ratio", "high cost", "outgoing"]),
    ("profit", ["profit", "margin", "thin margin", "low profit"]),
    ("debt", ["debt", "loan", "borrow", "repay"]),
    ("revenue", ["revenue", "sales", "income", "turnover"]),
]


def _deduplicate_insights(items: list[str]) -> list[str]:
    """Remove semantically duplicate insights, keeping the more specific (longer) one per topic."""
    def get_topics(text: str) -> frozenset[str]:
        t = text.lower()
        return frozenset(topic for topic, kws in _INSIGHT_TOPIC_KEYWORDS if any(kw in t for kw in kws))

    result: list[str] = []
    for item in items:
        item_topics = get_topics(item)
        merged = False
        for i, existing in enumerate(result):
            if item_topics and item_topics == get_topics(existing):
                if len(item) > len(existing):
                    result[i] = item
                merged = True
                break
        if not merged:
            result.append(item)
    return result


def _compute_top3_expense_summary(parsed_data: dict[str, Any]) -> tuple[list[tuple[str, float]], float]:
    """Compute top-3 expenses and their verified total in Python before passing to any LLM."""
    breakdown = parsed_data.get("expenses_breakdown") or {}
    items = sorted(
        [(k, float(v or 0)) for k, v in breakdown.items() if float(v or 0) > 0],
        key=lambda x: x[1], reverse=True,
    )
    top3 = items[:3]
    total = round(sum(v for _, v in top3), 2)
    return top3, total


def _generate_executive_summary(
    *,
    advisor: dict[str, Any],
    cashflow: dict[str, Any],
    loan: dict[str, Any],
    business_health: dict[str, Any],
    analysis_basis: list[str] | None,
    parsed_data: dict[str, Any] | None = None,
    business_profile: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    pd = parsed_data or {}
    bp = business_profile or {}
    top3_expenses, top3_total = _compute_top3_expense_summary(pd)
    verified_expense_line = ""
    if top3_expenses:
        items_str = ", ".join(f"{k}: {format_currency(v)}" for k, v in top3_expenses)
        verified_expense_line = (
            f"\nVERIFIED TOP-3 EXPENSE TOTAL (computed by Python): EXACTLY {format_currency(top3_total)}"
            f" ({items_str}). Use this exact figure — do NOT recompute the sum yourself."
        )

    identity = _business_identity_snippet(bp, pd)
    anchors = _numeric_anchor_block(
        parsed_data=pd, cashflow=cashflow, business_health=business_health, loan=loan
    )

    exec_system = """You are Duka AI's Executive Summary Agent. Write for an African SME owner in plain English.

STRICT RULES:
- ONE paragraph only (3–5 short sentences).
- Sentence 1 MUST feel specific to THIS business: name the business type and city/area from BUSINESS IDENTITY when provided.
- Sentence 1 MUST quote these exact figures using K for Kwacha (copy from NUMERIC ANCHORS): monthly revenue, expenses, profit, profit margin %, and outstanding debt. Do not substitute synonyms for the amounts.
- Do NOT open with vague hedge language alone (e.g. "moderate cash flow situation", "thin profit margin", "faces challenges", "overall", "generally", "it appears that"). If you use qualitative words, they must appear together with the exact K amounts in the same sentence or right after.
- Reference the biggest expense lever when TOP-3 expense lines are provided (name categories with K amounts).
- Mention loan readiness briefly using the score supplied; no invented limits.
- CRITICAL: Use ONLY numbers from the prompt — never compute new totals yourself."""

    llm_summary = request_llm(
        exec_system,
        (
            f"BUSINESS IDENTITY (use in sentence 1): {identity}\n\n"
            f"NUMERIC ANCHORS (copy these amounts exactly — do not round differently):\n{anchors}\n"
            f"Cash flow narrative (for context only — still anchor to NUMERIC ANCHORS):\n{cashflow.get('summary', '')}\n\n"
            f"What this means (advisor): {advisor.get('what_this_means', '')}\n"
            f"Best next move: {advisor.get('best_next_move', '')}\n"
            f"Borrowing: score {loan.get('loan_readiness_score')}/100 — {loan.get('safe_borrowing_advice', '')}\n"
            f"Analysis notes: {' '.join(analysis_basis or [])}\n"
            f"{verified_expense_line}\n\n"
            "Write the single paragraph executive summary now."
        ),
        max_tokens=220,
    )
    return (llm_summary.strip() if llm_summary else "", bool(llm_summary))


def generate_business_report(
    user_input: str,
    *,
    parsed_data_override: dict[str, Any] | None = None,
    analysis_basis: list[str] | None = None,
    source_labels: list[str] | None = None,
    transaction_summary: dict[str, Any] | None = None,
    business_profile: dict[str, Any] | None = None,
    document_analysis: dict[str, Any] | None = None,
    data_sources: list[str] | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    effective_input = user_input.strip() or "Uploaded business records were provided for analysis."
    parsed_data = dict(parsed_data_override or parse_business_input(effective_input))
    parsed_data["_analysis_context"] = {
        "business_profile": business_profile or {},
        "document_analysis": document_analysis or {},
        "transaction_summary": transaction_summary or {},
        "source_labels": source_labels or [],
        "analysis_basis": analysis_basis or [],
    }
    if progress_callback:
        progress_callback("cashflow")
    cashflow = analyze_cash_flow(effective_input, parsed_data)

    if progress_callback:
        progress_callback("advisor")
    advisor = analyze_financial_advice(effective_input, parsed_data, cashflow)

    if progress_callback:
        progress_callback("loan")
    loan = analyze_loan_readiness(effective_input, parsed_data, cashflow)

    if progress_callback:
        progress_callback("market")
    market = analyze_market_intelligence(effective_input, parsed_data, cashflow, transaction_summary=transaction_summary)
    business_health = calculate_business_health_score(
        cashflow["revenue"],
        cashflow["expenses"],
        parsed_data.get("debt", 0.0),
    )

    risk_warnings = _deduplicate_insights(
        _deduplicate_actions([*cashflow.get("warnings", []), *loan.get("warnings", [])])
    )
    if transaction_summary:
        transaction_patterns = transaction_summary.get("recurring_expense_patterns", [])
        if transaction_summary.get("cash_flow_risk") == "High":
            risk_warnings.append("Connected transaction data shows a high cash flow risk pattern.")
        if transaction_patterns:
            risk_warnings.append(f"Recurring expense pattern detected: {transaction_patterns[0]}")
        risk_warnings = _deduplicate_insights(_deduplicate_actions(risk_warnings))

    final_actions = _deduplicate_actions(
        [
            *advisor["next_actions"],
            market["recommendation"],
            "Keep weekly records so you can see revenue, expenses, profit, and debt clearly over time.",
        ]
    )[:5]

    if business_health["score"] < 60:
        final_actions = _deduplicate_actions(
            final_actions
            + [
                "Delay new borrowing until cash flow and records improve.",
                "Focus on reducing costs and improving sales consistency first.",
            ]
        )[:5]
    else:
        final_actions = _deduplicate_actions(
            final_actions
            + [
                "Build a cash buffer before taking on unnecessary debt.",
                "Reinvest only in fast-moving stock or low-risk growth opportunities.",
            ]
        )[:5]

    if progress_callback:
        progress_callback("summary")
    final_summary, summary_used_llm = _generate_executive_summary(
        advisor=advisor,
        cashflow=cashflow,
        loan=loan,
        business_health=business_health,
        analysis_basis=analysis_basis,
        parsed_data=parsed_data,
        business_profile=business_profile or {},
    )
    if transaction_summary:
        final_summary += (
            f" Connected financial data indicates average daily sales near {format_currency(transaction_summary['average_daily_sales'])} "
            f"with a {transaction_summary['cash_flow_risk'].lower()} transaction risk profile."
        )

    return {
        "input": effective_input,
        "parsed_data": parsed_data,
        "cashflow": cashflow,
        "advisor": advisor,
        "loan": loan,
        "market": market,
        "business_health": business_health,
        "transaction_summary": transaction_summary or {},
        "risk_warnings": risk_warnings,
        "final_recommended_actions": final_actions,
        "final_summary": final_summary,
        "analysis_basis": analysis_basis or [],
        "source_labels": source_labels or [],
        "data_sources": data_sources or [],
        "business_profile": business_profile or {},
        "document_analysis": document_analysis or {},
        "agent_statuses": {
            "Executive Summary": summary_used_llm,
            "Cash Flow Agent": cashflow["used_llm"],
            "Financial Advisor Agent": advisor["used_llm"],
            "Borrowing & Debt Agent": loan["used_llm"],
            "Market Intelligence Agent": market["used_llm"],
        },
        "summary_used_llm": summary_used_llm,
        "used_llm": any(
            [
                summary_used_llm,
                cashflow["used_llm"],
                advisor["used_llm"],
                loan["used_llm"],
                market["used_llm"],
            ]
        ),
    }
