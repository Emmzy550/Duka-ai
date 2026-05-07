from __future__ import annotations

from typing import Any

from agents import request_llm
from agents.advisor_agent import analyze_financial_advice
from agents.cashflow_agent import analyze_cash_flow
from agents.loan_agent import analyze_loan_readiness
from agents.market_intelligence_agent import analyze_market_intelligence
from tools.financial_calculator import calculate_business_health_score, format_currency
from tools.text_parser import parse_business_input


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
) -> tuple[str, bool]:
    top3_expenses, top3_total = _compute_top3_expense_summary(parsed_data or {})
    verified_expense_line = ""
    if top3_expenses:
        items_str = ", ".join(f"{k}: {format_currency(v)}" for k, v in top3_expenses)
        verified_expense_line = (
            f"\nVERIFIED TOP-3 EXPENSE TOTAL (computed by Python): EXACTLY {format_currency(top3_total)}"
            f" ({items_str}). Use this exact figure — do NOT recompute the sum yourself."
        )

    llm_summary = request_llm(
        "You are Duka AI's Executive Summary Agent. Write a short, natural, practical summary for an African SME owner in simple English. Lead with business health, profit, and next move. Mention borrowing only briefly at the end. CRITICAL: use ONLY the numbers supplied — never compute totals yourself.",
        (
            f"Business health: {business_health['status']} ({business_health['score']}/100)\n"
            f"Cash flow summary: {cashflow['summary']}\n"
            f"Financial advice: {advisor['what_this_means']}\n"
            f"Best next move: {advisor['best_next_move']}\n"
            f"Borrowing summary: score {loan['loan_readiness_score']}/100, {loan['safe_borrowing_advice']}\n"
            f"Analysis notes: {' '.join(analysis_basis or [])}\n"
            f"{verified_expense_line}\n\n"
            "Write one short paragraph only. Do NOT add up any numbers yourself."
        ),
        max_tokens=180,
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
