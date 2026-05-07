from __future__ import annotations

import json
from typing import Any

from agents import load_prompt, request_llm_json
from tools.financial_calculator import (
    calculate_expense_ratio,
    calculate_profit,
    calculate_profit_margin,
    classify_cash_flow_status,
    format_currency,
)

SYSTEM_PROMPT = """You are Duka AI's Cash Flow Agent.

PERSONALITY: Analytical and data-focused, but plain-spoken. You are like a sharp accountant who explains numbers in plain English. You lead with figures, then explain what they mean for the business owner's day-to-day decisions.

YOUR JOB: Analyze cash flow, revenue, expenses, profit, and margins for Zambian small businesses.

RULES:
- Use ONLY the financial figures provided in the prompt — never invent or recalculate numbers.
- Reference the user's actual figures (revenue, expenses, profit, expense ratio) from the supplied data.
- Use Kwacha (K) naturally when referring to amounts — not forced into every sentence.
- Explain what the numbers mean practically for this specific business.
- Be honest about risks, but frame them constructively.
- Do NOT include specific advice like market names, example store names, or fixed savings amounts — base all advice on the actual data supplied.
- End responses naturally. Do NOT end every response with "...in Kwacha?" or any templated phrase.
- Return JSON only.

Return this JSON shape:
{
  "summary": "specific explanation using the figures from the supplied data",
  "cash_flow_status": "Healthy | Moderate | Tight | Negative | Unclear",
  "warnings": ["specific warning derived from the supplied figures"],
  "reasoning": "plain-English reasoning using the exact figures provided"
}
"""


def analyze_cash_flow(user_input: str, parsed_data: dict[str, Any]) -> dict[str, Any]:
    revenue = parsed_data.get("revenue", 0.0)
    expenses = parsed_data.get("expenses", 0.0)
    debt = parsed_data.get("debt", 0.0)

    profit = calculate_profit(revenue, expenses)
    profit_margin = calculate_profit_margin(revenue, profit)
    expense_ratio = calculate_expense_ratio(revenue, expenses)
    cash_flow_status = classify_cash_flow_status(
        revenue=revenue,
        expenses=expenses,
        profit=profit,
        profit_margin=profit_margin,
        debt=debt,
    )

    warnings: list[str] = []
    if revenue <= 0:
        warnings.append("Revenue could not be confidently detected from the business description.")
    if expenses <= 0:
        warnings.append("Expenses look incomplete, so the profit estimate may be too optimistic.")
    if profit <= 0:
        warnings.append("The business is not covering its expenses with the current numbers.")
    if expense_ratio >= 80:
        warnings.append("Expenses are taking a very large share of revenue.")
    if 0 < profit_margin < 15:
        warnings.append("Profit margin is thin, so the cash cushion is still weak.")
    if debt > 0 and debt >= max(profit, 1):
        warnings.append("Existing debt is high compared with current profit.")

    context_bundle = parsed_data.get("_analysis_context", {})

    # Compute top-3 expense total in Python so the LLM never has to sum them itself
    breakdown = parsed_data.get("expenses_breakdown") or {}
    top3_items = sorted(
        [(k, float(v or 0)) for k, v in breakdown.items() if float(v or 0) > 0],
        key=lambda x: x[1], reverse=True,
    )[:3]
    top3_total = round(sum(v for _, v in top3_items), 2)
    verified_expense_block = ""
    if top3_items:
        items_str = ", ".join(f"{k}: {format_currency(v)}" for k, v in top3_items)
        verified_expense_block = (
            f"\n\nVERIFIED TOP-3 EXPENSE TOTAL (computed by Python — do NOT recompute):\n"
            f"Items: {items_str}\n"
            f"Total: EXACTLY {format_currency(top3_total)} — use this figure verbatim if mentioning a combined total."
        )

    llm_result = request_llm_json(
        SYSTEM_PROMPT + "\n\n" + load_prompt("cashflow_prompt.md"),
        (
            f"User input:\n{user_input}\n\n"
            f"Structured financial data:\n{json.dumps(parsed_data, indent=2)}\n\n"
            f"Analysis context:\n{json.dumps(context_bundle, indent=2)}\n\n"
            f"Calculated metrics:\n{json.dumps({'revenue': revenue, 'expenses': expenses, 'profit': profit, 'profit_margin': profit_margin, 'expense_ratio': expense_ratio, 'cash_flow_status': cash_flow_status, 'warnings': warnings}, indent=2)}"
            f"{verified_expense_block}\n\n"
            "Analyze the cash flow for a Zambian SME owner using these exact numbers. Do NOT add up any figures yourself."
        ),
    )
    llm_summary = llm_result.get("summary") if llm_result else None
    llm_status = llm_result.get("cash_flow_status") if llm_result else None
    llm_warnings = llm_result.get("warnings") if isinstance(llm_result.get("warnings") if llm_result else None, list) else None
    llm_reasoning = llm_result.get("reasoning") if llm_result else None
    merged_warnings = warnings[:]
    if llm_warnings:
        for warning in llm_warnings:
            if isinstance(warning, str) and warning.strip() and warning.strip() not in merged_warnings:
                merged_warnings.append(warning.strip())

    return {
        "revenue": revenue,
        "expenses": expenses,
        "profit": profit,
        "profit_margin": profit_margin,
        "expense_ratio": expense_ratio,
        "cash_flow_status": llm_status or cash_flow_status,
        "summary": llm_summary or "",
        "warning": merged_warnings[0] if merged_warnings else "",
        "warnings": merged_warnings,
        "reasoning": llm_reasoning or "",
        "used_llm": bool(llm_result),
    }
