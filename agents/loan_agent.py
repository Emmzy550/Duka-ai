from __future__ import annotations

import json
from typing import Any

from agents import load_prompt, request_llm_json
from tools.financial_calculator import estimate_loan_readiness, format_currency

SYSTEM_PROMPT = """You are Duka AI's Loan Readiness Agent.

PERSONALITY: Evaluative and clear. You think in scores, factors, and thresholds. You are honest about risk without being discouraging — helping the business owner understand exactly where they stand and what would genuinely improve their position. Like a credit officer who is on the owner's side.

YOUR JOB: Assess loan readiness, explain the score's drivers, and give concrete improvement steps based on the supplied data.

RULES:
- Use ONLY the financial figures provided — never recalculate amounts or ratios yourself.
- State the score and identify specific factors from the data that raised or lowered it.
- Improvement steps must directly address the gaps visible in the actual data — not generic advice.
- FORMULA RULE: When citing a suggested loan amount, always include the formula: "Based on 1.5× your monthly profit (a common SME lending heuristic), a manageable amount is K{amount}." Never mention a suggested amount without this formula.
- Reference Zambian lending institutions only where genuinely relevant — do not force institution names into every response.
- Use Kwacha (K) naturally. Do not force the word.
- Be honest and conservative — never present borrowing as a goal.
- End responses naturally. Do not use templated endings like "...in Kwacha?"
- Return JSON only.

Return this JSON shape:
{
  "risk_level": "Low | Moderate | High",
  "reason": "specific reason with exact figures from the supplied data",
  "safe_borrowing_advice": "specific advice with amounts derived from the actual profit and debt figures",
  "how_to_improve": ["concrete step 1 tied to the actual data gaps", "concrete step 2", "concrete step 3"],
  "warnings": ["specific warning derived from the supplied figures"],
  "reasoning": "brief explanation using the exact figures"
}
"""


def analyze_loan_readiness(
    user_input: str,
    parsed_data: dict[str, Any],
    cashflow_result: dict[str, Any],
) -> dict[str, Any]:
    revenue = parsed_data.get("revenue", 0.0)
    expenses = parsed_data.get("expenses", 0.0)
    debt = parsed_data.get("debt", 0.0)
    requested_loan_amount = parsed_data.get("loan_amount", 0.0)
    profit = cashflow_result["profit"]

    readiness = estimate_loan_readiness(revenue, expenses, debt=debt)
    score = readiness["score"]

    if score >= 75:
        risk_level = "Low"
    elif score >= 50:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # Conservative safe-loan formula:
    # - No existing debt → suggest up to 1.5× monthly profit (more capacity available)
    # - Existing debt present → cap at 1.0× monthly profit (debt already competes for cash)
    # Example: profit=K550, no debt → K550 × 1.5 = K825 suggested max loan
    suggested_loan_amount = max(0.0, round(profit * (1.0 if debt > 0 else 1.5), 2))
    warnings: list[str] = []

    if requested_loan_amount > 0 and requested_loan_amount > max(profit * 2, 1):
        warnings.append("The requested loan looks large compared with current profit.")
    if debt > profit:
        warnings.append("Existing debt already exceeds current profit.")
    if profit <= 0:
        warnings.append("Borrowing is especially risky while the business is not generating positive profit.")

    if requested_loan_amount <= 0:
        reason = "The business can be scored for borrowing readiness even though no specific loan amount was provided."
    elif requested_loan_amount <= suggested_loan_amount and score >= 60:
        reason = "The requested loan may be manageable, but only if repayment supports quick sales and spending stays disciplined."
    else:
        reason = "The requested loan looks risky relative to current profit, margin, and debt pressure."

    loan_multiplier = 1.0 if debt > 0 else 1.5
    safe_borrowing_advice = (
        f"Based on {loan_multiplier}× your monthly profit (a common SME lending heuristic), "
        f"a manageable amount is {format_currency(suggested_loan_amount)}. "
        f"Loan readiness score: {score}/100 ({readiness['status']})."
    )

    context_bundle = parsed_data.get("_analysis_context", {})
    document_type = ((context_bundle.get("document_analysis") or {}).get("document_type") or "").strip()
    how_to_improve = [
        "Improve records so you can show at least a few weeks of stable sales and expenses.",
        "Increase profit before taking a larger loan repayment burden.",
        "Reduce existing supplier debt where possible.",
    ]
    if document_type == "Loan or Debt Record":
        how_to_improve.insert(0, "Use the debt record to check outstanding balances and due dates before taking on any new loan.")
    if document_type in {"Bank Statement", "Mobile Money Statement"}:
        how_to_improve.insert(0, "Review the uploaded statement for repayment pressure and repeated debt-related outflows.")
    if requested_loan_amount > 0:
        how_to_improve.append("Choose a smaller loan that directly supports fast-moving stock or proven demand.")

    llm_result = request_llm_json(
        SYSTEM_PROMPT + "\n\n" + load_prompt("loan_prompt.md"),
        (
            f"User input:\n{user_input}\n\n"
            f"Structured financial data:\n{json.dumps(parsed_data, indent=2)}\n\n"
            f"Cash flow analysis:\n{json.dumps(cashflow_result, indent=2)}\n\n"
            f"Analysis context:\n{json.dumps(context_bundle, indent=2)}\n\n"
            f"Loan readiness baseline:\n{json.dumps(readiness, indent=2)}\n\n"
            f"Deterministic borrowing baseline:\n{json.dumps({'loan_readiness_score': score, 'status': readiness['status'], 'risk_level': risk_level, 'reason': reason, 'safe_borrowing_advice': safe_borrowing_advice, 'how_to_improve': how_to_improve[:4], 'warnings': warnings}, indent=2)}\n\n"
            "Explain borrowing readiness for this Zambian SME owner."
        ),
    )
    llm_improvements = llm_result.get("how_to_improve") if isinstance(llm_result.get("how_to_improve") if llm_result else None, list) else None
    llm_warnings = llm_result.get("warnings") if isinstance(llm_result.get("warnings") if llm_result else None, list) else None
    merged_warnings = warnings[:]
    if llm_warnings:
        for warning in llm_warnings:
            if isinstance(warning, str) and warning.strip() and warning.strip() not in merged_warnings:
                merged_warnings.append(warning.strip())

    return {
        "loan_readiness_score": score,
        "status": readiness["status"],
        "risk_level": (llm_result or {}).get("risk_level", risk_level),
        "reason": (llm_result or {}).get("reason", reason),
        "safe_borrowing_advice": (llm_result or {}).get("safe_borrowing_advice", safe_borrowing_advice),
        "requested_loan_amount": requested_loan_amount,
        "suggested_loan_amount": suggested_loan_amount,
        "loan_multiplier": loan_multiplier,
        "how_to_improve": llm_improvements[:4] if llm_improvements else how_to_improve[:4],
        "warnings": merged_warnings,
        "reasoning": (llm_result or {}).get("reasoning", ""),
        "used_llm": bool(llm_result),
    }
