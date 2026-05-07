from __future__ import annotations

from typing import Any


def calculate_profit(revenue: float, expenses: float) -> float:
    return round(revenue - expenses, 2)


def calculate_profit_margin(revenue: float, profit: float) -> float:
    if revenue <= 0:
        return 0.0
    return round((profit / revenue) * 100, 2)


def calculate_expense_ratio(revenue: float, expenses: float) -> float:
    if revenue <= 0:
        return 0.0
    return round((expenses / revenue) * 100, 2)


def calculate_business_health_score(revenue: float, expenses: float, debt: float = 0) -> dict[str, Any]:
    profit = calculate_profit(revenue, expenses)
    margin = calculate_profit_margin(revenue, profit)
    expense_ratio = calculate_expense_ratio(revenue, expenses)
    score = 0

    if profit > 0:
        score += 30
    if margin > 20:
        score += 25
    if expense_ratio < 70:
        score += 20
    if debt < max(profit, 0):
        score += 15
    if revenue > expenses:
        score += 10

    score = max(0, min(100, score))

    if score >= 80:
        status = "Healthy"
    elif score >= 60:
        status = "Stable but needs attention"
    elif score >= 40:
        status = "At risk"
    else:
        status = "Critical"

    return {
        "score": score,
        "status": status,
        "profit": profit,
        "margin": margin,
        "expense_ratio": expense_ratio,
    }


def estimate_loan_readiness(revenue: float, expenses: float, debt: float = 0) -> dict[str, Any]:
    profit = calculate_profit(revenue, expenses)
    margin = calculate_profit_margin(revenue, profit)
    expense_ratio = calculate_expense_ratio(revenue, expenses)

    score = 50

    if profit > 0:
        score += 20
    else:
        score -= 25

    if margin >= 25:
        score += 20
    elif margin >= 10:
        score += 10
    else:
        score -= 10

    if debt > profit:
        score -= 20
    elif debt > 0:
        score -= 10

    score = max(0, min(100, score))

    if score >= 75:
        status = "Loan ready"
    elif score >= 50:
        status = "Needs improvement"
    else:
        status = "High risk"

    return {
        "score": score,
        "status": status,
        "profit": profit,
        "margin": margin,
        "expense_ratio": expense_ratio,
    }


def classify_cash_flow_status(
    *,
    revenue: float,
    expenses: float,
    profit: float,
    profit_margin: float,
    debt: float = 0,
) -> str:
    if revenue <= 0:
        return "Unclear"
    if profit <= 0:
        return "Negative"
    if profit_margin >= 25 and debt <= profit and expenses < revenue:
        return "Healthy"
    if profit_margin >= 10:
        return "Moderate"
    return "Tight"


def format_currency(value: float) -> str:
    return f"K{value:,.2f}"
