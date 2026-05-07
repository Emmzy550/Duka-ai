from __future__ import annotations

import re
from typing import Any

from tools.financial_calculator import calculate_profit, format_currency
from tools.financial_engine import build_forecast_brief, detect_forecast_intent


def build_followup_visual(question: str, report: dict[str, Any]) -> dict[str, Any] | None:
    question_lower = question.lower().strip()
    cashflow = report["cashflow"]
    document = report.get("document_analysis") or {}
    transaction = report.get("transaction_summary") or {}
    loan = report["loan"]

    if "where is most of my money going" in question_lower or "where is my money going" in question_lower:
        items: list[dict[str, str]] = []
        for category in document.get("biggest_expense_categories", [])[:3]:
            items.append({"label": "Expense category", "value": category})
        if transaction.get("biggest_expense_category"):
            items.append({"label": "Largest transaction expense", "value": str(transaction["biggest_expense_category"])})
        if not items:
            items.append({"label": "Expense ratio", "value": f"{cashflow['expense_ratio']:.1f}%"})
        return {"title": "Expense Focus", "items": items}

    if "uploaded document" in question_lower or "document show" in question_lower or "income statement" in question_lower:
        return {
            "title": "Document Snapshot",
            "items": [
                {"label": "Revenue", "value": format_currency(document.get("revenue", cashflow["revenue"]))},
                {"label": "Expenses", "value": format_currency(document.get("expenses", cashflow["expenses"]))},
                {"label": "Profit", "value": format_currency(document.get("profit", cashflow["profit"]))},
                {"label": "Profit Margin", "value": f"{document.get('profit_margin', cashflow['profit_margin']):.1f}%"},
            ],
        }

    if "sales drop" in question_lower or "drop next month" in question_lower:
        revenue = cashflow["revenue"]
        expenses = cashflow["expenses"]
        ten_drop_profit = calculate_profit(revenue * 0.9, expenses)
        twenty_drop_profit = calculate_profit(revenue * 0.8, expenses)
        return {
            "title": "Sales Drop Scenario",
            "items": [
                {"label": "Current Profit", "value": format_currency(cashflow["profit"])},
                {"label": "Profit if Sales Drop 10%", "value": format_currency(ten_drop_profit)},
                {"label": "Profit if Sales Drop 20%", "value": format_currency(twenty_drop_profit)},
            ],
        }

    if "borrow" in question_lower or "loan" in question_lower:
        match = re.search(r"(?:k|zmw)?\s*([\d,]+(?:\.\d+)?)", question_lower)
        requested = float(match.group(1).replace(",", "")) if match else loan.get("requested_loan_amount", 0.0)
        if requested > 0:
            risk_level = "Higher than safer range" if requested > loan["suggested_loan_amount"] else "Within safer range"
            return {
                "title": "Borrowing Scenario",
                "items": [
                    {"label": "Current Profit", "value": format_currency(cashflow["profit"])},
                    {"label": "Requested Loan", "value": format_currency(requested)},
                    {"label": "Safer Level", "value": format_currency(loan["suggested_loan_amount"])},
                    {"label": "Risk View", "value": risk_level},
                ],
            }

    if "cash flow" in question_lower:
        return {
            "title": "Cash Flow Snapshot",
            "items": [
                {"label": "Revenue", "value": format_currency(cashflow["revenue"])},
                {"label": "Expenses", "value": format_currency(cashflow["expenses"])},
                {"label": "Profit", "value": format_currency(cashflow["profit"])},
                {"label": "Status", "value": cashflow["cash_flow_status"]},
            ],
        }

    return None


_CHART_TRIGGERS = [
    "graph", "chart", "plot", "visualize", "visualise",
    "pie chart", "bar chart", "line graph", "line chart",
    "show me a graph", "show me a chart", "show on a graph", "on a graph",
    "show graph", "show chart",
]


_CHART_AFFIRM = ("yes", "yeah", "yep", "sure", "ok", "okay", "please", "go ahead", "show me", "show it")


def _is_short_affirmation(q: str) -> bool:
    if len(q.split()) > 6:
        return False
    return any(q == w or q.startswith(w + " ") or q.startswith(w + ",") or q.startswith(w + ".") for w in _CHART_AFFIRM)


def _last_user_forecast_intent(messages: list[dict] | None, current_question: str) -> dict[str, Any] | None:
    if not messages:
        return None
    for prev in reversed(messages):
        if prev.get("role") != "user":
            continue
        content = prev.get("content", "")
        if content == current_question:
            continue
        intent = detect_forecast_intent(content)
        if intent:
            return intent
    return None


def build_chart_data(
    question: str,
    report: dict[str, Any],
    messages: list[dict] | None = None,
) -> dict[str, Any] | None:
    """Return chart data when the user asks for a graph/chart/plot."""
    q = question.lower().strip()

    has_chart_kw = any(kw in q for kw in _CHART_TRIGGERS)
    is_affirmation = _is_short_affirmation(q)

    if not has_chart_kw and not is_affirmation:
        return None

    cashflow = report["cashflow"]
    parsed = report.get("parsed_data", {})
    document = report.get("document_analysis") or {}
    loan = report.get("loan", {})

    # Forecast line chart — current question is forward-looking + chart kw,
    # OR the user is affirming a prior chart offer that followed a forecast question.
    forecast_intent = detect_forecast_intent(question) if has_chart_kw else None
    if forecast_intent is None and is_affirmation:
        forecast_intent = _last_user_forecast_intent(messages, question)

    # If we got here with only an affirmation and no forecast context, do nothing.
    if not has_chart_kw and not forecast_intent:
        return None

    if forecast_intent:
        brief = build_forecast_brief(report, forecast_intent["months"])
        if brief:
            base = brief["base_case"]["months"]
            opt = brief["optimistic"]["months"]
            pess = brief["pessimistic"]["months"]
            return {
                "chart_type": "line",
                "title": f"Forecast — Next {brief['months']} Months",
                "months": [m["month_name"] for m in base],
                "series": [
                    {"name": "Base Case Profit", "values": [m["profit"] for m in base], "color": "#2563EB"},
                    {"name": "Optimistic Profit", "values": [m["profit"] for m in opt], "color": "#10B981"},
                    {"name": "Pessimistic Profit", "values": [m["profit"] for m in pess], "color": "#EF4444"},
                ],
            }

    # Monthly trend — line chart (requires monthly breakdown data)
    monthly = document.get("monthly_breakdown") or []
    if monthly and any(kw in q for kw in ["month", "trend", "over time", "history", "line"]):
        months = [str(e.get("month", "")) for e in monthly]
        revenues = [float(e.get("revenue") or e.get("Revenue") or 0) for e in monthly]
        expenses_vals = [float(e.get("expenses") or e.get("Expenses") or 0) for e in monthly]
        profits = [float(e.get("profit") or e.get("Profit") or 0) for e in monthly]
        if any(revenues) or any(profits):
            return {
                "chart_type": "line",
                "title": "Monthly Performance Trend",
                "months": months,
                "series": [
                    {"name": "Revenue", "values": revenues, "color": "#10B981"},
                    {"name": "Expenses", "values": expenses_vals, "color": "#EF4444"},
                    {"name": "Profit", "values": profits, "color": "#2563EB"},
                ],
            }

    # Expense breakdown — pie chart
    if any(kw in q for kw in ["expense", "spending", "cost", "breakdown", "money going", "where"]):
        breakdown = parsed.get("expenses_breakdown") or {}
        items = [(k, float(v or 0)) for k, v in breakdown.items() if float(v or 0) > 0]
        if items:
            return {
                "chart_type": "pie",
                "title": "Expense Breakdown",
                "labels": [k.replace("_", " ").title() for k, _ in items],
                "values": [v for _, v in items],
            }

    # Loan scenario — bar chart
    if any(kw in q for kw in ["loan", "borrow", "credit"]):
        safe_loan = float(loan.get("suggested_loan_amount") or 0)
        profit = float(cashflow.get("profit", 0))
        return {
            "chart_type": "bar",
            "title": "Loan Overview",
            "labels": ["Monthly Profit", "Safe Loan Amount"],
            "values": [profit, safe_loan],
            "colors": ["#10B981", "#2563EB"],
        }

    # Default — cash flow bar chart
    return {
        "chart_type": "bar",
        "title": "Cash Flow Overview",
        "labels": ["Revenue", "Expenses", "Profit"],
        "values": [
            float(cashflow.get("revenue", 0)),
            float(cashflow.get("expenses", 0)),
            float(cashflow.get("profit", 0)),
        ],
        "colors": ["#10B981", "#EF4444", "#2563EB"],
    }
