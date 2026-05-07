from __future__ import annotations

from typing import Any

import pandas as pd


def load_transactions_frame(source: str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        frame = pd.read_csv(source)

    frame.columns = [str(column).strip() for column in frame.columns]
    if "Amount" in frame.columns:
        frame["Amount"] = pd.to_numeric(frame["Amount"], errors="coerce").fillna(0.0)
    else:
        frame["Amount"] = 0.0

    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame


def _normalized_type(value: Any) -> str:
    return str(value).strip().lower()


def analyze_transactions(source: str | pd.DataFrame) -> dict[str, Any]:
    frame = load_transactions_frame(source)
    if frame.empty:
        return {
            "rows_analyzed": 0,
            "total_revenue": 0.0,
            "total_expenses": 0.0,
            "debt_payments": 0.0,
            "net_cash_flow": 0.0,
            "average_daily_sales": 0.0,
            "biggest_expense_category": "Unknown",
            "recurring_expense_patterns": [],
            "cash_flow_risk": "Unclear",
            "transaction_stability_score": 0,
            "date_range": "Unknown",
            "providers": [],
            "summary": "No transactions were available for analysis.",
        }

    frame["TypeNormalized"] = frame.get("Type", "").apply(_normalized_type)
    revenue_rows = frame[frame["TypeNormalized"] == "revenue"]
    expense_rows = frame[frame["TypeNormalized"] == "expense"]
    debt_rows = frame[frame["TypeNormalized"] == "debt"]

    total_revenue = float(revenue_rows["Amount"].sum())
    total_expenses = float(expense_rows["Amount"].sum())
    debt_payments = float(debt_rows["Amount"].sum())
    net_cash_flow = total_revenue - total_expenses

    if not revenue_rows.empty and "Date" in revenue_rows.columns and revenue_rows["Date"].notna().any():
        daily_sales = revenue_rows.groupby(revenue_rows["Date"].dt.date)["Amount"].sum()
        average_daily_sales = float(daily_sales.mean())
        stability_ratio = float(daily_sales.std(ddof=0) / daily_sales.mean()) if daily_sales.mean() else 1.0
    else:
        average_daily_sales = total_revenue
        stability_ratio = 1.0

    biggest_expense_category = "Unknown"
    recurring_expense_patterns: list[str] = []
    if not expense_rows.empty and "Category" in expense_rows.columns:
        expense_by_category = expense_rows.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        biggest_expense_category = str(expense_by_category.index[0])

        recurring_counts = expense_rows.groupby("Category").size().sort_values(ascending=False)
        recurring_expense_patterns = [
            f"{category}: {count} transaction(s)"
            for category, count in recurring_counts.items()
            if count > 1
        ]

    expense_ratio = (total_expenses / total_revenue) * 100 if total_revenue > 0 else 0.0
    if total_revenue <= 0:
        cash_flow_risk = "High"
    elif net_cash_flow <= 0 or expense_ratio >= 90:
        cash_flow_risk = "High"
    elif expense_ratio >= 70:
        cash_flow_risk = "Moderate"
    else:
        cash_flow_risk = "Low"

    stability_score = max(0, min(100, round(85 - (stability_ratio * 25) - max(0, expense_ratio - 60) * 0.5)))

    start_date = frame["Date"].min() if "Date" in frame.columns else None
    end_date = frame["Date"].max() if "Date" in frame.columns else None
    if pd.notna(start_date) and pd.notna(end_date):
        date_range = f"{start_date.date()} to {end_date.date()}"
    else:
        date_range = "Unknown"

    providers = sorted(set(frame.get("Provider", pd.Series(dtype=str)).dropna().astype(str).tolist()))

    summary = (
        "The business shows "
        f"{'positive' if net_cash_flow > 0 else 'tight'} net cash flow. "
        f"{biggest_expense_category} is the largest expense category, and expenses are taking "
        f"{expense_ratio:.1f}% of revenue."
    )

    return {
        "rows_analyzed": int(len(frame)),
        "total_revenue": round(total_revenue, 2),
        "total_expenses": round(total_expenses, 2),
        "debt_payments": round(debt_payments, 2),
        "net_cash_flow": round(net_cash_flow, 2),
        "average_daily_sales": round(average_daily_sales, 2),
        "biggest_expense_category": biggest_expense_category,
        "recurring_expense_patterns": recurring_expense_patterns,
        "cash_flow_risk": cash_flow_risk,
        "transaction_stability_score": int(stability_score),
        "date_range": date_range,
        "providers": providers,
        "summary": summary,
    }
