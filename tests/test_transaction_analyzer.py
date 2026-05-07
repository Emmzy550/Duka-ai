from __future__ import annotations

import pandas as pd

from tools.transaction_analyzer import analyze_transactions


def test_analyze_transactions_returns_expected_totals() -> None:
    frame = pd.DataFrame(
        [
            {"Date": "2026-04-01", "Provider": "MTN Mobile Money", "Description": "Customer payment", "Category": "Sales", "Amount": 1200, "Type": "Revenue"},
            {"Date": "2026-04-01", "Provider": "MTN Mobile Money", "Description": "Stock purchase", "Category": "Supplier", "Amount": 700, "Type": "Expense"},
            {"Date": "2026-04-02", "Provider": "Airtel Money", "Description": "Customer payment", "Category": "Sales", "Amount": 900, "Type": "Revenue"},
            {"Date": "2026-04-02", "Provider": "Airtel Money", "Description": "Transport payment", "Category": "Transport", "Amount": 100, "Type": "Expense"},
            {"Date": "2026-04-03", "Provider": "Bank Account", "Description": "Supplier debt payment", "Category": "Debt", "Amount": 500, "Type": "Debt"},
        ]
    )

    result = analyze_transactions(frame)

    assert result["total_revenue"] == 2100
    assert result["total_expenses"] == 800
    assert result["debt_payments"] == 500
    assert result["net_cash_flow"] == 1300
    assert result["biggest_expense_category"] == "Supplier"
    assert result["cash_flow_risk"] in {"Low", "Moderate", "High"}
    assert result["rows_analyzed"] == 5
