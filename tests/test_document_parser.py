from __future__ import annotations

import io
from dataclasses import dataclass

from openpyxl import Workbook

from tools.document_parser import classify_transaction_row, parse_csv_file, parse_text_file


@dataclass
class MockUploadedFile:
    name: str
    payload: bytes

    def getvalue(self) -> bytes:
        return self.payload


def test_parse_income_statement_csv_extracts_totals() -> None:
    csv_payload = (
        "Date,Item,Category,Amount,Type\n"
        "2026-04-01,Sales revenue,Sales,4500,Revenue\n"
        "2026-04-01,Stock purchases,Cost of Goods Sold,2700,Expense\n"
        "2026-04-01,Rent,Operating Expense,500,Expense\n"
        "2026-04-01,Transport,Operating Expense,250,Expense\n"
        "2026-04-01,Other expenses,Operating Expense,300,Expense\n"
    ).encode("utf-8")
    uploaded = MockUploadedFile(name="income_statement.csv", payload=csv_payload)

    result = parse_csv_file(uploaded, "Income Statement")

    assert result["document_type"] == "Income Statement"
    assert result["source_type"] == "csv"
    assert result["rows_analyzed"] == 5
    assert result["revenue"] == 4500
    assert result["expenses"] == 3750
    assert result["profit"] == 750
    assert result["profit_margin"] == 16.67


def test_parse_cash_flow_record_detects_inflows_and_outflows() -> None:
    csv_payload = (
        "Date,Description,Cash In,Cash Out,Category\n"
        "2026-04-01,Customer sales,1200,0,Sales\n"
        "2026-04-01,Stock purchase,0,700,Stock\n"
        "2026-04-02,Customer sales,900,0,Sales\n"
        "2026-04-02,Transport payment,0,100,Transport\n"
    ).encode("utf-8")
    uploaded = MockUploadedFile(name="cash_flow.csv", payload=csv_payload)

    result = parse_csv_file(uploaded, "Cash Flow Record")

    assert result["cash_inflow"] == 2100
    assert result["cash_outflow"] == 800
    assert result["revenue"] == 2100
    assert result["expenses"] == 800
    assert result["profit"] == 1300


def test_parse_text_file_uses_text_parser() -> None:
    text_payload = (
        "I made K4,500 in sales and spent K2,700 on stock, K500 on rent, "
        "K250 on transport, and K300 on other expenses. I owe K1,000."
    ).encode("utf-8")
    uploaded = MockUploadedFile(name="notes.txt", payload=text_payload)

    result = parse_text_file(uploaded, "General Business Notes")

    assert result["source_type"] == "txt"
    assert result["document_type"] == "General Business Notes"
    assert result["revenue"] == 4500
    assert result["expenses"] == 3750
    assert result["debt"] == 1000
    assert result["profit"] == 750


def test_classify_transaction_row_uses_document_type() -> None:
    sales_row = {"Product": "Mealie meal", "Total Sales": 900}
    expense_row = {"Description": "Rent", "Amount": 500}

    assert classify_transaction_row(sales_row, "Sales Record") == "revenue"
    assert classify_transaction_row(expense_row, "Expense Record") == "expense"


def test_parse_excel_income_statement_derives_totals_when_total_cells_blank() -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Income Statement"
    ws.append(["Line Item", "Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026"])
    ws.append(["REVENUE", None, None, None, None, None, None])
    ws.append(["Product sales", 8200, 8900, 6800, 6400, 6100, 5900])
    ws.append(["Service income", 1200, 1400, 900, 800, 750, 700])
    ws.append(["Other income", 300, 400, 200, 150, 100, 80])
    ws.append(["Total Revenue", None, None, None, None, None, None])
    ws.append(["COST OF GOODS SOLD", None, None, None, None, None, None])
    ws.append(["Cost of goods sold", 6800, 7400, 6200, 6000, 5900, 5800])
    ws.append(["Freight & delivery", 450, 500, 380, 360, 340, 330])
    ws.append(["OPERATING EXPENSES", None, None, None, None, None, None])
    ws.append(["Rent", 3200, 3200, 3200, 3200, 3200, 3200])
    ws.append(["Staff wages", 2800, 2800, 2800, 2800, 2800, 2800])
    ws.append(["Loan repayment", 800, 800, 800, 800, 800, 800])
    ws.append(["OTHER EXPENSES", None, None, None, None, None, None])
    ws.append(["Other expenses", 420, 380, 310, 290, 270, 260])
    ws.append(["Bank interest (overdraft)", 120, 130, 140, 150, 160, 170])
    ws.append(["Supplier late payment fees", 80, 0, 200, 150, 100, 80])
    ws.append(["Total Opex", 8900, None, None, None, None, None])
    ws.append(["NET PROFIT / (LOSS)", None, None, None, None, None, None])

    payload = io.BytesIO()
    wb.save(payload)
    uploaded = MockUploadedFile(name="FinAgent_Loss_Income_Statement.xlsx", payload=payload.getvalue())

    # Totals expected from summing monthly detailed rows
    expected_revenue = 49280.0
    expected_expenses = 84670.0
    expected_profit = -35390.0

    from tools.document_parser import parse_excel_file

    result = parse_excel_file(uploaded, "Income Statement")

    assert result["source_type"] == "xlsx"
    assert result["document_type"] == "Income Statement"
    assert result["warnings"] == []
    assert result["rows_analyzed"] == 19
    assert result["period_description"] == "Nov 2025 – Apr 2026"
    assert result["revenue"] == expected_revenue
    assert result["expenses"] == expected_expenses
    assert result["profit"] == expected_profit


def test_parse_excel_income_statement_reconciles_operating_vs_net_profit() -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Income Statement"
    ws.append(["Line Item", "Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026", "Total"])
    ws.append(["Total Revenue", 9850, 10890, 8100, 7575, 7230, 6908, 50553])
    ws.append(["Total COGS", 6575, 7264, 5404, 5054, 4826, 4611, 33734])
    ws.append(["Total Operating Expenses", 7245, 7380, 6845, 6788, 6715, 6660, 41633])
    ws.append(["OPERATING PROFIT (EBIT)", -3970, -3754, -4149, -4267, -4311, -4363, -24814])
    ws.append(["Total Finance Costs", 97, 92, 87, 82, 77, 72, 507])
    ws.append(["NET PROFIT", -4067, -3846, -4236, -4349, -4388, -4435, -25321])

    payload = io.BytesIO()
    wb.save(payload)
    uploaded = MockUploadedFile(name="Duka_AI_Income_Statement.xlsx", payload=payload.getvalue())

    from tools.document_parser import parse_excel_file

    result = parse_excel_file(uploaded, "Income Statement")

    assert result["revenue"] == 50553.0
    # Should reconcile to net profit, not operating-expense subtotal.
    assert result["profit"] == -25321.0
    assert result["expenses"] == 75874.0
