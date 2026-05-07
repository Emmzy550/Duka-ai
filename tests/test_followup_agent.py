from __future__ import annotations

from agents.followup_agent import answer_followup_question, stream_followup_for_agent


def _sample_report() -> dict:
    return {
        "cashflow": {
            "revenue": 4500.0,
            "expenses": 3750.0,
            "profit": 750.0,
            "profit_margin": 16.67,
            "expense_ratio": 83.33,
            "cash_flow_status": "Moderate",
        },
        "advisor": {
            "needs_attention": "Expenses are taking a large share of revenue.",
            "best_next_move": "Focus on fast-moving stock and tighten cost control.",
            "this_weeks_action": "Track daily sales and expenses for one full week.",
        },
        "loan": {
            "loan_readiness_score": 60,
            "suggested_loan_amount": 1000.0,
            "reason": "The requested loan looks risky relative to current profit and debt pressure.",
        },
        "market": {
            "market_summary": "For a grocery shop, focus on fast-moving essentials.",
            "monitor_next": ["supplier prices", "transport costs", "weekly sales trend"],
        },
        "business_health": {
            "score": 60,
            "status": "Stable but needs attention",
        },
        "document_analysis": {
            "document_type": "Income Statement",
            "revenue": 4500.0,
            "expenses": 3750.0,
            "profit": 750.0,
            "debt": 1000.0,
            "summary": "This income statement shows positive profit but tight margins.",
            "biggest_expense_categories": ["Cost of Goods Sold", "Operating Expense"],
        },
        "parsed_data": {
            "revenue": 4500.0,
            "expenses": 3750.0,
            "debt": 1000.0,
            "expenses_breakdown": {
                "Cost Of Goods Sold": 2700.0,
                "Rent": 500.0,
                "Transport": 250.0,
                "Other Expenses": 300.0,
            },
        },
        "transaction_summary": {
            "biggest_expense_category": "Supplier",
        },
    }


def test_followup_answers_about_uploaded_document(monkeypatch) -> None:
    monkeypatch.setattr("agents.followup_agent.request_llm", lambda *args, **kwargs: None)
    response = answer_followup_question("What does my uploaded document show?", _sample_report(), [])

    assert "income statement" in response["answer"].lower()
    assert "K4,500.00" in response["answer"]
    assert "K3,750.00" in response["answer"]


def test_followup_answers_where_money_is_going(monkeypatch) -> None:
    monkeypatch.setattr("agents.followup_agent.request_llm", lambda *args, **kwargs: None)
    response = answer_followup_question("Where is most of my money going?", _sample_report(), [])

    assert "Cost of Goods Sold" in response["answer"] or "Supplier" in response["answer"]


def test_cashflow_stream_breakdown_is_deterministic_without_llm() -> None:
    chunks = list(
        stream_followup_for_agent(
            "cashflow",
            "Where is my money going?",
            _sample_report(),
            session_messages=[],
        )
    )
    text = "".join(chunks)
    assert "Top expense drivers" in text
    assert "Cost Of Goods Sold" in text
    assert "Rent" in text
