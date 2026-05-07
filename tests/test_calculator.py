from tools.financial_calculator import (
    calculate_business_health_score,
    calculate_profit,
    calculate_profit_margin,
    estimate_loan_readiness,
)


def test_calculate_profit() -> None:
    assert calculate_profit(4500, 3750) == 750


def test_calculate_profit_margin() -> None:
    assert calculate_profit_margin(4500, 750) == 16.67


def test_estimate_loan_readiness_for_demo_case() -> None:
    result = estimate_loan_readiness(4500, 3750, debt=1000)
    assert result["score"] == 60
    assert result["status"] == "Needs improvement"


def test_calculate_business_health_score_for_healthy_case() -> None:
    result = calculate_business_health_score(6000, 3600, debt=500)
    assert result["score"] == 100
    assert result["status"] == "Healthy"
