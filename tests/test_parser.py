from tools.text_parser import parse_business_input


def test_parse_business_input_with_kwacha_values() -> None:
    sample = (
        "I run a small grocery shop in Lusaka. This week I made K4,500 in sales. "
        "I spent K2,700 on stock, K500 on rent, K250 on transport, and K300 on other expenses. "
        "I also owe K1,000 to my supplier. Can I afford to take a K3,000 loan?"
    )

    result = parse_business_input(sample)

    assert result["revenue"] == 4500
    assert result["expenses"] == 3750
    assert result["expenses_breakdown"]["stock"] == 2700
    assert result["expenses_breakdown"]["rent"] == 500
    assert result["expenses_breakdown"]["transport"] == 250
    assert result["expenses_breakdown"]["other_expenses"] == 300
    assert result["debt"] == 1000
    assert result["loan_amount"] == 3000
    assert result["business_type"] == "grocery shop"
    assert result["location"] == "Lusaka"


def test_parse_business_input_with_plain_numbers() -> None:
    sample = "I made 2000 in sales and my total expenses were 1400."
    result = parse_business_input(sample)

    assert result["revenue"] == 2000
    assert result["expenses"] == 1400
