from __future__ import annotations

from typing import Any


def get_fallback_market_context(business_type: str | None, location: str | None) -> dict[str, Any]:
    business_key = (business_type or "default").lower()
    location_name = location or "the local market"

    base = {
        "market_summary": (
            f"Duka AI is using fallback market intelligence for this MVP. "
            f"Live trusted-source market search can be added in the production version."
        ),
        "risk_factors": ["General operating costs may shift faster than selling prices."],
        "opportunity": "Track where cash turns over fastest and protect working capital there first.",
        "recommendation": "Use financial decisions that protect short-term cash flow and proven demand.",
        "monitor_next": ["weekly sales trend", "supplier pricing", "transport costs"],
    }

    playbooks = {
        "grocery shop": {
            "market_summary": f"For a grocery or small shop business in {location_name}, cash should stay in fast-moving essentials rather than slow inventory.",
            "risk_factors": [
                "Transport costs can reduce margins.",
                "Slow-moving stock can trap cash.",
                "Supplier price changes can weaken restocking power.",
            ],
            "opportunity": "Focus on daily essentials and products customers buy repeatedly.",
            "recommendation": "Use any new borrowing only for fast-moving goods with quick turnover.",
            "monitor_next": ["supplier prices", "transport costs", "weekly sales trend"],
        },
        "salon": {
            "market_summary": f"For a salon or barbershop in {location_name}, demand may cluster around weekends and month-end pay cycles.",
            "risk_factors": [
                "Slow weekdays can reduce cash stability.",
                "Product costs can rise faster than service pricing.",
                "Equipment debt can create pressure if demand slows.",
            ],
            "opportunity": "Bundle high-demand services and simple add-ons to raise average spend.",
            "recommendation": "Use promotions carefully and avoid taking debt for equipment unless bookings are stable.",
            "monitor_next": ["weekday booking levels", "product refill costs", "month-end demand"],
        },
        "farmer": {
            "market_summary": f"For farming or agri-business near {location_name}, seasonality, weather, and input costs can shape cash flow sharply.",
            "risk_factors": [
                "Weather risk can delay expected sales.",
                "Fertilizer, feed, or input costs may rise before harvest.",
                "Borrowing too much before income lands can create repayment gaps.",
            ],
            "opportunity": "Track crop or produce prices before selling and plan cash around seasonal gaps.",
            "recommendation": "Avoid heavy borrowing before harvest unless demand, yield, and sale timing are reasonably clear.",
            "monitor_next": ["weather outlook", "input prices", "crop sale prices"],
        },
        "freelancer": {
            "market_summary": f"For a freelance or service business in {location_name}, payment timing matters as much as total booked work.",
            "risk_factors": [
                "Late payments can weaken cash flow.",
                "Demand may vary from month to month.",
                "Low pricing can hide a workload problem.",
            ],
            "opportunity": "Recurring retainers and deposit-based work can smooth income.",
            "recommendation": "Push for upfront deposits and keep a cash buffer for delayed client payments.",
            "monitor_next": ["invoice collection time", "repeat clients", "monthly income consistency"],
        },
        "default": {
            "market_summary": f"For this business in {location_name}, inventory discipline, pricing awareness, and cash protection matter most.",
            "risk_factors": [
                "Supplier costs may move faster than prices charged to customers.",
                "Cash can get tied up in slow products or low-return activities.",
            ],
            "opportunity": "Direct money toward the products or services that already convert quickly.",
            "recommendation": "Review what sells fastest and avoid new spending unless it supports proven demand.",
            "monitor_next": ["weekly revenue trend", "supplier prices", "customer demand shifts"],
        },
    }

    return playbooks.get(business_key, playbooks["default"]) | {
        "fallback_notice": base["market_summary"],
    }


def get_market_context(business_type: str | None, location: str | None) -> dict[str, Any]:
    return get_fallback_market_context(business_type, location)


def search_trusted_market_news(query: str) -> list[dict[str, str]]:
    # Placeholder for a future production integration using vetted sources.
    return []
