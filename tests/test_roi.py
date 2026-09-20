"""
Unit tests for Real Estate ROI & Financial Modeling Engine (roi.py).
"""

import math
import pytest

from roi import (
    FinancialAssumptions,
    FinancingPlan,
    RealEstateROIEngine,
    calculate_mortgage_payment,
)


def test_calculate_mortgage_payment():
    # 1. Zero rate: straight line payment
    p1 = calculate_mortgage_payment(1_200_000, annual_rate=0.0, years=10)
    assert math.isclose(p1, 10_000.0, rel_tol=1e-4)

    # 2. Standard positive mortgage: 1,000,000 at 12% over 10 years
    p2 = calculate_mortgage_payment(1_000_000, annual_rate=0.12, years=10)
    # Approx 14,347.09 EGP/mo
    assert 14_000.0 <= p2 <= 15_000.0

    # 3. Edge cases
    assert calculate_mortgage_payment(0, 0.1, 10) == 0.0
    assert calculate_mortgage_payment(1_000_000, 0.1, 0) == 0.0


def test_roi_yield_calculations():
    engine = RealEstateROIEngine(model_bundle=None)

    purchase_price = 10_000_000.0 # 10 Million EGP
    monthly_rent = 75_000.0       # 75,000 EGP/mo

    report = engine.evaluate_investment(
        purchase_price_egp=purchase_price,
        monthly_rent_egp=monthly_rent,
        accessibility_score=85.0,
        is_compound=True,
    )

    # Annual rent = 75,000 * 12 = 900,000 EGP
    assert report.annual_gross_rent_egp == 900_000.0

    # Gross yield = 900,000 / 10,000,000 = 9.0%
    assert math.isclose(report.gross_rental_yield_pct, 9.0, abs_tol=0.05)

    # Net yield must be strictly lower than gross yield due to expenses
    assert report.net_rental_yield_pct < report.gross_rental_yield_pct
    assert report.net_rental_yield_pct > 0.0

    # Net operating income must be positive
    assert report.net_operating_income_egp > 0.0

    # Payback period
    assert 10.0 <= report.payback_period_years <= 30.0


def test_financing_cash_flow_comparisons():
    engine = RealEstateROIEngine(model_bundle=None)
    price = 6_000_000.0
    rent = 50_000.0

    # 1. Cash Purchase
    cash_plan = FinancingPlan(plan_type="cash")
    rep_cash = engine.evaluate_investment(price, rent, financing=cash_plan)
    assert rep_cash.monthly_installment_egp == 0.0
    assert rep_cash.annual_debt_service_egp == 0.0
    assert rep_cash.annual_net_cash_flow_egp == rep_cash.net_operating_income_egp

    # 2. Developer Installments (15% down, 7 years, 0% formal interest)
    dev_plan = FinancingPlan(plan_type="developer_installment", down_payment_pct=0.15, duration_years=7)
    rep_dev = engine.evaluate_investment(price, rent, financing=dev_plan)
    financed = price * 0.85
    expected_monthly = financed / (7 * 12)
    assert math.isclose(rep_dev.monthly_installment_egp, expected_monthly, abs_tol=1.0)
    assert rep_dev.initial_equity_invested_egp < rep_cash.initial_equity_invested_egp


def test_capital_appreciation_projections():
    engine = RealEstateROIEngine(model_bundle=None)
    price = 5_000_000.0
    rent = 40_000.0

    report = engine.evaluate_investment(price, rent)

    # 5-year scenarios
    app5 = report.appreciation_5yr
    assert app5["high_growth"] > app5["base_case"] > app5["conservative"] > price

    # 10-year scenarios
    app10 = report.appreciation_10yr
    assert app10["base_case"] > app5["base_case"]
    # 18% CAGR over 5 years is roughly 2.28x
    assert math.isclose(app5["base_case"], price * (1.18 ** 5), rel_tol=1e-3)


def test_investment_score_and_tiers():
    engine = RealEstateROIEngine(model_bundle=None)

    # Prime yield asset
    prime_rep = engine.evaluate_investment(
        purchase_price_egp=5_000_000.0,
        monthly_rent_egp=60_000.0, # 14.4% gross yield
        accessibility_score=95.0,
        is_compound=True,
    )
    assert prime_rep.investment_score >= 75.0
    assert "A" in prime_rep.investment_tier

    # Poor yield asset
    poor_rep = engine.evaluate_investment(
        purchase_price_egp=20_000_000.0,
        monthly_rent_egp=30_000.0, # 1.8% gross yield
        accessibility_score=20.0,
        is_compound=False,
    )
    assert poor_rep.investment_score < prime_rep.investment_score


def test_dual_valuation_fallback():
    engine = RealEstateROIEngine(model_bundle=None)
    buy, rent = engine.estimate_dual_valuation({"size": 200.0, "property_type": "Apartment"})
    assert buy > 0.0
    assert rent > 0.0
    assert buy > rent * 50 # Buy is orders of magnitude larger than monthly rent
