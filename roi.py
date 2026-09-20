"""
Real Estate Investment ROI & Financial Modeling Engine for Egyptian Properties.

Computes Gross/Net Rental Yields, Cash-on-Cash Return, Developer Installment / Mortgage Cash Flow,
Multi-Year Capital Appreciation Projections, Payback Period, and Investment Opportunity Scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FinancialAssumptions:
    vacancy_rate: float = 0.08             # 8% annual vacancy (~1 month vacant)
    property_management_rate: float = 0.07 # 7% of collected rent
    maintenance_reserve_rate: float = 0.01 # 1% of property value annually
    property_tax_rate: float = 0.005       # 0.5% municipal & real estate tax
    closing_cost_rate: float = 0.025       # 2.5% registration / legal closing costs


@dataclass
class FinancingPlan:
    plan_type: str = "developer_installment" # "developer_installment" or "mortgage" or "cash"
    down_payment_pct: float = 0.15          # 15% down payment
    duration_years: int = 7                 # 7 years installments
    annual_interest_rate: float = 0.0       # 0% for standard developer off-plan; ~18-22% for bank mortgage


@dataclass
class ROIReport:
    purchase_price_egp: float
    monthly_rent_egp: float
    annual_gross_rent_egp: float
    effective_gross_income_egp: float
    operating_expenses_egp: float
    net_operating_income_egp: float
    gross_rental_yield_pct: float
    net_rental_yield_pct: float
    initial_equity_invested_egp: float
    annual_debt_service_egp: float
    monthly_installment_egp: float
    annual_net_cash_flow_egp: float
    cash_on_cash_return_pct: float
    payback_period_years: float
    investment_score: float
    investment_tier: str
    appreciation_5yr: Dict[str, float]
    appreciation_10yr: Dict[str, float]


def calculate_mortgage_payment(principal: float, annual_rate: float, years: int) -> float:
    """
    Computes fixed monthly mortgage payment using standard amortization.
    """
    if principal <= 0 or years <= 0:
        return 0.0
    if annual_rate <= 0:
        return principal / (years * 12.0)

    monthly_rate = annual_rate / 12.0
    n_payments = years * 12
    numerator = monthly_rate * ((1.0 + monthly_rate) ** n_payments)
    denominator = ((1.0 + monthly_rate) ** n_payments) - 1.0
    return principal * (numerator / denominator)


class RealEstateROIEngine:
    """
    Investment Analysis & ROI Modeling Engine for Property Finder Egypt.
    """

    def __init__(self, model_bundle: Optional[Dict[str, Any]] = None):
        self.bundle = model_bundle

    def estimate_dual_valuation(
        self,
        base_input: Dict[str, Any],
    ) -> Tuple[float, float]:
        """
        Estimates both Buy (purchase price) and Rent (monthly rental rate) using the model bundle.
        """
        if self.bundle is None or "model" not in self.bundle:
            # Fallback heuristic: 1 sqm ~ 45,000 EGP buy, rent ~ 0.7% of buy price monthly
            size = float(base_input.get("size", 150.0))
            buy_price = size * 45000.0
            rent_price = buy_price * 0.007
            return round(buy_price, 2), round(rent_price, 2)

        model = self.bundle["model"]
        feature_columns = self.bundle["feature_columns"]

        prop_type = base_input.get("property_type", "Apartment")
        is_commercial = "commercial" in str(base_input.get("category", "")).lower() or prop_type in [
            "Office Space", "Retail", "Commercial", "Warehouse"
        ]

        # 1. Buy row
        buy_cat = "commercial_buy" if is_commercial else "buy"
        buy_row = dict(base_input)
        buy_row["category"] = buy_cat
        buy_row["price_period"] = "sell"
        df_buy = pd.DataFrame([buy_row])
        for col in feature_columns:
            if col not in df_buy.columns:
                df_buy[col] = np.nan
        pred_buy_log = model.predict(df_buy[feature_columns])[0]
        buy_price = float(np.expm1(pred_buy_log))

        # 2. Rent row
        rent_cat = "commercial_rent" if is_commercial else "rent"
        rent_row = dict(base_input)
        rent_row["category"] = rent_cat
        rent_row["price_period"] = "monthly"
        df_rent = pd.DataFrame([rent_row])
        for col in feature_columns:
            if col not in df_rent.columns:
                df_rent[col] = np.nan
        pred_rent_log = model.predict(df_rent[feature_columns])[0]
        rent_price = float(np.expm1(pred_rent_log))

        return max(10000.0, round(buy_price, 2)), max(500.0, round(rent_price, 2))

    def evaluate_investment(
        self,
        purchase_price_egp: float,
        monthly_rent_egp: float,
        assumptions: Optional[FinancialAssumptions] = None,
        financing: Optional[FinancingPlan] = None,
        accessibility_score: float = 70.0,
        is_compound: bool = False,
    ) -> ROIReport:
        """
        Executes full investment financial analysis.
        """
        assump = assumptions or FinancialAssumptions()
        fin = financing or FinancingPlan()

        # Gross & Net Rental Revenue
        annual_gross_rent = monthly_rent_egp * 12.0
        effective_gross_income = annual_gross_rent * (1.0 - assump.vacancy_rate)

        mgmt_expense = effective_gross_income * assump.property_management_rate
        maint_expense = purchase_price_egp * assump.maintenance_reserve_rate
        tax_expense = purchase_price_egp * assump.property_tax_rate
        total_operating_expenses = mgmt_expense + maint_expense + tax_expense

        net_operating_income = effective_gross_income - total_operating_expenses

        # Yields
        gross_yield_pct = (annual_gross_rent / purchase_price_egp) * 100.0
        net_yield_pct = (net_operating_income / purchase_price_egp) * 100.0

        # Financing & Initial Capital
        down_payment = purchase_price_egp * fin.down_payment_pct
        closing_costs = purchase_price_egp * assump.closing_cost_rate
        initial_equity = down_payment + closing_costs

        financed_amount = purchase_price_egp - down_payment
        if fin.plan_type == "cash":
            monthly_installment = 0.0
            annual_debt_service = 0.0
            initial_equity = purchase_price_egp + closing_costs
        elif fin.plan_type == "mortgage":
            monthly_installment = calculate_mortgage_payment(
                financed_amount, fin.annual_interest_rate, fin.duration_years
            )
            annual_debt_service = monthly_installment * 12.0
        else: # Developer Installment (0% interest, straight distribution)
            total_payments = max(1, fin.duration_years * 12)
            monthly_installment = financed_amount / total_payments
            annual_debt_service = monthly_installment * 12.0

        annual_net_cash_flow = net_operating_income - annual_debt_service
        cash_on_cash = (annual_net_cash_flow / initial_equity) * 100.0 if initial_equity > 0 else 0.0

        # Multi-Year Capital Appreciation Projections (12% Conservative, 18% Base, 25% High Inflation)
        cagr_scenarios = {"conservative": 0.12, "base_case": 0.18, "high_growth": 0.25}
        appreciation_5yr = {
            name: round(purchase_price_egp * ((1.0 + rate) ** 5), 2)
            for name, rate in cagr_scenarios.items()
        }
        appreciation_10yr = {
            name: round(purchase_price_egp * ((1.0 + rate) ** 10), 2)
            for name, rate in cagr_scenarios.items()
        }

        # Payback Period (Years for cumulative net operating income to equal purchase price)
        if net_operating_income > 0:
            payback_years = round(purchase_price_egp / net_operating_income, 1)
        else:
            payback_years = 99.9

        # Investment Opportunity Score (1 to 100)
        # 1. Yield Score (35 pts max): 10% net yield gets ~35 pts
        yield_score = min(35.0, max(0.0, net_yield_pct * 3.5))

        # 2. Location & Compound Growth Score (25 pts max)
        loc_score = (accessibility_score / 100.0) * 15.0 + (10.0 if is_compound else 5.0)

        # 3. Cash flow resilience (25 pts max)
        cash_score = 25.0 if annual_net_cash_flow >= 0 else max(5.0, 25.0 + (annual_net_cash_flow / initial_equity) * 20.0)

        # 4. Gross yield benchmark (15 pts max): >= 8% is 15 pts
        gross_benchmark = min(15.0, max(0.0, gross_yield_pct * 1.8))

        investment_score = round(float(np.clip(yield_score + loc_score + cash_score + gross_benchmark, 15.0, 99.0)), 1)

        if investment_score >= 85.0:
            tier = "A+ (Prime Institutional Asset)"
        elif investment_score >= 72.0:
            tier = "A (Strong Growth & Solid Yield)"
        elif investment_score >= 58.0:
            tier = "B (Balanced Market Asset)"
        elif investment_score >= 45.0:
            tier = "C (Income Lagging / Speculative)"
        else:
            tier = "High Capital Risk"

        return ROIReport(
            purchase_price_egp=round(purchase_price_egp, 2),
            monthly_rent_egp=round(monthly_rent_egp, 2),
            annual_gross_rent_egp=round(annual_gross_rent, 2),
            effective_gross_income_egp=round(effective_gross_income, 2),
            operating_expenses_egp=round(total_operating_expenses, 2),
            net_operating_income_egp=round(net_operating_income, 2),
            gross_rental_yield_pct=round(gross_yield_pct, 2),
            net_rental_yield_pct=round(net_yield_pct, 2),
            initial_equity_invested_egp=round(initial_equity, 2),
            annual_debt_service_egp=round(annual_debt_service, 2),
            monthly_installment_egp=round(monthly_installment, 2),
            annual_net_cash_flow_egp=round(annual_net_cash_flow, 2),
            cash_on_cash_return_pct=round(cash_on_cash, 2),
            payback_period_years=payback_years,
            investment_score=investment_score,
            investment_tier=tier,
            appreciation_5yr=appreciation_5yr,
            appreciation_10yr=appreciation_10yr,
        )
