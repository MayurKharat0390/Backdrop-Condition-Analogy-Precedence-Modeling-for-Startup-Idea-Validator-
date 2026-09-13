"""
BCAPM Economic Decision Package.
"""

from src.economic_utility import (
    calculate_optimal_threshold,
    calculate_emv,
    make_investment_decision,
    explain_economic_decision
)
from src.decision.topk import rank_and_select_top_k, allocate_by_capital_budget
from src.portfolio_simulation import simulate_portfolio
from src.threshold_optimization import sweep_thresholds, run_sensitivity_analysis

__all__ = [
    "calculate_optimal_threshold",
    "calculate_emv",
    "make_investment_decision",
    "explain_economic_decision",
    "rank_and_select_top_k",
    "allocate_by_capital_budget",
    "simulate_portfolio",
    "sweep_thresholds",
    "run_sensitivity_analysis"
]

