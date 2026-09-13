"""
test_decision.py: Unit tests for economic utility, breakeven hurdle, and Top-K selection.
"""

import pytest
import numpy as np
from src.decision.topk import rank_and_select_top_k
from src.economic_utility import calculate_optimal_threshold, calculate_emv


def test_breakeven_threshold():
    p_star = calculate_optimal_threshold(investment_amount=1.0, success_payoff=20.0)
    assert np.isclose(p_star, 0.05)
    
    p_star_high = calculate_optimal_threshold(investment_amount=2.0, success_payoff=20.0)
    assert np.isclose(p_star_high, 0.10)


def test_emv_calculation():
    emv_win = calculate_emv(p_success=0.50, investment_amount=1.0, success_payoff=20.0)
    assert np.isclose(emv_win, 9.0) # 0.5 * 20 - 1 = 9
    
    emv_breakeven = calculate_emv(p_success=0.05, investment_amount=1.0, success_payoff=20.0)
    assert np.isclose(emv_breakeven, 0.0)


def test_top_k_ranking():
    probs = np.array([0.10, 0.80, 0.40, 0.90, 0.05])
    labels = np.array([0, 1, 0, 1, 0])
    
    res = rank_and_select_top_k(probs, ground_truth=labels, capacity_k=2)
    assert res["capacity_k"] == 2
    # Indices 3 (0.90) and 1 (0.80) should be selected
    assert set(res["selected_indices"]) == {1, 3}
    assert res["precision_at_k"] == 1.0 # Both are winners
