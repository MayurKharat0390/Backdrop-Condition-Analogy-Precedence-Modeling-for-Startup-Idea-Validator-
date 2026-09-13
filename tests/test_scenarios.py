"""
test_scenarios.py: Unit tests for scenario-based Monte Carlo sampling.
"""

import pytest
import numpy as np
from src.simulation.scenarios import sample_scenario_multiples, evaluate_scenario_sensitivity, SCENARIO_PROFILES


def test_scenario_sampling_ranges():
    # Conservative should have lower multiples
    cons_mults = sample_scenario_multiples(100, scenario_name="conservative", random_state=42)
    assert len(cons_mults) == 100
    assert np.all(cons_mults >= 1.0)
    
    # Aggressive should have higher average multiple than Conservative
    agg_mults = sample_scenario_multiples(100, scenario_name="aggressive", random_state=42)
    assert np.mean(agg_mults) > np.mean(cons_mults)


def test_scenario_sensitivity_profiles():
    sens = evaluate_scenario_sensitivity(p_success=0.60, investment_amount_m=1.0, n_simulations=100)
    assert "conservative" in sens
    assert "base" in sens
    assert "aggressive" in sens
    assert sens["aggressive"]["mean_roi"] > sens["conservative"]["mean_roi"]
