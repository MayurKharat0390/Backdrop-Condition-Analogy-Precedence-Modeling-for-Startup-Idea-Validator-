"""
test_similarity.py: Unit tests for PrecedentSimilarityEngine.
"""

import pytest
import pandas as pd
import numpy as np
from src.precedent.schema import StartupIdea
from src.precedent.similarity import PrecedentSimilarityEngine


def test_similarity_bounds_and_components():
    engine = PrecedentSimilarityEngine()
    idea = StartupIdea(
        name="FinTech Hub",
        market_category="finance",
        business_model="B2B",
        country_code="USA",
        initial_funding_usd=2_000_000.0,
        founder_count=3,
        worked_in_top_companies=True
    )
    
    mock_row = pd.Series({
        "market_category": "finance",
        "b2c_b2b_venture": 1,
        "country_code": "USA",
        "funding_total_usd": 2_000_000.0,
        "founder_count": 3,
        "worked_in_top_companies": 1
    })
    
    score, components = engine.score_startup(idea, mock_row)
    
    # Perfect match should have high similarity
    assert 0.0 <= score <= 1.0
    assert score >= 0.90
    assert components["industry"] == 1.0
    assert components["business_model"] == 1.0
    assert components["geography"] == 1.0
    assert components["capital_scale"] == 1.0


def test_weights_sum_to_one():
    engine = PrecedentSimilarityEngine(w_industry=10, w_business_model=10, w_geography=10, w_capital=10, w_team=10)
    assert np.isclose(np.sum(engine.weights), 1.0)
