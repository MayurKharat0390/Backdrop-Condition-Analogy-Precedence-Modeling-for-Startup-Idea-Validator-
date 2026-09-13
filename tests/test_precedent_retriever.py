"""
test_precedent_retriever.py: Integration test for HistoricalPrecedentRetriever and Validator pipeline.
"""

import pytest
from src.precedent.schema import StartupIdea
from src.precedent.retrieval import HistoricalPrecedentRetriever
from src.validator.pipeline import BCAPMValidator


def test_retriever_returns_top_k():
    idea = StartupIdea(
        name="FinFlow Analytics",
        market_category="software",
        business_model="B2B",
        country_code="USA",
        initial_funding_usd=2_000_000.0,
        founder_count=2
    )
    retriever = HistoricalPrecedentRetriever()
    res = retriever.retrieve_precedents(idea, top_k=3)
    
    assert res["query_idea"] == "FinFlow Analytics"
    assert res["top_k"] == 3
    assert len(res["precedents"]) == 3
    assert 0.0 <= res["empirical_precedent_exit_rate"] <= 1.0
    for p in res["precedents"]:
        assert "company_name" in p
        assert 0.0 <= p["similarity_score"] <= 1.0
        assert p["is_exit"] in [0, 1]


def test_validator_generates_verdict():
    idea = StartupIdea(
        name="LogiHealth SaaS",
        market_category="health",
        business_model="B2B",
        country_code="USA",
        initial_funding_usd=1_500_000.0,
        founder_count=3,
        worked_in_top_companies=True
    )
    validator = BCAPMValidator(top_k=3)
    verdict = validator.validate_idea(idea, check_size_m=1.5, target_exit_m=25.0)
    
    assert verdict.verdict in ["INVEST", "REVIEW", "REJECT"]
    assert 0.0 <= verdict.calibrated_probability <= 1.0
    assert verdict.economic_threshold > 0.0
    assert len(verdict.scenario_sensitivities) == 3
    assert "conservative" in verdict.scenario_sensitivities
    assert "base" in verdict.scenario_sensitivities
    assert "aggressive" in verdict.scenario_sensitivities
