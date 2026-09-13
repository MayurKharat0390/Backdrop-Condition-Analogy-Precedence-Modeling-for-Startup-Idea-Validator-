"""
test_schema.py: Unit tests for StartupIdea schema validation.
"""

import pytest
from src.precedent.schema import StartupIdea


def test_valid_startup_idea():
    idea = StartupIdea(
        name="Test Venture",
        market_category="software",
        business_model="B2B",
        country_code="USA",
        initial_funding_usd=1_000_000.0,
        founder_count=2
    )
    idea.validate()
    d = idea.to_dict()
    assert d["name"] == "Test Venture"
    assert d["market_category"] == "software"
    assert d["business_model"] == "B2B"
    assert d["initial_funding_usd"] == 1_000_000.0


def test_invalid_business_model():
    with pytest.raises(ValueError, match="business_model"):
        idea = StartupIdea(
            name="Bad Model",
            market_category="software",
            business_model="INVALID"
        )
        idea.validate()


def test_negative_capital():
    with pytest.raises(ValueError, match="initial_funding_usd"):
        idea = StartupIdea(
            name="Negative Capital",
            market_category="software",
            initial_funding_usd=-500.0
        )
        idea.validate()


def test_empty_name():
    with pytest.raises(ValueError, match="name"):
        idea = StartupIdea(
            name="",
            market_category="software"
        )
        idea.validate()
