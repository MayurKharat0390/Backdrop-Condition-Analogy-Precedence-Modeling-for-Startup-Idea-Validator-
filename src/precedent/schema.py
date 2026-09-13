"""
schema.py: Startup Idea Representation & Validation Schema for BCAPM.

Defines the structured input contract for founders and investors evaluating
a new startup opportunity against historical precedents and economic hurdles.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class StartupIdea:
    """
    Validated representation of a startup opportunity entered for validation.
    Strictly aligns with all 28 historically grounded variables available in BCAPM datasets.
    """
    name: str
    market_category: str
    business_model: str = "B2B"          # "B2B", "B2C", or "Hybrid"
    country_code: str = "USA"            # ISO-3 alpha country code
    city: str = ""                       # Metropolitan ecosystem (e.g. San Francisco, London, Berlin)
    initial_funding_usd: float = 1_000_000.0  # Check size / seed capital
    funding_rounds_count: int = 1        # Funding round stage (1=Seed, 2=Series A, 3+=Growth)
    founder_count: int = 2
    female_founder_ratio: float = 0.0    # Ratio of female founders (0.0 to 1.0)
    worked_in_top_companies: bool = False # Tier-1 tech / FAANG alumni pedigree
    cax_cofounders: bool = False         # Top-tier accelerator backing (YC, Techstars, etc.)
    team_senior_leadership_size: int = 2 # Executive / VP / C-suite team depth
    repeat_investor_count: int = 0       # Follow-on tier-1 institutional VC backers
    is_ml_based: bool = False            # Proprietary AI / Machine Learning engine
    hn_sentiment_score: float = 0.65     # Public developer / early adopter sentiment (0.0 to 1.0)
    hn_public_engagement: float = 50.0   # Public buzz / discussion volume index (0 to 100)
    founding_year: int = 2026
    description: str = ""
    target_customer: str = "Enterprise"
    internet_penetration_at_founding: float = 85.0
    entrepreneurial_financing_index: float = 0.70
    government_support_index: float = 0.60
    tax_bureaucracy_index: float = 0.40
    
    def validate(self) -> None:
        """Validate input ranges and business logic."""
        if not self.name or not self.name.strip():
            raise ValueError("StartupIdea 'name' cannot be empty.")
        if not self.market_category or not self.market_category.strip():
            raise ValueError("StartupIdea 'market_category' must be specified.")
        if self.initial_funding_usd <= 0:
            raise ValueError("StartupIdea 'initial_funding_usd' must be strictly positive.")
        if self.founder_count < 1:
            raise ValueError("StartupIdea 'founder_count' must be at least 1.")
        if self.business_model.upper() not in ["B2B", "B2C", "HYBRID"]:
            raise ValueError("StartupIdea 'business_model' must be 'B2B', 'B2C', or 'Hybrid'.")
        if len(self.country_code.strip()) != 3:
            raise ValueError(f"StartupIdea 'country_code' must be a 3-letter ISO code (got '{self.country_code}').")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "market_category": self.market_category.lower().strip(),
            "business_model": self.business_model.upper().strip(),
            "country_code": self.country_code.upper().strip(),
            "city": self.city,
            "initial_funding_usd": float(self.initial_funding_usd),
            "funding_rounds_count": int(self.funding_rounds_count),
            "founder_count": int(self.founder_count),
            "female_founder_ratio": float(self.female_founder_ratio),
            "worked_in_top_companies": int(self.worked_in_top_companies),
            "cax_cofounders": int(self.cax_cofounders),
            "team_senior_leadership_size": int(self.team_senior_leadership_size),
            "repeat_investor_count": int(self.repeat_investor_count),
            "is_ml_based": int(self.is_ml_based),
            "hn_sentiment_score": float(self.hn_sentiment_score),
            "hn_public_engagement": float(self.hn_public_engagement),
            "founding_year": int(self.founding_year),
            "description": self.description,
            "target_customer": self.target_customer,
            "internet_penetration_at_founding": float(self.internet_penetration_at_founding),
            "entrepreneurial_financing_index": float(self.entrepreneurial_financing_index),
            "government_support_index": float(self.government_support_index),
            "tax_bureaucracy_index": float(self.tax_bureaucracy_index)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StartupIdea":
        """Instantiate from dictionary with type coercion and validation."""
        idea = cls(
            name=str(data.get("name", "Untitled Startup Idea")),
            market_category=str(data.get("market_category", "software")),
            business_model=str(data.get("business_model", "B2B")),
            country_code=str(data.get("country_code", "USA")),
            city=str(data.get("city", "")),
            initial_funding_usd=float(data.get("initial_funding_usd", 1_000_000.0)),
            funding_rounds_count=int(data.get("funding_rounds_count", 1)),
            founder_count=int(data.get("founder_count", 2)),
            female_founder_ratio=float(data.get("female_founder_ratio", 0.0)),
            worked_in_top_companies=bool(data.get("worked_in_top_companies", False)),
            cax_cofounders=bool(data.get("cax_cofounders", False)),
            team_senior_leadership_size=int(data.get("team_senior_leadership_size", 2)),
            repeat_investor_count=int(data.get("repeat_investor_count", 0)),
            is_ml_based=bool(data.get("is_ml_based", False)),
            hn_sentiment_score=float(data.get("hn_sentiment_score", 0.65)),
            hn_public_engagement=float(data.get("hn_public_engagement", 50.0)),
            founding_year=int(data.get("founding_year", 2026)),
            description=str(data.get("description", "")),
            target_customer=str(data.get("target_customer", "Enterprise")),
            internet_penetration_at_founding=float(data.get("internet_penetration_at_founding", 85.0)),
            entrepreneurial_financing_index=float(data.get("entrepreneurial_financing_index", 0.70)),
            government_support_index=float(data.get("government_support_index", 0.60)),
            tax_bureaucracy_index=float(data.get("tax_bureaucracy_index", 0.40))
        )
        idea.validate()
        return idea
