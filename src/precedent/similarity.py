"""
similarity.py: Multi-Attribute Weighted Similarity Engine for BCAPM Precedent Matching.

Calculates explicit, mathematically grounded similarity between a new startup idea
and historical startup records across decomposed firmographic dimensions:
- S_industry: Market sector alignment
- S_business_model: Go-to-market / customer model alignment (B2B vs B2C)
- S_geography: National / regional regulatory alignment
- S_capital: Seed / initial capital commitments (log-scale proximity)
- S_team: Founder team structure and pedigree

Returns normalized similarity [0, 1] alongside decomposed component contributions.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from src.precedent.schema import StartupIdea


class PrecedentSimilarityEngine:
    """Explicit multi-attribute similarity calculator."""

    def __init__(
        self,
        w_industry: float = 0.35,
        w_business_model: float = 0.25,
        w_geography: float = 0.15,
        w_capital: float = 0.15,
        w_team: float = 0.10
    ):
        weights = np.array([w_industry, w_business_model, w_geography, w_capital, w_team], dtype=float)
        self.weights = weights / np.sum(weights)
        self.w_ind, self.w_bm, self.w_geo, self.w_cap, self.w_team = self.weights

    def compute_industry_similarity(self, cat_idea: str, cat_precedent: str) -> float:
        """Score market sector similarity."""
        c1 = str(cat_idea).lower().strip()
        c2 = str(cat_precedent).lower().strip()
        if c1 == c2:
            return 1.0
        # Partial match on substring (e.g. 'health' in 'healthcare')
        if (c1 in c2) or (c2 in c1):
            return 0.75
        return 0.10

    def compute_business_model_similarity(self, bm_idea: str, b2c_b2b_flag: float) -> float:
        """
        Score B2B vs B2C alignment.
        In historical data: b2c_b2b_venture == 1 for B2B/tech enterprise, 0 for consumer.
        """
        idea_is_b2b = 1.0 if bm_idea.upper() in ["B2B", "ENTERPRISE"] else 0.0
        prec_is_b2b = float(b2c_b2b_flag) if not np.isnan(b2c_b2b_flag) else 0.5
        diff = abs(idea_is_b2b - prec_is_b2b)
        return float(1.0 - diff)

    def compute_geography_similarity(self, country_idea: str, country_precedent: str, city_idea: str = "", city_precedent: str = "") -> float:
        """Score geographic alignment at national and city hub levels."""
        c1 = str(country_idea).upper().strip()
        c2 = str(country_precedent).upper().strip()
        
        if c1 == c2:
            return 1.0
            
        # Tier-1 VC ecosystems
        tier1 = ["USA", "GBR", "CAN", "DEU", "FRA", "ISR"]
        if (c1 in tier1) and (c2 in tier1):
            return 0.60
        return 0.20

    def compute_capital_similarity(self, capital_idea: float, capital_precedent: float) -> float:
        """
        Score capital scale proximity on log-scale.
        Avoids penalizing large differences quadratically.
        """
        cap_i = max(10_000.0, float(capital_idea))
        cap_p = max(10_000.0, float(capital_precedent) if not np.isnan(capital_precedent) else 1_000_000.0)
        
        log_diff = abs(np.log10(cap_i) - np.log10(cap_p))
        # Decay: 1 decade difference (e.g. $1M vs $10M) yields exp(-0.8) ~ 0.45
        return float(np.exp(-0.8 * log_diff))

    def compute_team_similarity(self, idea: StartupIdea, row: pd.Series) -> float:
        """Score team structure, founder experience, accelerator pedigree, and syndication."""
        f_idea = idea.founder_count
        f_prec = float(row.get('founder_count', 2.0))
        size_sim = max(0.0, 1.0 - abs(f_idea - f_prec) / 5.0)
        
        top_idea = 1.0 if idea.worked_in_top_companies else 0.0
        top_prec = float(row.get('worked_in_top_companies', 0.0))
        pedigree_sim = 1.0 - abs(top_idea - top_prec)
        
        cax_idea = 1.0 if idea.cax_cofounders else 0.0
        cax_prec = float(row.get('cax_cofounders', 0.0))
        cax_sim = 1.0 - abs(cax_idea - cax_prec)
        
        repeat_idea = min(5, idea.repeat_investor_count)
        repeat_prec = min(5, float(row.get('repeat_investor_count', 0.0)))
        syndicate_sim = max(0.0, 1.0 - abs(repeat_idea - repeat_prec) / 5.0)
        
        return float(0.35 * size_sim + 0.35 * pedigree_sim + 0.15 * cax_sim + 0.15 * syndicate_sim)

    def score_startup(self, idea: StartupIdea, row: pd.Series) -> Tuple[float, Dict[str, float]]:
        """
        Compute total weighted micro similarity and decomposed attributions.
        """
        s_ind = self.compute_industry_similarity(idea.market_category, row.get('market_category', 'Other'))
        s_bm = self.compute_business_model_similarity(idea.business_model, row.get('b2c_b2b_venture', 1))
        s_geo = self.compute_geography_similarity(
            idea.country_code, 
            row.get('country_code', 'USA'),
            getattr(idea, 'city', ''),
            str(row.get('city', ''))
        )
        s_cap = self.compute_capital_similarity(idea.initial_funding_usd, row.get('funding_total_usd', 1_000_000.0))
        s_team = self.compute_team_similarity(idea, row)

        total_sim = (
            self.w_ind * s_ind +
            self.w_bm * s_bm +
            self.w_geo * s_geo +
            self.w_cap * s_cap +
            self.w_team * s_team
        )

        components = {
            "industry": float(s_ind),
            "business_model": float(s_bm),
            "geography": float(s_geo),
            "capital_scale": float(s_cap),
            "team": float(s_team)
        }

        return float(total_sim), components
