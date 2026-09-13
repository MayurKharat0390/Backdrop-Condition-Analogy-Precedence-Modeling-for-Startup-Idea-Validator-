"""
analogy.py: Macroeconomic & Backdrop Analogy Evaluator.

Evaluates contextual similarity between a new idea's operating climate and
the historical environment in which precedent startups were founded and operated.
Explicitly decouples firmographic venture similarity from environmental backdrop similarity.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.backdrop.condition_vector import BackdropConditionVector, get_current_backdrop_vector
from src.precedent.schema import StartupIdea


class BackdropAnalogyEngine:
    """Evaluates contextual macro condition alignment between an idea and historical precedents."""

    def __init__(self):
        pass

    def compute_backdrop_similarity(
        self,
        idea_vector: BackdropConditionVector,
        row: pd.Series
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute normalized backdrop similarity between the idea and a historical startup record.
        """
        v_idea = idea_vector.to_array()
        
        # Extract row backdrop
        net_prec = float(row.get('internet_penetration_at_founding', 50.0)) / 100.0
        hdi_prec = float(row.get('country_hdi', 0.80))
        fin_prec = float(row.get('entrepreneurial_financing_index', 5.0)) / 10.0
        gov_prec = float(row.get('government_support_index', 5.0)) / 10.0
        tax_prec = (10.0 - float(row.get('tax_bureaucracy_index', 5.0))) / 10.0
        fail_prec = 1.0 - float(row.get('macro_failure_rate_at_founding', 0.20))
        
        v_prec = np.array([net_prec, hdi_prec, fin_prec, gov_prec, tax_prec, fail_prec], dtype=float)
        
        # Euclidean distance normalized to similarity in [0, 1]
        dist = np.linalg.norm(v_idea - v_prec)
        max_dist = np.sqrt(len(v_idea))  # Theoretical upper bound
        sim = max(0.0, 1.0 - (dist / max_dist))
        
        # Era distance
        prec_year = int(row.get('founded_year', 2005))
        year_diff = abs(idea_vector.year - prec_year)
        
        breakdown = {
            "backdrop_similarity": float(sim),
            "era_difference_years": int(year_diff),
            "precedent_founding_year": prec_year,
            "precedent_internet_penetration": float(net_prec * 100.0),
            "precedent_country_hdi": float(hdi_prec)
        }
        
        return float(sim), breakdown

    def evaluate_precedents_backdrop(
        self,
        idea: StartupIdea,
        precedents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate backdrop conditions across the retrieved precedent pool.
        """
        idea_backdrop = get_current_backdrop_vector(idea.country_code, idea.founding_year)
        
        backdrop_scores = []
        era_diffs = []
        
        for prec in precedents:
            # Reconstruct mock row from precedent dictionary
            mock_row = pd.Series({
                "country_code": prec.get("country_code", "USA"),
                "founded_year": prec.get("founded_year", 2005),
                "internet_penetration_at_founding": 75.0,
                "country_hdi": 0.90,
                "entrepreneurial_financing_index": 6.5,
                "government_support_index": 6.0,
                "tax_bureaucracy_index": 4.5,
                "macro_failure_rate_at_founding": 0.20
            })
            sim, breakdown = self.compute_backdrop_similarity(idea_backdrop, mock_row)
            prec["backdrop_similarity"] = sim
            prec["era_difference_years"] = breakdown["era_difference_years"]
            backdrop_scores.append(sim)
            era_diffs.append(breakdown["era_difference_years"])
            
        mean_backdrop = float(np.mean(backdrop_scores)) if backdrop_scores else 0.5
        mean_era_diff = float(np.mean(era_diffs)) if era_diffs else 10.0
        
        alignment_label = "High" if mean_backdrop >= 0.80 else ("Medium" if mean_backdrop >= 0.60 else "Low")
        
        return {
            "mean_backdrop_similarity": round(mean_backdrop, 4),
            "mean_backdrop_percentage": f"{mean_backdrop * 100:.1f}%",
            "backdrop_alignment": alignment_label,
            "mean_era_difference_years": round(mean_era_diff, 1),
            "current_context": idea_backdrop.to_dict()
        }
