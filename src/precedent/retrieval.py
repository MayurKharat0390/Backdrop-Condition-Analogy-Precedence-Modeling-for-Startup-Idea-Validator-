"""
retrieval.py: Historical Precedent Retrieval Engine for BCAPM.

Searches the historical knowledge base of startups to find the Top-K most
analogous historical precedent ventures for a new startup idea, extracting their
realized outcomes (Acquired, IPO, Closed) and decomposed similarity components.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.config import CLEAN_PATH, DATA_RAW_DIR
from src.utils import get_logger, load_csv_file, canonicalize_name
from src.precedent.schema import StartupIdea
from src.precedent.similarity import PrecedentSimilarityEngine

logger = get_logger("Precedent_Retrieval")


class HistoricalPrecedentRetriever:
    """Retrieves analogous historical startups and extracts empirical precedent outcomes."""

    def __init__(
        self,
        clean_path: Optional[str] = None,
        similarity_engine: Optional[PrecedentSimilarityEngine] = None
    ):
        self.clean_path = clean_path or CLEAN_PATH
        self.sim_engine = similarity_engine or PrecedentSimilarityEngine()
        self._load_precedent_pool()

    def _load_precedent_pool(self) -> None:
        """Load and cache precedent startup pool with realized outcomes."""
        logger.info(f"Loading historical precedent pool from {self.clean_path}...")
        df_clean = load_csv_file(self.clean_path)
        
        # Load raw status from companies.csv to verify realized outcome if available
        raw_comp_file = DATA_RAW_DIR / "D1" / "companies.csv"
        try:
            if raw_comp_file.exists():
                df_raw = load_csv_file(raw_comp_file, usecols=['name', 'status'])
                df_raw['clean_name'] = df_raw['name'].apply(canonicalize_name)
                df_raw = df_raw.drop_duplicates(subset=['clean_name'])
                merged = pd.merge(df_clean, df_raw[['clean_name', 'status']], on='clean_name', how='left')
                merged['status'] = merged['status'].fillna('operating').str.lower()
                pool = merged[merged['status'].isin(['acquired', 'ipo', 'closed'])].copy().reset_index(drop=True)
                pool['is_exit'] = np.where(pool['status'].isin(['acquired', 'ipo']), 1, 0)
                self.precedent_pool = pool
            else:
                raise FileNotFoundError("companies.csv not found, using clean target_success fallback")
        except Exception as e:
            logger.warning(f"Using direct BCAPM_Clean target_success fallback ({e})")
            pool = df_clean.copy().reset_index(drop=True)
            pool['is_exit'] = pool['target_success'].astype(int)
            pool['status'] = np.where(pool['is_exit'] == 1, 'acquired', 'closed')
            self.precedent_pool = pool
        
        logger.info(f"Precedent pool initialized with {len(self.precedent_pool)} historical ventures with realized outcomes.")

    def retrieve_precedents(self, idea: StartupIdea, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve the top-K most analogous historical startups for a given idea.
        
        Returns:
            Dictionary containing:
            - idea_name
            - top_k_precedents: list of precedent dictionaries
            - empirical_exit_rate: float in [0, 1]
            - disclaimer: explicit scientific disclaimer
        """
        idea.validate()
        
        scores = []
        components_list = []
        
        # Vectorized / iteration score computation
        for idx, row in self.precedent_pool.iterrows():
            total_sim, components = self.sim_engine.score_startup(idea, row)
            scores.append(total_sim)
            components_list.append(components)
            
        scores_arr = np.array(scores, dtype=float)
        top_indices = np.argsort(scores_arr)[::-1][:top_k]
        
        precedents = []
        for idx in top_indices:
            row = self.precedent_pool.iloc[idx]
            raw_status = row['status'].title()
            outcome_label = "Liquidity Exit (Acquired/IPO)" if row['is_exit'] == 1 else "Closed / Liquidated (Failure)"
            
            precedents.append({
                "company_name": str(row.get('name', 'Unknown')),
                "similarity_score": round(float(scores_arr[idx]), 4),
                "similarity_percentage": f"{float(scores_arr[idx]) * 100:.1f}%",
                "market_category": str(row.get('market_category', 'N/A')),
                "country_code": str(row.get('country_code', 'N/A')),
                "founded_year": int(row.get('founded_year', 2005)),
                "funding_total_usd": float(row.get('funding_total_usd', 0.0)),
                "raw_status": raw_status,
                "realized_outcome": outcome_label,
                "is_exit": int(row['is_exit']),
                "component_attributions": components_list[idx]
            })
            
        exit_count = sum(p['is_exit'] for p in precedents)
        empirical_exit_rate = float(exit_count / top_k) if top_k > 0 else 0.0
        
        return {
            "query_idea": idea.name,
            "top_k": top_k,
            "empirical_precedent_exit_rate": empirical_exit_rate,
            "empirical_exit_percentage": f"{empirical_exit_rate * 100:.1f}%",
            "exits_count": exit_count,
            "total_retrieved": top_k,
            "precedents": precedents,
            "disclaimer": (
                "NOTE ON SCIENTIFIC INTERPRETATION: The empirical exit rate represents historical outcomes "
                "among the closest retrieved analogues. It serves as observational case-based evidence, "
                "not a standalone causal probability."
            )
        }
