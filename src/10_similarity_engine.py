"""
10_similarity_engine.py: Backdrop-Conditioned Analogy Precedent (BCAPM) Similarity Engine.

Purpose:
- Load preprocessed precedent dataset (BCAPM_Preprocessed.csv) and metadata mapping.
- Compute cosine similarity across high-dimensional feature representations.
- Find top-k similar historical startup precedent analogs.
- Return similarity score, matched startup profiles, predicted outcome, and explanatory reasoning.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Union
from sklearn.metrics.pairwise import cosine_similarity
import importlib
preprocess_data = importlib.import_module('src.06_preprocessing').preprocess_data

from src.config import PREPROCESSED_PATH, DATA_PROCESSED_DIR, TARGET_COL
from src.utils import get_logger, canonicalize_name, load_csv_file

logger = get_logger("10_Similarity_Engine")

class BCAPMSimilarityEngine:
    """BCAPM Analogy Precedent Matching Engine using vector cosine similarity."""
    
    def __init__(self, preprocessed_path: Union[str, Path] = None):
        """Initialize analogy engine and load precedent knowledge base."""
        if preprocessed_path is None:
            preprocessed_path = PREPROCESSED_PATH
        else:
            preprocessed_path = Path(preprocessed_path)
            
        meta_path = DATA_PROCESSED_DIR / "BCAPM_Metadata.csv"
        
        if not preprocessed_path.exists() or not meta_path.exists():
            logger.info(f"Generating preprocessed dataset and metadata...")
            self.df = preprocess_data()
            meta_df = pd.read_csv(meta_path)
        else:
            self.df = load_csv_file(preprocessed_path)
            meta_df = load_csv_file(meta_path)
            
        logger.info(f"BCAPM Analogy Engine loaded {len(self.df)} precedent startup records.")
        
        self.feature_cols = [c for c in self.df.columns if c != TARGET_COL]
        self.matrix = self.df[self.feature_cols].fillna(0.0).values
        
        self.clean_names = meta_df['clean_name'].astype(str).tolist()
        self.display_names = meta_df['name'].astype(str).tolist()
        self.targets = self.df[TARGET_COL].tolist()

    def find_analogs_by_name(self, company_name: str, top_k: int = 5) -> Dict:
        """Find top-k precedent startup analogs for a given company name in the database."""
        c_name = canonicalize_name(company_name)
        if c_name not in self.clean_names:
            logger.warning(f"Company '{company_name}' (clean: '{c_name}') not found in precedent dataset.")
            matches = [i for i, name in enumerate(self.clean_names) if c_name in name or name in c_name]
            if not matches:
                return {"error": f"Company '{company_name}' not found in precedent pool."}
            target_idx = matches[0]
        else:
            target_idx = self.clean_names.index(c_name)
            
        query_vector = self.matrix[target_idx].reshape(1, -1)
        sim_scores = cosine_similarity(query_vector, self.matrix)[0]
        
        sorted_indices = np.argsort(sim_scores)[::-1]
        precedent_indices = [idx for idx in sorted_indices if idx != target_idx][:top_k]
        
        matched_precedents = []
        for idx in precedent_indices:
            matched_precedents.append({
                "Company_Name": self.display_names[idx],
                "Similarity_Score": float(sim_scores[idx]),
                "Historical_Outcome": "Acquired / Operating (Success)" if self.targets[idx] == 1 else "Closed / Failed (Unsuccessful)"
            })
            
        top_targets = [self.targets[idx] for idx in precedent_indices]
        success_probability = sum(top_targets) / float(len(top_targets))
        
        explanation = (
            f"Target Startup '{self.display_names[target_idx]}' matched {top_k} historical precedent analogs. "
            f"{sum(top_targets)} out of {top_k} similar precedents achieved successful exit outcomes. "
            f"Precedent-conditioned success probability: {success_probability * 100:.1f}%."
        )
        
        return {
            "Query_Company": self.display_names[target_idx],
            "Precedent_Success_Probability": success_probability,
            "Top_K_Analogs": matched_precedents,
            "Explanation": explanation
        }

def main():
    parser = argparse.ArgumentParser(description="Find top-k startup precedent analogs using BCAPM engine.")
    parser.add_argument("--company", type=str, default="Dropbox", help="Name of company to query analogs for.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of precedent analogs to retrieve.")
    args = parser.parse_args()
    
    engine = BCAPMSimilarityEngine()
    results = engine.find_analogs_by_name(args.company, top_k=args.top_k)
    
    print("\n==========================================================")
    print(f"  BCAPM ANALOGY PRECEDENT REPORT: {args.company}")
    print("==========================================================")
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Query Startup: {results['Query_Company']}")
        print(f"Predicted Success Probability: {results['Precedent_Success_Probability']*100:.1f}%\n")
        print("Top Matched Precedent Analogs:")
        for idx, item in enumerate(results['Top_K_Analogs'], 1):
            print(f"  {idx}. {item['Company_Name']} | Similarity: {item['Similarity_Score']:.4f} | Outcome: {item['Historical_Outcome']}")
        print("\nExplanatory Insight:")
        print(f"  {results['Explanation']}")
    print("==========================================================\n")

if __name__ == "__main__":
    main()
