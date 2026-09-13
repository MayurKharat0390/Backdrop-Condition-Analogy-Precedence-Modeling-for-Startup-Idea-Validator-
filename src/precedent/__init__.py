"""
BCAPM Precedent Retrieval & Analogy Package.
"""

from src.precedent.schema import StartupIdea
from src.precedent.similarity import PrecedentSimilarityEngine
from src.precedent.retrieval import HistoricalPrecedentRetriever
from src.precedent.explanation import explain_precedent_match, generate_precedent_summary

__all__ = [
    "StartupIdea",
    "PrecedentSimilarityEngine",
    "HistoricalPrecedentRetriever",
    "explain_precedent_match",
    "generate_precedent_summary"
]
