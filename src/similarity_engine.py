"""
similarity_engine.py: Export wrapper for BCAPMSimilarityEngine.
Allows clean Python import: from src.similarity_engine import BCAPMSimilarityEngine
"""
import importlib
_mod = importlib.import_module("src.10_similarity_engine")
BCAPMSimilarityEngine = _mod.BCAPMSimilarityEngine

__all__ = ["BCAPMSimilarityEngine"]
