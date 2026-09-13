"""
BCAPM Backdrop & Condition Analogy Package.
"""

from src.backdrop.condition_vector import BackdropConditionVector, get_current_backdrop_vector
from src.backdrop.analogy import BackdropAnalogyEngine
from src.backdrop.friction import ConditionFrictionAnalyzer

__all__ = [
    "BackdropConditionVector",
    "get_current_backdrop_vector",
    "BackdropAnalogyEngine",
    "ConditionFrictionAnalyzer"
]
