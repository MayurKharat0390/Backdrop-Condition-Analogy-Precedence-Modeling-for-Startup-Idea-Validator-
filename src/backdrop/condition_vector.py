"""
condition_vector.py: Macroeconomic & Environmental Backdrop Vector Extractor.

Extracts normalized backdrop condition vectors capturing the operating environment
of startups:
- Internet penetration at founding
- Country Human Development Index (HDI)
- Entrepreneurial financing index
- Government support & policy index
- Tax & bureaucracy friction index
- Historical macroeconomic failure rate
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class BackdropConditionVector:
    """Represents the macroeconomic and regulatory environment of a venture."""
    country_code: str
    year: int
    internet_penetration_pct: float = 85.0
    country_hdi: float = 0.92
    entrepreneurial_financing_index: float = 6.5
    government_support_index: float = 6.0
    tax_bureaucracy_index: float = 4.5
    macro_failure_rate: float = 0.20

    def to_array(self) -> np.ndarray:
        """Convert continuous backdrop dimensions to array."""
        return np.array([
            self.internet_penetration_pct / 100.0,
            self.country_hdi,
            self.entrepreneurial_financing_index / 10.0,
            self.government_support_index / 10.0,
            (10.0 - self.tax_bureaucracy_index) / 10.0,  # Invert so higher is better
            1.0 - self.macro_failure_rate               # Invert so higher is lower failure
        ], dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "country_code": self.country_code,
            "year": self.year,
            "internet_penetration_pct": self.internet_penetration_pct,
            "country_hdi": self.country_hdi,
            "entrepreneurial_financing_index": self.entrepreneurial_financing_index,
            "government_support_index": self.government_support_index,
            "tax_bureaucracy_index": self.tax_bureaucracy_index,
            "macro_failure_rate": self.macro_failure_rate
        }


def get_current_backdrop_vector(country_code: str = "USA", year: int = 2026) -> BackdropConditionVector:
    """
    Returns realistic macroeconomic and tech backdrop values for modern ventures.
    Grounded in current World Bank and GEM (Global Entrepreneurship Monitor) benchmarks.
    """
    country = str(country_code).upper().strip()
    
    # Defaults for Tier-1 VC ecosystems
    if country in ["USA", "US"]:
        return BackdropConditionVector(
            country_code="USA",
            year=year,
            internet_penetration_pct=92.0,
            country_hdi=0.93,
            entrepreneurial_financing_index=7.2,
            government_support_index=6.4,
            tax_bureaucracy_index=4.2,
            macro_failure_rate=0.18
        )
    elif country in ["GBR", "UK"]:
        return BackdropConditionVector(
            country_code="GBR",
            year=year,
            internet_penetration_pct=95.0,
            country_hdi=0.92,
            entrepreneurial_financing_index=6.8,
            government_support_index=6.2,
            tax_bureaucracy_index=4.0,
            macro_failure_rate=0.19
        )
    elif country in ["IND", "IN"]:
        return BackdropConditionVector(
            country_code="IND",
            year=year,
            internet_penetration_pct=55.0,
            country_hdi=0.64,
            entrepreneurial_financing_index=6.0,
            government_support_index=5.8,
            tax_bureaucracy_index=5.5,
            macro_failure_rate=0.24
        )
    else:
        return BackdropConditionVector(
            country_code=country,
            year=year,
            internet_penetration_pct=75.0,
            country_hdi=0.80,
            entrepreneurial_financing_index=5.5,
            government_support_index=5.0,
            tax_bureaucracy_index=5.0,
            macro_failure_rate=0.22
        )
