# BCAPM Data Pipeline Architectural Design

**Project**: Backdrop-Conditioned Analogy Precedent Modeling (BCAPM) for Startup Success Prediction

## Overview & Iterative Evolution

The BCAPM data pipeline recreates the step-by-step engineering progression of transforming heterogeneous, multi-source raw startup and macro-economic datasets into a ML-ready, backdrop-conditioned feature tensor.

```
+-----------------------------------------------------------------------------------+
|                                  RAW DATASETS                                     |
| Crunchbase (D1), Founder Profiles (D2), YC & HN Sentiment (D5, D6), Operational   |
| Metrics (D8, D9), Funding/Geography (D3, D4), World Bank Indicators (Root & D10)   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                              STAGE 1: BCAPM_Master_V1                             |
| Entity Resolution & Micro-Level Integration (Firmographics + Founders + HN        |
| Sentiment + Operational Metrics)                                                  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                              STAGE 2: BCAPM_Master_V2                             |
| Environmental Conditioning & Macro Backdrop Merging (World Bank Internet % +      |
| National Economy HDI/Policy + Sector/Year Failure Baselines)                      |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                              STAGE 3: BCAPM_Clean                                 |
| Deduplication, Multicollinearity Filter, Schema Normalization, Robust Missing     |
| Value Imputation                                                                  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                           STAGE 4: BCAPM_Preprocessed                             |
| Feature Engineering (Capital Efficiency, Media/Funding Ratios, Backdrop Growth)   |
| + Categorical Encoding + Continuous Robust Scaling -> Ready for Analogy Matching  |
+-----------------------------------------------------------------------------------+
```

---

## Detailed Stage-by-Stage Specifications

### Stage 1: Raw Datasets -> `BCAPM_Master_V1`

#### 1. Transformations Performed
- Entity name canonicalization (lowercasing, whitespace trimming, stripping punctuation and suffix tags `Inc`, `LLC`, `Corp`).
- Entity resolution across Crunchbase permalinks (`D1/companies.csv`), YC company names (`D5/CleanStartupsFull3.csv`), and CAX records (`D9/CAX_Startup_Data.csv`).
- Aggregation of founder-level records (`D2/founder_V0.3_founder.csv`, `D5/Founders.csv`) by company to produce team-level summary statistics.
- Text sentiment signal aggregation from Hacker News posts and comments (`D5/CompleteSet3.csv`).

#### 2. Merged Datasets
- `D1/companies.csv` (Base firmographics)
- `D1/rounds.csv` (Funding round dynamics)
- `D1/acquisitions.csv` (Exit outcome details)
- `D2/founder_V0.3_founder.csv` (Founder track record)
- `D5/Founders.csv` (YC team composition)
- `D5/CleanStartupsFull3.csv` (YC batch metrics)
- `D5/CompleteSet3.csv` (HN sentiment scores)
- `D8/Startup_Data.csv` / `D9/CAX_Startup_Data.csv` (116 operational features)
- `investments_VC.csv` (VC round distribution)

#### 3. Columns Retained
- `company_name`, `market_category`, `country_code`, `state_code`, `city`, `founded_year`, `funding_total_usd`, `funding_rounds_count`
- `founder_count`, `founder_news_articles_avg`, `founder_prior_orgs_max`, `female_founder_ratio`
- `hn_sentiment_score`, `hn_public_engagement`
- `team_senior_leadership_size`, `repeat_investor_count`, `is_ml_based`, `is_b2b`
- `target_success` (Binary target: 1 = Acquired/Operating/Successful, 0 = Closed/Dead/Failed)

#### 4. Columns Removed
- Raw HTML URLs, logo file paths, unparsed JSON strings, redundant raw text bodies, index columns (`Unnamed: 0`).

#### 5. Engineered Features
- `founder_count`: Total founders associated with company entity.
- `hn_public_engagement`: `TitlePoints` + `TopCommentPoints`.
- `company_age_years`: Current processing year (2026) minus `founded_year`.

#### 6. Necessity
Fragmented datasets capture isolated aspects of a startup (founders, funding, community sentiment, operational tech stack). Stage 1 aligns these micro-level signals onto a single entity row.

---

### Stage 2: `BCAPM_Master_V1` -> `BCAPM_Master_V2`

#### 1. Transformations Performed
- Unpivoted (melted) World Bank annual Internet Penetration time series (`API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv`) from wide format (`1960`..`2023`) to long format (`Country Code`, `Year`, `Internet_Penetration_Pct`).
- Joined country macroeconomic backdrop metrics from `D10/Final - Copy.csv` (`HDI`, `Financing for entrepreneurs`, `Governmental support and policies`, `Taxes and bureaucracy`).
- Attached annual historical macro failure rate (`D10/Startup_failure.csv`) matched by `founded_year`.
- Attached industry baseline success probability (`D10/startup_indus.csv`) matched by `market_category`.

#### 2. Merged Datasets
- `BCAPM_Master_V1.csv`
- `API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv` (World Bank Internet %)
- `Metadata_Country_API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv` (Income groups)
- `D10/Final - Copy.csv` (National HDI & policy scores)
- `D10/Startup_failure.csv` (Annual macro failure rate)
- `D10/startup_indus.csv` (Sector success baseline)

#### 3. Columns Retained
- All Stage 1 features plus:
- `internet_penetration_at_founding`, `country_hdi`, `entrepreneurial_financing_index`, `government_support_index`, `tax_bureaucracy_index`, `macro_failure_rate_at_founding`, `industry_baseline_success_rate`

#### 4. Columns Removed
- Yearly raw time series columns (`1960`..`2023`), country metadata special note text fields.

#### 5. Engineered Features
- `internet_penetration_at_founding`: Internet penetration percentage in the company's founding country in its specific founding year.
- `macro_backdrop_score`: Composite average of national `HDI`, `entrepreneurial_financing_index`, and inverse `tax_bureaucracy_index`.

#### 6. Necessity
Startups founded during different eras (e.g. 1999 Dot-Com boom vs. 2008 Financial Crisis vs. 2020 Remote Work transition) face drastically different external conditions. Conditioning precedent search on the macroeconomic backdrop prevents false analogies between non-comparable eras.

---

### Stage 3: `BCAPM_Master_V2` -> `BCAPM_Clean`

#### 1. Transformations Performed
- Strict deduplication on canonicalized `company_name`.
- Removal of invalid records lacking target label `target_success`.
- Type coercion (numeric variables to `float64`, integer counts to `int64`, categorical strings to clean category codes).
- Group-based median imputation for missing numerical features (grouped by `market_category` and `country_code`).
- Mode imputation for missing categorical attributes.

#### 2. Merged Datasets
- Operating directly on `BCAPM_Master_V2.csv`.

#### 3. Columns Retained
- Cleaned entity metadata, numerical features, macro backdrop metrics, target label.

#### 4. Columns Removed
- Duplicate entity rows, low-variance/zero-variance columns.

#### 5. Engineered Features
- Imputation flags (`is_imputed_funding`, `is_imputed_sentiment`) for transparency.

#### 6. Necessity
Machine learning and distance metrics (e.g., Euclidean/Cosine precedent matching) fail when encountering missing values, NaNs, infinite numbers, or corrupted data types. Stage 3 guarantees dataset mathematical hygiene.

---

### Stage 4: `BCAPM_Clean` -> `BCAPM_Preprocessed`

#### 1. Transformations Performed
- Non-linear feature engineering (ratios, interaction terms, logarithmic transformations).
- Categorical one-hot encoding for high-cardinality nominal features (`market_category`, `country_code`).
- Robust feature scaling (`StandardScaler`) applied to continuous numeric attributes.

#### 2. Merged Datasets
- Operating directly on `BCAPM_Clean.csv`.

#### 3. Columns Retained
- Preprocessed numeric features, one-hot encoded indicators, engineered ratio features, target vector.

#### 4. Columns Removed
- Raw non-numeric string identifier columns (retained separately in index mapping file).

#### 5. Engineered Features
- `capital_efficiency`: `funding_total_usd` / (`company_age_years` + 1)
- `media_to_funding_ratio`: `hn_public_engagement` / (`funding_total_usd` + 1000)
- `backdrop_growth_factor`: `internet_penetration_at_founding` * `entrepreneurial_financing_index`

#### 6. Necessity
Converts cleaned data into a standardized mathematical tensor ready for model training, analogy distance search, KNN precedent modeling, and neural network inference.
