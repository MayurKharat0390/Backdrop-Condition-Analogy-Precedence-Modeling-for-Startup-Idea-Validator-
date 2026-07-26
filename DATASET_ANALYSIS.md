# Dataset Analysis & Schema Ingestion Report
**Project**: Backdrop-Conditioned Analogy Precedent Modeling (BCAPM) for Startup Success Prediction
**Source**: `Final.zip` Ingestion Audit

---

## Executive Summary
This document provides a systematic analysis of all 37 CSV dataset files contained within `Final.zip`. 
In accordance with BCAPM methodology, datasets are categorized into six micro-level and macro-level feature domains: Crunchbase Ingestion, Founder/Executive Profiles, Y Combinator & Hacker News Sentiment, Detailed Startup Operational Metrics, Funding & Geography, and World Bank Macroeconomic Backdrop Indicators.

### Master Ingestion Summary Table

| Index | File Path | Category | Rows | Cols | Duplicates | Missing Total | Target Merge Key |
|---|---|---|---|---|---|---|---|
| 1 | `API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv` | Macro Backdrop & Economy | 265 | 71 | 0 | 10,859 | `Country Code` / `Year` |
| 2 | `D1/acquisitions.csv` | Crunchbase & Investment | 18,968 | 18 | 44 | 46,118 | `company_permalink` / `name` |
| 3 | `D1/companies.csv` | Crunchbase & Investment | 66,368 | 14 | 0 | 55,015 | `company_permalink` / `name` |
| 4 | `D1/rounds.csv` | Crunchbase & Investment | 114,949 | 12 | 0 | 147,165 | `company_permalink` / `name` |
| 5 | `D10/Final - Copy.csv` | Crunchbase & Investment | 63 | 46 | 3 | 211 | `company_permalink` / `name` |
| 6 | `D10/Startup_failure.csv` | Crunchbase & Investment | 10 | 2 | 0 | 0 | `company_permalink` / `name` |
| 7 | `D10/startup_indus.csv` | Crunchbase & Investment | 11 | 3 | 0 | 0 | `company_permalink` / `name` |
| 8 | `D2/founder_V0.3_founder.csv` | Founder & Executive | 18,361 | 21 | 1,114 | 0 | `Company` / `Founder` |
| 9 | `D3/funding-data.csv` | Funding & Geography | 670 | 8 | 4 | 816 | `name` / `id` / `state` |
| 10 | `D3/states.csv` | Funding & Geography | 51 | 2 | 0 | 0 | `name` / `id` / `state` |
| 11 | `D3/test-data.csv` | Funding & Geography | 4,337 | 47 | 0 | 11,619 | `name` / `id` / `state` |
| 12 | `D3/train-data.csv` | Funding & Geography | 1,154 | 49 | 0 | 1,744 | `name` / `id` / `state` |
| 13 | `D4/data.csv` | Funding & Geography | 61,398 | 11 | 0 | 52,583 | `name` / `id` / `state` |
| 14 | `D5/ChartData.csv` | Diagnostics | 10 | 2 | 0 | 0 | N/A |
| 15 | `D5/CleanStartups.csv` | YC & HN Sentiment | 617 | 25 | 0 | 4,826 | `Company` |
| 16 | `D5/CleanStartupsFull.csv` | YC & HN Sentiment | 617 | 28 | 0 | 5,064 | `Company` |
| 17 | `D5/CleanStartupsFull2.csv` | YC & HN Sentiment | 617 | 28 | 0 | 5,417 | `Company` |
| 18 | `D5/CleanStartupsFull3.csv` | YC & HN Sentiment | 617 | 28 | 0 | 5,370 | `Company` |
| 19 | `D5/CompleteSet.csv` | YC & HN Sentiment | 507 | 35 | 0 | 4,038 | `Company` |
| 20 | `D5/CompleteSet2.csv` | YC & HN Sentiment | 507 | 35 | 0 | 4,316 | `Company` |
| 21 | `D5/CompleteSet3.csv` | YC & HN Sentiment | 506 | 36 | 0 | 4,272 | `Company` |
| 22 | `D5/Founders.csv` | Founder & Executive | 998 | 3 | 0 | 0 | `Company` / `Founder` |
| 23 | `D5/FullSet.csv` | YC & HN Sentiment | 508 | 31 | 0 | 3,863 | `Company` |
| 24 | `D5/HNTitleCommentsSentimentData.csv` | YC & HN Sentiment | 508 | 7 | 0 | 0 | `Company` |
| 25 | `D5/HNTitleCommentsSentimentData2.csv` | YC & HN Sentiment | 506 | 8 | 0 | 0 | `Company` |
| 26 | `D5/PlotData.csv` | Diagnostics | 30 | 2 | 0 | 0 | N/A |
| 27 | `D5/Startups.csv` | YC & HN Sentiment | 688 | 19 | 0 | 2,501 | `Company` |
| 28 | `D6/AB.csv` | YC & HN Sentiment | 688 | 6 | 0 | 208 | `Company` |
| 29 | `D7/50_Startups.csv` | Startup Operations & CAX | 50 | 5 | 0 | 0 | `Company_Name` |
| 30 | `D8/Startup_Data.csv` | Startup Operations & CAX | 472 | 116 | 0 | 2,983 | `Company_Name` |
| 31 | `D8/Startup_Data_Dictionary.csv` | Startup Operations & CAX | 116 | 2 | 0 | 75 | `Company_Name` |
| 32 | `D9/CAX_Startup_Data.csv` | Startup Operations & CAX | 472 | 116 | 0 | 2,983 | `Company_Name` |
| 33 | `D9/after_chi_sqr_0.01.csv` | Startup Operations & CAX | 447 | 21 | 0 | 0 | `Company_Name` |
| 34 | `D9/final_clean_data.csv` | Startup Operations & CAX | 439 | 17 | 0 | 0 | `Company_Name` |
| 35 | `Metadata_Country_API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv` | Macro Backdrop & Economy | 264 | 6 | 0 | 488 | `Country Code` / `Year` |
| 36 | `Metadata_Indicator_API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv` | Macro Backdrop & Economy | 1 | 5 | 0 | 1 | `Country Code` / `Year` |
| 37 | `investments_VC.csv` | Crunchbase & Investment | 54,294 | 39 | 4,855 | 281,768 | `company_permalink` / `name` |

---

## Detailed Dataset Profiles

### API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv (`API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv`)
- **Purpose**: World Bank annual Internet Penetration indicator data (% of population using the internet) across 265 countries/regions from 1960 to 2023.
- **Rows**: 265
- **Columns**: 71
- **Duplicates**: 0
- **Missing Values Total**: 10,859
- **Important Features**: `Country Name`, `Country Code`, `Indicator Name`, `Indicator Code`, Yearly columns (`1960` through `2023`)
- **Suggested Merge Keys**: `Country Code` + `Year` (via unpivot / melt)
- **Potential Preprocessing Steps**: Skip first 4 header metadata lines; melt yearly columns into tidy format (`Country Code`, `Year`, `Internet_Penetration_Pct`); forward fill missing annual values.
- **Role in BCAPM Dataset**: Core macroeconomic backdrop feature integrated into `BCAPM_Master_V2`.

### acquisitions.csv (`D1/acquisitions.csv`)
- **Purpose**: Tracks M&A transactions, acquisition dates, price amounts, acquirer and acquired company permalinks.
- **Rows**: 18,968
- **Columns**: 18
- **Duplicates**: 44
- **Missing Values Total**: 46,118
- **Important Features**: `company_permalink`, `company_name`, `acquirer_permalink`, `acquirer_name`, `acquired_at`, `price_amount_usd`
- **Suggested Merge Keys**: `company_permalink` -> `company_name`
- **Potential Preprocessing Steps**: Standardize permalinks to lower case; clean `acquired_at` date formats; convert `price_amount_usd` to numeric; extract acquisition year/quarter.
- **Role in BCAPM Dataset**: Defines positive outcome label (`Acquired`) and exit valuation for BCAPM precedent modeling.

### companies.csv (`D1/companies.csv`)
- **Purpose**: Primary Crunchbase company directory containing core firmographics, founding dates, locations, categories, and final status.
- **Rows**: 66,368
- **Columns**: 14
- **Duplicates**: 0
- **Missing Values Total**: 55,015
- **Important Features**: `permalink`, `name`, `category_list`, `funding_total_usd`, `status`, `country_code`, `state_code`, `city`, `founded_at`
- **Suggested Merge Keys**: `permalink` / `name`
- **Potential Preprocessing Steps**: Filter out corrupted rows; extract primary market from `category_list`; fill missing location codes with `UNKNOWN`; normalize funding amounts.
- **Role in BCAPM Dataset**: Serves as the core entity backbone for `BCAPM_Master_V1` startup records.

### rounds.csv (`D1/rounds.csv`)
- **Purpose**: Detailed funding round histories per company including round type (seed, series A/B/C), raised amounts, and investor counts.
- **Rows**: 114,949
- **Columns**: 12
- **Duplicates**: 0
- **Missing Values Total**: 147,165
- **Important Features**: `company_permalink`, `company_name`, `funding_round_type`, `funding_round_code`, `raised_amount_usd`, `funded_at`
- **Suggested Merge Keys**: `company_permalink` -> `company_name`
- **Potential Preprocessing Steps**: Group by `company_permalink`; aggregate total raised amount, round count, and elapsed days between rounds; compute trajectory velocity.
- **Role in BCAPM Dataset**: Provides dynamic funding trajectory precedent features for analogy matching.

### Final - Copy.csv (`D10/Final - Copy.csv`)
- **Purpose**: Global economic indicators dataset covering 63 economies with Human Development Index (HDI), entrepreneurial financing, government support, and bureaucracy metrics.
- **Rows**: 63
- **Columns**: 46
- **Duplicates**: 3
- **Missing Values Total**: 211
- **Important Features**: `Code`, `Economy`, `HDI`, `Corrected_HDI`, `Financing for entrepreneurs`, `Governmental support and policies`, `Taxes and bureaucracy`
- **Suggested Merge Keys**: `Code` -> `Country Code`
- **Potential Preprocessing Steps**: Clean trailing whitespaces; standardize country codes; normalize index scores (0-10 scale).
- **Role in BCAPM Dataset**: Provides national backdrop metrics (policy, HDI, tax burden) for `BCAPM_Master_V2`.

### Startup_failure.csv (`D10/Startup_failure.csv`)
- **Purpose**: Historical macro startup failure rates tracked annually across economic cycles.
- **Rows**: 10
- **Columns**: 2
- **Duplicates**: 0
- **Missing Values Total**: 0
- **Important Features**: `Year`, `Failure`
- **Suggested Merge Keys**: `Year`
- **Potential Preprocessing Steps**: Convert `Failure` rate string to float ratio.
- **Role in BCAPM Dataset**: Provides annual macro market risk backdrop indicator.

### startup_indus.csv (`D10/startup_indus.csv`)
- **Purpose**: Industry-level baseline success and failure rates.
- **Rows**: 11
- **Columns**: 3
- **Duplicates**: 0
- **Missing Values Total**: 0
- **Important Features**: `Serial`, `Industry`, `Success`
- **Suggested Merge Keys**: `Industry` / `Market`
- **Potential Preprocessing Steps**: Clean industry string names; calculate sector baseline success rate.
- **Role in BCAPM Dataset**: Sector-level backdrop baseline probability indicator.

### founder_V0.3_founder.csv (`D2/founder_V0.3_founder.csv`)
- **Purpose**: Granular founder profile data capturing prior investments, lead investments, news coverage, and organizational count.
- **Rows**: 18,361
- **Columns**: 21
- **Duplicates**: 1,114
- **Missing Values Total**: 0
- **Important Features**: `Full Name`, `Primary Job Title`, `Bio`, `Gender`, `Number of News Articles`, `Number of Founded Organizations`, `Number of Portfolio Companies`, `Number of Investments_x`
- **Suggested Merge Keys**: `Full Name` / Organization mapping
- **Potential Preprocessing Steps**: Extract executive experience counts; handle 1,114 duplicate rows; encode gender; impute missing news coverage scores.
- **Role in BCAPM Dataset**: Provides founder human capital and track record precedents for BCAPM.

### funding-data.csv (`D3/funding-data.csv`)
- **Purpose**: Acquisition and funding transaction logs mapping acquired vs acquiring company object IDs and USD prices.
- **Rows**: 670
- **Columns**: 8
- **Duplicates**: 4
- **Missing Values Total**: 816
- **Important Features**: `id`, `acquiring_company_name`, `acquired_company_name`, `price_amount`, `funding_total_usd`, `status`
- **Suggested Merge Keys**: `acquired_company_name` -> `Company`
- **Potential Preprocessing Steps**: Reconcile duplicate records; parse string currency values into float USD; align company naming conventions.
- **Role in BCAPM Dataset**: Provides secondary acquisition price validation and outcome label sanity checks.

### states.csv (`D3/states.csv`)
- **Purpose**: Lookup table mapping US state names to standard 2-letter state postal abbreviations.
- **Rows**: 51
- **Columns**: 2
- **Duplicates**: 0
- **Missing Values Total**: 0
- **Important Features**: `State`, `Abbreviation`
- **Suggested Merge Keys**: `State` / `state_code`
- **Potential Preprocessing Steps**: Ensure clean uppercase mapping across state names and 2-letter postal abbreviations.
- **Role in BCAPM Dataset**: Geographic location normalization across Crunchbase, YC, and World Bank datasets.

### test-data.csv (`D3/test-data.csv`)
- **Purpose**: Validation partition of startup attributes including geographic coordinates, category codes, milestones, and status.
- **Rows**: 4,337
- **Columns**: 47
- **Duplicates**: 0
- **Missing Values Total**: 11,619
- **Important Features**: `state`, `latitude`, `longitude`, `zip_code`, `id`, `city`, `name`, `status`, `labels`, `founded_at`
- **Suggested Merge Keys**: `name`
- **Potential Preprocessing Steps**: Combine coordinates into spatial features; clean missing zip codes; align category features.
- **Role in BCAPM Dataset**: Validation cohort for evaluating precedent similarity and model generalization.

### train-data.csv (`D3/train-data.csv`)
- **Purpose**: Training partition of startup features containing geographic coordinates, relationships, funding rounds, and milestone counts.
- **Rows**: 1,154
- **Columns**: 49
- **Duplicates**: 0
- **Missing Values Total**: 1,744
- **Important Features**: `state_code`, `latitude`, `longitude`, `zip_code`, `id`, `city`, `name`, `status`, `funding_rounds`, `funding_total_usd`
- **Suggested Merge Keys**: `name`
- **Potential Preprocessing Steps**: Clean invalid column headers (`Unnamed: 0`, `Unnamed: 6`); impute missing USD amounts with median by category.
- **Role in BCAPM Dataset**: Primary training substrate for initial precedent feature extraction.

### data.csv (`D4/data.csv`)
- **Purpose**: Extended market profile dataset covering 61,398 startups with market categories, funding totals, and location metrics.
- **Rows**: 61,398
- **Columns**: 11
- **Duplicates**: 0
- **Missing Values Total**: 52,583
- **Important Features**: `name`, `market`, `funding_total_usd`, `status`, `country_code`, `state_code`, `city`, `funding_rounds`, `founded_at`
- **Suggested Merge Keys**: `name`
- **Potential Preprocessing Steps**: Strip trailing whitespace from `market`; convert funding strings to floats; compute company age from `founded_at`.
- **Role in BCAPM Dataset**: Broadens startup universe to 61k+ records for market density and analog selection.

### ChartData.csv (`D5/ChartData.csv`)
- **Purpose**: Model evaluation artifact storing feature importance ranking metrics.
- **Rows**: 10
- **Columns**: 2
- **Duplicates**: 0
- **Missing Values Total**: 0
- **Important Features**: `Features`, `Feature Values`
- **Suggested Merge Keys**: None (Diagnostic)
- **Potential Preprocessing Steps**: Parse feature name strings and numerical importance weights.
- **Role in BCAPM Dataset**: Reference benchmark for diagnostic feature importance comparison.

### CleanStartups.csv (`D5/CleanStartups.csv`)
- **Purpose**: Cleaned dataset of Y Combinator startups detailing company names, fate (Operating/Acquired/Dead), YC batch year/session, total funds, and city.
- **Rows**: 617
- **Columns**: 25
- **Duplicates**: 0
- **Missing Values Total**: 4,826
- **Important Features**: `Company`, `Fate`, `Description`, `YCYear`, `YCSession`, `TotalFunds`, `City`, `Logo`
- **Suggested Merge Keys**: `Company`
- **Potential Preprocessing Steps**: Remove `Unnamed: 0` index column; clean currency strings in `TotalFunds`; standardize `Fate` into binary success label.
- **Role in BCAPM Dataset**: Baseline seed dataset for YC cohort precedent modeling.

### CleanStartupsFull.csv (`D5/CleanStartupsFull.csv`)
- **Purpose**: Enriched YC startup dataset incorporating founder lists and investor names.
- **Rows**: 617
- **Columns**: 28
- **Duplicates**: 0
- **Missing Values Total**: 5,064
- **Important Features**: `Company`, `Fate`, `Description`, `Market`, `Founders`, `YCYear`, `YCSession`, `Investors`, `TotalFunds`
- **Suggested Merge Keys**: `Company`
- **Potential Preprocessing Steps**: Parse stringified founder and investor lists; calculate co-founder counts and investor pool size.
- **Role in BCAPM Dataset**: Adds network features (founders/investors) to YC startup records.

### CleanStartupsFull2.csv (`D5/CleanStartupsFull2.csv`)
- **Purpose**: Iteration 2 of enriched YC dataset with refined market classifications.
- **Rows**: 617
- **Columns**: 28
- **Duplicates**: 0
- **Missing Values Total**: 5,417
- **Important Features**: `Company`, `Fate`, `Description`, `Market`, `Founders`, `YCYear`, `YCSession`, `Investors`, `TotalFunds`
- **Suggested Merge Keys**: `Company`
- **Potential Preprocessing Steps**: Reconcile duplicate market definitions; impute missing total funds using market sector medians.
- **Role in BCAPM Dataset**: Intermediate staging version for YC startup enrichment.

### CleanStartupsFull3.csv (`D5/CleanStartupsFull3.csv`)
- **Purpose**: Production version of YC startup dataset with complete founder, investor, market, and funding details.
- **Rows**: 617
- **Columns**: 28
- **Duplicates**: 0
- **Missing Values Total**: 5,370
- **Important Features**: `Company`, `Fate`, `Description`, `Market`, `Founders`, `YCYear`, `YCSession`, `Investors`, `TotalFunds`, `City`
- **Suggested Merge Keys**: `Company`
- **Potential Preprocessing Steps**: Final missing value audit; standardize company names; normalize fate outcome labels into target `Success` variable.
- **Role in BCAPM Dataset**: Core component merged into `BCAPM_Master_V1`.

### CompleteSet.csv (`D5/CompleteSet.csv`)
- **Purpose**: Integrated dataset joining YC startups with Hacker News post titles, points, top comments, and sentiment scores.
- **Rows**: 507
- **Columns**: 35
- **Duplicates**: 0
- **Missing Values Total**: 4,038
- **Important Features**: `Company`, `Title`, `TitlePoints`, `TopComment`, `Sentiment`, `TopCommentPoints`, `Fate`, `Description`, `Market`
- **Suggested Merge Keys**: `Company`
- **Potential Preprocessing Steps**: Clean text noise; handle missing sentiment values; numeric scaling of title and comment points.
- **Role in BCAPM Dataset**: Early integration test set for Hacker News public perception signals.

### CompleteSet2.csv (`D5/CompleteSet2.csv`)
- **Purpose**: Iteration 2 of integrated sentiment & startup dataset with enhanced sentiment point weighting.
- **Rows**: 507
- **Columns**: 35
- **Duplicates**: 0
- **Missing Values Total**: 4,316
- **Important Features**: `Company`, `Title`, `TitlePoints`, `TopComment`, `Sentiment`, `TopCommentPoints`, `Fate`, `Market`
- **Suggested Merge Keys**: `Company`
- **Potential Preprocessing Steps**: Impute missing sentiment scores using neutral polarity (0.0); derive community engagement index.
- **Role in BCAPM Dataset**: Intermediate sentiment integration version.

### CompleteSet3.csv (`D5/CompleteSet3.csv`)
- **Purpose**: Final production integrated dataset containing complete Hacker News title URLs, top comments, sentiment polarity, and points.
- **Rows**: 506
- **Columns**: 36
- **Duplicates**: 0
- **Missing Values Total**: 4,272
- **Important Features**: `Company`, `Title`, `TitlePoints`, `TitleURL`, `TopComment`, `Sentiment`, `TopCommentPoints`, `Fate`, `Description`, `Market`, `Founders`
- **Suggested Merge Keys**: `Company`
- **Potential Preprocessing Steps**: Standardize sentiment scores; merge title & comment engagement scores into unified `HN_Public_Sentiment_Score`.
- **Role in BCAPM Dataset**: Primary sentiment input merged into `BCAPM_Master_V1`.

### Founders.csv (`D5/Founders.csv`)
- **Purpose**: Mapping of YC startup co-founders with gender attributes.
- **Rows**: 998
- **Columns**: 3
- **Duplicates**: 0
- **Missing Values Total**: 0
- **Important Features**: `Founder`, `Company`, `Gender`
- **Suggested Merge Keys**: `Company` / `Founder`
- **Potential Preprocessing Steps**: Group by `Company`; calculate total founder count, female founder presence ratio, and founder team diversity.
- **Role in BCAPM Dataset**: Provides team composition metrics for `BCAPM_Master_V1`.

### FullSet.csv (`D5/FullSet.csv`)
- **Purpose**: Full join of YC startups, founder demographic metrics, and HN sentiment indices.
- **Rows**: 508
- **Columns**: 31
- **Duplicates**: 0
- **Missing Values Total**: 3,863
- **Important Features**: `Id`, `Company`, `Title`, `TitlePoints`, `TopComment`, `Sentiment`, `TopCommentPoints`, `Fate`, `YCYear`
- **Suggested Merge Keys**: `Company`
- **Potential Preprocessing Steps**: Remove redundant ID indices; align YC year cohorts.
- **Role in BCAPM Dataset**: Structural baseline for complete feature verification.

### HNTitleCommentsSentimentData.csv (`D5/HNTitleCommentsSentimentData.csv`)
- **Purpose**: Extracted Hacker News title and comment text with associated NLP sentiment analysis scores.
- **Rows**: 508
- **Columns**: 7
- **Duplicates**: 0
- **Missing Values Total**: 0
- **Important Features**: `Company`, `Title`, `TitlePoints`, `TopComment`, `Sentiment`, `TopCommentPoints`
- **Suggested Merge Keys**: `Company`
- **Potential Preprocessing Steps**: Extract mean sentiment polarity; filter out non-startup HN submissions.
- **Role in BCAPM Dataset**: Provides text sentiment features for early public sentiment analogy matching.

### HNTitleCommentsSentimentData2.csv (`D5/HNTitleCommentsSentimentData2.csv`)
- **Purpose**: Updated Hacker News sentiment dataset including submission URLs and comment scores.
- **Rows**: 506
- **Columns**: 8
- **Duplicates**: 0
- **Missing Values Total**: 0
- **Important Features**: `Company`, `Title`, `TitlePoints`, `TitleURL`, `TopComment`, `Sentiment`, `TopCommentPoints`
- **Suggested Merge Keys**: `Company`
- **Potential Preprocessing Steps**: Extract URL domain authority metrics; clean comment strings.
- **Role in BCAPM Dataset**: Refined NLP sentiment source dataset.

### PlotData.csv (`D5/PlotData.csv`)
- **Purpose**: Diagnostic dataset mapping hyperparameter tuning iterations (`nTrees`) to validation performance scores.
- **Rows**: 30
- **Columns**: 2
- **Duplicates**: 0
- **Missing Values Total**: 0
- **Important Features**: `nTrees`, `Scores`
- **Suggested Merge Keys**: None (Diagnostic)
- **Potential Preprocessing Steps**: N/A
- **Role in BCAPM Dataset**: Performance diagnostic trace.

### Startups.csv (`D5/Startups.csv`)
- **Purpose**: Raw YC startup listing including locations, descriptions, categories, founders, YC batch info, and investors.
- **Rows**: 688
- **Columns**: 19
- **Duplicates**: 0
- **Missing Values Total**: 2,501
- **Important Features**: `Company`, `Satus`, `Year Founded`, `Mapping Location`, `Description`, `Categories`, `Founders`, `Y Combinator Year`
- **Suggested Merge Keys**: `Company`
- **Potential Preprocessing Steps**: Fix typo column `Satus` -> `Status`; parse location hierarchy into City/State/Country.
- **Role in BCAPM Dataset**: Raw entry point for YC startup pipeline.

### AB.csv (`D6/AB.csv`)
- **Purpose**: YC startup dataset with funding amounts raised across different rounds and headquarters city.
- **Rows**: 688
- **Columns**: 6
- **Duplicates**: 0
- **Missing Values Total**: 208
- **Important Features**: `Company`, `Satus`, `Amounts_raised_in_different_funding_rounds`, `Categories`, `YCombinatorYear`, `HeadquartersCity`
- **Suggested Merge Keys**: `Company`
- **Potential Preprocessing Steps**: Parse list of round amounts; extract total funding, max round funding, and round count.
- **Role in BCAPM Dataset**: Provides granular round progression features for YC startups.

### 50_Startups.csv (`D7/50_Startups.csv`)
- **Purpose**: Benchmark financial dataset containing R&D spend, administration spend, marketing spend, state location, and profit.
- **Rows**: 50
- **Columns**: 5
- **Duplicates**: 0
- **Missing Values Total**: 0
- **Important Features**: `R&D Spend`, `Administration`, `Marketing Spend`, `State`, `Profit`
- **Suggested Merge Keys**: State / Macro benchmark
- **Potential Preprocessing Steps**: Normalize spend ratios; compute R&D-to-Marketing spend efficiency ratio.
- **Role in BCAPM Dataset**: Provides financial burn rate benchmarks for precedent modeling.

### Startup_Data.csv (`D8/Startup_Data.csv`)
- **Purpose**: High-dimensional startup profile dataset containing 116 operational, technical, executive, and market features across 472 companies.
- **Rows**: 472
- **Columns**: 116
- **Duplicates**: 0
- **Missing Values Total**: 2,983
- **Important Features**: `Company_Name`, `Dependent-Company Status`, `year of founding`, `Age of company in years`, `Internet Activity Score`, `Industry of company`, `Number of Co-founders`, `Team size Senior leadership`, `Machine Learning based business`, `B2C or B2B venture`
- **Suggested Merge Keys**: `Company_Name`
- **Potential Preprocessing Steps**: Impute missing values using median/mode; encode binary indicators (0/1); clean special characters in feature headers.
- **Role in BCAPM Dataset**: Core operational feature source merged into `BCAPM_Master_V1`.

### Startup_Data_Dictionary.csv (`D8/Startup_Data_Dictionary.csv`)
- **Purpose**: Data dictionary detailing description and definitions for all 116 variables in `Startup_Data.csv`.
- **Rows**: 116
- **Columns**: 2
- **Duplicates**: 0
- **Missing Values Total**: 75
- **Important Features**: `Variable`, `Description`
- **Suggested Merge Keys**: `Variable`
- **Potential Preprocessing Steps**: Clean formatting strings.
- **Role in BCAPM Dataset**: Metadata documentation reference for feature engineering.

### CAX_Startup_Data.csv (`D9/CAX_Startup_Data.csv`)
- **Purpose**: Competitive Analytics Exchange (CAX) startup dataset containing identical 116 high-dimensional features as D8.
- **Rows**: 472
- **Columns**: 116
- **Duplicates**: 0
- **Missing Values Total**: 2,983
- **Important Features**: `Company_Name`, `Dependent-Company Status`, `year of founding`, `Internet Activity Score`, `Number of Investors in Seed`, `Team Composition score`, `Percent_skill_Data Science`
- **Suggested Merge Keys**: `Company_Name`
- **Potential Preprocessing Steps**: Deduplicate against D8 records; standardize binary label format (`Success` vs `Failed`).
- **Role in BCAPM Dataset**: Primary operational dataset providing deep behavioral metrics.

### after_chi_sqr_0.01.csv (`D9/after_chi_sqr_0.01.csv`)
- **Purpose**: Feature-selected subset of CAX startup dataset post Chi-Square test (p < 0.01 threshold) containing 21 statistically significant predictors.
- **Rows**: 447
- **Columns**: 21
- **Duplicates**: 0
- **Missing Values Total**: 0
- **Important Features**: `Company_Name`, `Dependent.Company.Status`, `Number.of.Co.founders`, `Focus.functions.of.company`, `Average.size.of.companies.worked.for.in.the.past`, `Machine.Learning.based.business`, `Barriers.of.entry.for.the.competitors`
- **Suggested Merge Keys**: `Company_Name`
- **Potential Preprocessing Steps**: Strip dots from column names; ensure numeric encoding.
- **Role in BCAPM Dataset**: Identifies high-signal operational features for precedent distance computation.

### final_clean_data.csv (`D9/final_clean_data.csv`)
- **Purpose**: Refined subset of 17 core operational features after Chi-Square feature selection and multicollinearity filtering.
- **Rows**: 439
- **Columns**: 17
- **Duplicates**: 0
- **Missing Values Total**: 0
- **Important Features**: `Company_Name`, `Dependent.Company.Status`, `Number.of.Co.founders`, `Number.of.of.advisors`, `Team.size.Senior.leadership`, `Number.of.of.repeat.investors`, `Worked.in.top.companies`
- **Suggested Merge Keys**: `Company_Name`
- **Potential Preprocessing Steps**: Standardize variable names; align status categories.
- **Role in BCAPM Dataset**: Provides validated lean operational feature set.

### Metadata_Country_API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv (`Metadata_Country_API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv`)
- **Purpose**: World Bank country metadata detailing country region, income group classification, and special economic notes.
- **Rows**: 264
- **Columns**: 6
- **Duplicates**: 0
- **Missing Values Total**: 488
- **Important Features**: `Country Code`, `Region`, `IncomeGroup`, `SpecialNotes`, `TableName`
- **Suggested Merge Keys**: `Country Code`
- **Potential Preprocessing Steps**: Fill missing region codes; standardize income group categories (`High income`, `Upper middle income`, etc.).
- **Role in BCAPM Dataset**: Enriches macroeconomic backdrop with country income tier classifications for BCAPM.

### Metadata_Indicator_API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv (`Metadata_Indicator_API_IT.NET.USER.ZS_DS2_en_csv_v2_33086.csv`)
- **Purpose**: World Bank indicator definition metadata detailing indicator code, name, source notes, and organization.
- **Rows**: 1
- **Columns**: 5
- **Duplicates**: 0
- **Missing Values Total**: 1
- **Important Features**: `INDICATOR_CODE`, `INDICATOR_NAME`, `SOURCE_NOTE`, `SOURCE_ORGANIZATION`
- **Suggested Merge Keys**: `INDICATOR_CODE`
- **Potential Preprocessing Steps**: Extract indicator metadata.
- **Role in BCAPM Dataset**: Documentation metadata for macroeconomic indicators.

### investments_VC.csv (`investments_VC.csv`)
- **Purpose**: Comprehensive Venture Capital investment transaction log covering 54,294 funding events across global startups, including funding rounds A through H, seed, angel, and debt funding.
- **Rows**: 54,294
- **Columns**: 39
- **Duplicates**: 4,855
- **Missing Values Total**: 281,768
- **Important Features**: `permalink`, `name`, `category_list`, `market`, `funding_total_usd`, `status`, `country_code`, `state_code`, `city`, `funding_rounds`, `founded_year`, `seed`, `venture`, `round_A`, `round_B`
- **Suggested Merge Keys**: `permalink` / `name`
- **Potential Preprocessing Steps**: Clean column name whitespace (e.g. ` market ` -> `market`); parse funding amounts; aggregate funding round distribution.
- **Role in BCAPM Dataset**: Provides broad investor network and VC funding round metrics for `BCAPM_Master_V1`.
