# Egypt Real Estate Geospatial Intelligence & Investment Platform

[![CI](https://github.com/MohammedHssan11/property-finder-egypt/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammedHssan11/property-finder-egypt/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-28%20Passed-success?style=flat&logo=pytest&logoColor=white)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **GitHub Repository**: `property-finder-egypt`  
> **Repository Description**: Enterprise Real Estate Valuation and Investment Platform featuring Geospatial Landmark Proximity, Dual Buy/Rent Modeling, Gross/Net Rental Yields, Developer Installment Cash-Flow Simulators, and Benchmark Evaluations on Egyptian Property Listings.

A machine learning and geospatial intelligence platform trained on over 62,000 Property Finder Egypt listings. Predicts residential and commercial property values, computes geodesic landmark proximity, generates dual Buy-vs-Rent cash-flow models, simulates developer installments and mortgages, and scores investment opportunities across Egypt's primary growth corridors.

---

## 1. Enterprise System Architecture

```text
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                         INPUT PROPERTY & LOCATION PROFILE                        │
 │  Area (sqm) | Bedrooms | Bathrooms | Property Type | Location (2,907 locations)  │
 └────────────────────────────────────────┬─────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
   [Geospatial Feature Extractor]              [Dual-Engine Model Valuation]
   - Haversine landmark distances              - Purchase Price (Buy Model)
     (Cairo CBD, NAC, Golden Sq, Zayed)        - Monthly & Annual Rent (Rent Model)
   - Accessibility & Centrality Score          - Price per sqm vs Market Median
   - Compound & Premium Zone Flag              - Confidence Intervals (±IQR)
   - Local listing density & price index       
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                      [Real Estate ROI & Financial Modeling Engine]
                      - Gross Rental Yield (GRY) & Net Rental Yield (NRY)
                      - Vacancy, Maintenance Reserve & Property Management
                      - Developer Installment / Mortgage Cash Flow Simulator
                      - Multi-Year Capital Appreciation (Conservative / Base / High)
                      - Payback Period & Investment Score (A+ to C)
                                          │
             ┌────────────────────────────┼────────────────────────────┐
             ▼                            ▼                            ▼
   [Enterprise CLI Tool]        [Streamlit Dual Apps]        [Benchmark Harness]
   - predict / roi / audit      - app.py (Valuation + ROI)   - Ground-truth evaluation
   - batch portfolio rank       - app2.py (4-Tab Studio)     - evals/BENCHMARK_RESULTS.md
```

---

## 2. Empirical Benchmark Evaluation

The platform includes an automated evaluation harness ([`evals/runner.py`](runner.py)) evaluating ground-truth properties across 8 primary Egyptian micro-markets:

### Executive Performance Matrix

| Metric | Measured Value | Target Standard | Evaluation Status |
| :--- | :---: | :---: | :---: |
| **Spatial Hub Match & Proximity Rate** | **`100.0%`** | $\ge 85.0\%$ | **PASSED** |
| **Rental Yield Sanity Pass Rate** | **`87.5%`** | $\ge 85.0\%$ | **PASSED** |
| **Mean Portfolio Gross Rental Yield** | **`6.33%`** | $6.0\% - 12.0\%$ | **PASSED** |
| **Mean Accessibility Score** | **`57.4 / 100`** | $\ge 55.0$ | **PASSED** |
| **Mean Investment Opportunity Score** | **`57.4 / 100`** | $\ge 60.0$ | **PASSED** |

### Micro-Market Valuation & Yield Breakdown

| Market / Asset | Purchase Price (EGP) | Monthly Rent (EGP) | Gross Yield | Nearest Hub (Dist) | Investment Tier |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **New Cairo (Tagamoa) — Mivida** | `15,526,100` | `82,502` | **`6.38%`** | Golden Square (4.2 km) | **B (Balanced Market Asset)** |
| **New Administrative Capital (NAC)** | `5,697,877` | `40,619` | **`8.55%`** | Golden Square (9.8 km) | **A (Strong Growth & Yield)** |
| **Sheikh Zayed — Arkan Central** | `12,721,357` | `87,874` | **`8.29%`** | Zayed Central (2.1 km) | **A (Strong Growth & Yield)** |
| **Downtown Cairo — High Density** | `4,504,092` | `25,905` | **`6.90%`** | Cairo Airport (7.1 km) | **B (Balanced Market Asset)** |
| **6th of October — Palm Hills** | `12,010,914` | `51,444` | **`5.14%`** | Zayed Central (5.4 km) | **C (Income Lagging)** |
| **Alexandria — Stanley Waterfront** | `7,465,331` | `30,632` | **`4.92%`** | Alex Corniche (4.8 km) | **High Capital Risk** |
| **Red Sea — El Gouna Resort** | `14,249,364` | `52,157` | **`4.39%`** | Red Sea Hub (22.1 km) | **High Capital Risk** |
| **North Coast — New Alamein** | `9,037,331` | `45,735` | **`6.07%`** | Sahel Hub (150.6 km) | **C (Income Lagging)** |

*Detailed benchmark logs are published in [`evals/BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md).*

---

## 3. Key Engineering Features

1. **Geospatial Feature Extraction (`spatial.py`)**:
   - Geodesic (Haversine) and directional bearing distance calculation to primary hubs: Cairo CBD, New Capital Financial District, New Cairo Golden Square, Sheikh Zayed Arkan, Cairo International Airport, Alexandria Corniche, and Red Sea Hub.
   - Inverse distance-decay Accessibility Index (0–100).
   - Regex-based Master Developer & Gated Compound detection (`is_compound`, `developer_tier`).
   - Regional & city centroid imputation for uncoordinated listings.
2. **Real Estate ROI & Financial Modeling (`roi.py`)**:
   - **Dual Valuation**: Simultaneous prediction of asset purchase price and monthly tenancy rate.
   - **Rental Yields**: Computes Gross Rental Yield (GRY), Effective Gross Income (EGI), Net Operating Income (NOI), and Net Rental Yield (NRY).
   - **Financing Simulator**: Supports Developer Off-Plan Installments (equal 0% formal interest payments over 5, 7, 8 years), Bank Mortgages, and Cash purchases.
   - **Cash-on-Cash Return & Monthly Net Cash Flow**: Net rent minus debt service burden.
   - **Multi-Scenario Capital Appreciation**: 5-Year and 10-Year projections under Conservative (12%), Base (18%), and High Inflation (25%) CAGR.
   - **Investment Opportunity Score**: Composite rating from `A+ (Prime Institutional)` to `Speculative / High Risk`.
3. **Enterprise Command-Line Tooling (`cli.py`)**:
   - `predict`: Value property with confidence interval and proximity telemetry.
   - `roi`: Comprehensive real estate yield and debt service audit.
   - `spatial-audit`: Audit landmark distances, accessibility score, and active listings density.
   - `batch`: Batch evaluate and rank property portfolios by yield or investment score.
4. **Modernized Dual Web Applications**:
   - `app.py`: Clean, fast valuation and ROI estimator with landmark badges.
   - `app2.py`: NileLens AI Enterprise Studio featuring 4 interactive tabs (Valuation, ROI Simulator, Micro-Market Heatmap, Asset Comparator).

---

## 4. Repository Structure

```text
.
├── app.py                      # Production property valuation & ROI estimator
├── app2.py                     # NileLens AI Enterprise Real Estate Studio (4 tabs)
├── spatial.py                  # Geospatial feature extractor, landmark math & accessibility
├── roi.py                      # Real estate financial engine, yields & cash flow simulator
├── cli.py                      # Enterprise CLI tool (predict, roi, spatial-audit, batch)
├── evals/
│   ├── runner.py               # Automated benchmark evaluation harness
│   └── BENCHMARK_RESULTS.md    # Empirical evaluation report and metrics
├── models/
│   ├── property_price_model_bundle.joblib     # Production Random Forest pipeline & location index
│   └── candidate_catboost.cbm                 # Trained CatBoost benchmark model
├── artifacts/
│   └── model_card.json         # Transparent model card specifications
├── tests/
│   ├── test_valuation.py       # Basic bundle loading and prediction tests
│   ├── test_spatial.py         # Haversine, bearing, and accessibility tests
│   ├── test_roi.py             # Rental yield, cash flow, and appreciation tests
│   ├── test_cli.py             # CLI parser and subcommand tests
│   ├── test_evals.py           # Benchmark harness tests
│   └── test_valuation_deep.py  # Deep integration, commercial, and batch tests
├── requirements.txt            # Project dependencies
├── LICENSE                     # MIT License
└── README.md                   # Enterprise system documentation
```

---

## 5. Quick Start & Setup

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone https://github.com/your-username/property-finder-egypt.git
cd property-finder-egypt

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 6. Running the Applications

### 1. Simple Valuation & ROI Estimator
```bash
streamlit run app.py
```

### 2. NileLens AI Enterprise Studio
```bash
streamlit run app2.py
```

---

## 7. Command-Line Interface (CLI)

```bash
# 1. Predict property price with geospatial proximity telemetry
python cli.py predict --size 160 --bedrooms 3 --bathrooms 2 --location "Mivida"

# 2. Output valuation as JSON
python cli.py predict --size 140 --bedrooms 2 --location "Rehab" --json

# 3. Comprehensive Real Estate Investment ROI Audit
python cli.py roi --size 180 --bedrooms 3 --location "New Cairo" --financing developer_installment --years 7

# 4. Audit geospatial landmark distances and accessibility score
python cli.py spatial-audit --location "Sheikh Zayed"

# 5. Batch rank property portfolio by rental yield
python cli.py batch --input sample_portfolio.csv --output ranked_portfolio.csv --sort-by yield --top-k 5
```

---

## 8. Running Automated Tests & Benchmark

```bash
# Run all 28 automated tests
pytest tests/ -v

# Run with complete coverage report
pytest --cov=. --cov-report=term-missing tests/

# Run the benchmark evaluation suite
python evals/runner.py
```

---

## 9. License

Released under the [MIT License](LICENSE).
