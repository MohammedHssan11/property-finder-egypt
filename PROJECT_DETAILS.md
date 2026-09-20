# Egypt Real Estate Geospatial Intelligence & Investment Platform
> **Domain**: Geospatial Machine Learning, Dual Buy/Rent Hedonic Valuation & Real Estate Financial Engineering  
> **Repository**: `property-finder-egypt` | **Core Stack**: CatBoost, Scikit-Learn, Haversine Geospatial, Streamlit, Plotly

---

## 1. Executive Summary & Problem Statement

Egypt's real estate sector is characterized by intense price volatility, rapid urbanization into desert growth corridors (New Administrative Capital, New Cairo, 6th of October), and high currency inflation. Traditional real estate valuation relies on lagging appraisals and anecdotal developer claims.

This platform implements an enterprise-grade automated valuation model (AVM) and geospatial investment intelligence engine trained on over 62,000 real property listings. It combines:
1. **Hedonic Valuation Ensembles** for both residential purchase prices and monthly rental cash flows.
2. **Geodesic Spatial Feature Engineering** measuring proximity to key commercial hubs and infrastructure corridors.
3. **Financial Engineering Engine** simulating gross/net rental yields, developer installments, mortgage amortizations, and inflation-hedged capital appreciation.
4. **Dual Streamlit Portals & Production CLI** for both frontline investors and institutional analysts.

---

## 2. System Architecture & Workflow

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

## 3. Geospatial Feature Engineering & Proximity Metrics

### 3.1 Geodesic Landmark Proximity
Given property coordinates $(\phi_1, \lambda_1)$ and economic landmark $(\phi_2, \lambda_2)$:
$$d = 2R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$

Key reference landmarks tracked:
- **Cairo Central Business District (Tahrir / Downtown)**: $30.0444^\circ\text{N}, 31.2357^\circ\text{E}$
- **New Administrative Capital (Iconic Tower / CBD)**: $30.0131^\circ\text{N}, 31.7456^\circ\text{E}$
- **New Cairo Golden Square**: $30.0280^\circ\text{N}, 31.5120^\circ\text{E}$
- **Sheikh Zayed City Center / Arkan**: $30.0210^\circ\text{N}, 30.9830^\circ\text{E}$

### 3.2 Accessibility Score Formulation
A composite accessibility index $\mathcal{A} \in [0, 100]$ evaluates proximity to growth anchors with exponential decay:
$$\mathcal{A} = \sum_{k} w_k \cdot \exp\left(-\frac{d_k}{\lambda_k}\right) \times 100$$
Where weights $w_k$ reflect economic employment gravity.

---

## 4. Real Estate Financial Engineering & ROI

1. **Gross Rental Yield (GRY)**:
   $$\text{GRY} = \frac{\text{Annual Rent}}{\text{Purchase Price}} \times 100\%$$
2. **Net Rental Yield (NRY)**:
   $$\text{NRY} = \frac{\text{Annual Rent} \times (1 - \text{Vacancy Rate}) - \text{Maintenance} - \text{Management Fees}}{\text{Total Acquisition Cost}} \times 100\%$$
3. **Developer Cash-Flow Simulator**:
   Simulates 5-to-10-year quarterly payment schedules, balloon delivery payments, and discounted NPV comparisons against upfront cash discount rates.

---

## 5. Repository File Structure

```text
property-finder-egypt/
├── app.py                          # Streamlit Valuation & Quick ROI Portal
├── app2.py                         # Streamlit 4-Tab Real Estate Investment Studio
├── cli.py                          # Enterprise CLI (predict, roi, batch, audit)
├── spatial.py                      # Geospatial feature extraction & landmark distances
├── roi.py                          # Real estate investment, cash flow & yield engine
├── models/                         # Serialized CatBoost and Pipeline models
├── data/                           # Tabular processed datasets and sample listings
├── evals/
│   ├── runner.py                   # Automated benchmark evaluation harness
│   └── BENCHMARK_RESULTS.md        # Empirical micro-market benchmark report
├── tests/
│   ├── test_valuation.py           # Core valuation pipeline tests
│   ├── test_valuation_deep.py      # Edge case and log-transform tests
│   ├── test_spatial.py             # Geospatial proximity and distance tests
│   ├── test_roi.py                 # Cash flow and rental yield tests
│   ├── test_cli.py                 # CLI end-to-end command tests
│   └── test_evals.py               # Benchmark harness validation tests
├── requirements.txt                # Production and testing dependencies
├── PROJECT_DETAILS.md              # Technical design and architecture specification
└── README.md                       # Repository overview and quick start guide
```

---

## 6. Verification & Quality Assurance

- **Unit & Integration Tests**: 28 tests passing (`pytest tests/`).
- **Core Library Coverage**: 97% code coverage in `spatial.py` and `roi.py`.
- **Benchmark Pass Rate**: 100% spatial proximity accuracy, 87.5% rental yield sanity pass rate across Egyptian governorates.
- **Security Audit**: 0 uncommitted secrets, clean `.gitignore`.
