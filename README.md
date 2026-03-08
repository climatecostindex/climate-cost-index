# Climate Cost Index (CCI)

A composite household-level metric for climate-linked cost pressure across U.S. counties.

CCI measures the relative intensity and acceleration of climate-linked cost signals and hazard exposures experienced by households, combining climate-attributed cost components with hazard burden proxies into a standardized index. The national median is set to 100. A score of 115 indicates approximately 15 percent greater climate-linked cost pressure than the median county.

**Paper:** [The Climate Cost Index: Measuring Climate-Linked Household Cost Pressure Across U.S. Counties](docs/) (SSRN working paper, March 2026)

## v1.0 Results

- **3,111 counties scored** (95.9% of CONUS and Alaska)
- **CCI-Score:** median 106.3, IQR 89–129, range 47–197
- **CCI-Dollar:** mean $43/household/year in climate-attributed energy costs
- **CCI-National:** 8.73 (population-weighted absolute cost pressure)
- **Scaling constant:** k = 1.873 (fixed at v1 launch)

## Product Forms

| Product | Description |
|---------|-------------|
| **CCI-Score** | Relative composite index, national median = 100 |
| **CCI-Dollar** | Dollar-denominated climate cost adder (energy only in v1) |
| **CCI-Strain** | CCI-Score / median household income, re-indexed |
| **CCI-National** | Population-weighted aggregate of raw composites |

## Quick Start

### Use pre-built scores

Download the scored output from [Releases](https://github.com/climatecostindex/climate-cost-index/releases):

```bash
gh release download v1.0
tar -xzf cci-data-v1.0.tar.gz
```

### Full replication from source

```bash
git clone git@github.com:climatecostindex/climate-cost-index.git
cd climate-cost-index

# Set up environment
cp .env.example .env   # Add your API keys (all free, see below)
uv sync                # or: pip install -e .

# Run the full pipeline
python pipeline/run_ingest.py      # Download raw data (~6.6 GB, see note below)
python pipeline/run_transform.py   # Harmonize to county-year
python pipeline/run_score.py       # Compute CCI scores
python pipeline/run_validate.py    # Run validation suite
```

**Ingest timing:** Most sources download in under an hour. FEMA NFHL (~3 GB of geodatabase files) can take significantly longer depending on FEMA server responsiveness. Total ingest time ranges from 2–8+ hours. Each ingester is idempotent — if interrupted, rerun and it will skip completed downloads.

### Run individual stages

```bash
python pipeline/run_full.py        # End-to-end
pytest tests/ -v                   # Tests (34 passing)
```

## API Keys Required

All free. Add to `.env`:

| Key | Source |
|-----|--------|
| `NOAA_API_TOKEN` | [NCDC CDO](https://www.ncdc.noaa.gov/cdo-web/token) |
| `EPA_AQS_EMAIL` + `EPA_AQS_KEY` | [EPA AQS](https://aqs.epa.gov/aqsweb/documents/data_api.html) |
| `EIA_API_KEY` | [EIA Open Data](https://www.eia.gov/opendata/register.php) |
| `CENSUS_API_KEY` | [Census Bureau](https://api.census.gov/data/key_signup.html) |

## Architecture

```
ingest/       Fetch and cache raw federal data (no computation)
transform/    Harmonize to county-year, compute derived scores, attribution
score/        Statistical engine: percentiles, weights, penalties, acceleration
api/          FastAPI REST API (placeholder — not yet functional)
dashboard/    React + Tailwind + D3/Recharts (placeholder — not yet functional)
pipeline/     Entry points for each stage
config/       Component definitions, weights, tiers
tests/        Pytest suite
```

### Data Flow

```
data/raw/           ← Ingest output (as-downloaded)
data/harmonized/    ← Transform output (county-year, all components)
data/scored/        ← Score output (CCI scores + variants)
data/validation/    ← Validation output
```

Each layer is independently cacheable. If harmonized data exists, skip ingest. If scored data exists, skip transform.

## Scoring Pipeline

Deterministic 10-step computation. Order is inviolable:

1. Log-transform heavy-tailed variables
2. Winsorize at 99th percentile
3. Percentile rank across scoring universe
4. Center (subtract 50)
5. Overlap penalties (correlation-based, precedence hierarchy, floor 0.2)
6. Acceleration multipliers (Theil-Sen slopes, bounds 0.5–3.0)
7. Missingness handling
8. Weighted component scores
9. Composite sum → S(c)
10. Scale: CCI(c) = 100 + k × S(c)

## Components

| Component | Source | Attribution | Weight |
|-----------|--------|-------------|--------|
| HDD anomaly | NOAA GHCN-Daily | attributed | 0.10 |
| CDD anomaly | NOAA GHCN-Daily | attributed | 0.10 |
| Extreme heat days | NOAA GHCN-Daily | proxy | 0.05 |
| Storm severity | NCEI Storm Events | proxy | 0.12 |
| PM2.5 annual | EPA AQS | proxy | 0.06 |
| AQI unhealthy days | EPA AQS | proxy | 0.05 |
| Flood exposure | FEMA NFHL | proxy | 0.10 |
| Wildfire score | USFS WHP + NCEI | proxy | 0.08 |
| Drought score | USDM | proxy | 0.08 |
| Energy cost attributed | EIA + RECS | attributed | 0.15 |
| Health burden | CDC EPHT | proxy | 0.06 |
| FEMA IA burden | OpenFEMA | proxy | 0.05 |

Weights are derived from BLS Consumer Expenditure Survey budget shares. Only energy has full regression-based causal attribution in v1. All other components are hazard burden proxies. This distinction is maintained in all outputs.

## Data Sources

All public federal data:

- **NOAA NCEI** — GHCN-Daily bulk, Storm Events, 1991–2020 climate normals
- **NOAA HMS** — Hazard Mapping System smoke plume shapefiles
- **EPA AirNow** — AQS air quality monitors
- **FEMA** — National Flood Hazard Layer, Individual/Housing Assistance (OpenFEMA)
- **USFS** — Wildfire Hazard Potential raster
- **USDM** — Drought Monitor weekly classifications
- **EIA** — Electricity prices/consumption, RECS microdata
- **CDC EPHT** — Heat-related ED visits
- **BLS** — Consumer Expenditure Survey, CPI Food indices
- **Census** — ACS (county), block-group housing units, TIGER shapefiles

## Tech Stack

- Python 3.11+, pandas, numpy, scipy, scikit-learn, statsmodels, geopandas
- DuckDB (local storage)

## License

MIT — see [LICENSE](LICENSE)

## Citation

```
Kilpatrick, W. (2026). The Climate Cost Index: Measuring Climate-Linked
Household Cost Pressure Across U.S. Counties. Working Paper.
```
