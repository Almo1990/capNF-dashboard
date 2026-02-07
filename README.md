# Membrane Filtration Analytics Pipeline

**PWN CapNF System - Modular Data Analytics Pipeline**

Transform your membrane filtration TSV data into actionable insights with automated fouling detection, cleaning cycle analysis, predictive forecasting, and interactive dashboards.

---

## 🎯 Features

### **8-Stage Modular Pipeline**

```
DataLog.tsv → [1] Ingestion → [2] Validation → [3] Preprocessing 
→ [4] Feature Engineering → [5] Fouling Metrics → [6] Cleaning Analysis 
→ [7] KPI Engine → [8] Dashboard
```

- **Automated Cleaning Detection**: Identifies cleaning cycles via TMP drops + chemical timer activity
- **Fouling Analysis**: TMP slope calculation (global + rolling windows), permeability decline tracking
- **Cycle-by-Cycle Metrics**: Duration, fouling rate, cleaning effectiveness per cycle
- **Predictive Forecasting**: Linear/exponential TMP growth models with time-to-cleaning estimates
- **Alert System**: Configurable thresholds for TMP slope, permeability decline, energy consumption
- **Interactive Dashboards**: Plotly-based HTML visualizations with zoom, pan, and time range selection

---

## 🚀 Quick Start

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2. Place Your Data**

Put your TSV files in the `Data/` folder:
```
Data/
  ├── 2026-01-28_03-30-02 DataLog.tsv
  ├── 2026-01-29_16-08-47 DataLog.tsv
  └── ...
```

### **3. Run the Pipeline**

```bash
python main.py
```

That's it! The pipeline will:
- ✅ Load and combine all TSV files
- ✅ Apply bandwidth filtering (remove outliers)
- ✅ Calculate fouling metrics and detect cleaning cycles
- ✅ Generate KPIs, alerts, and TMP forecasts
- ✅ Create interactive HTML dashboards

**Output:**
- **Parquet files**: `outputs/01_raw.parquet` through `outputs/07_kpis.json`
- **Visualizations**: `combined_data_plots/*.html`
- **Reports**: `outputs/validation_report.json`, `cycles_summary.json`, etc.

---

## 📊 Pipeline Stages

### **[1] Ingestion** (`src/ingestion.py`)
- Discovers and loads all TSV files from `Data/` folder
- Converts timestamps from `"2026-01-28 03:30:04 48"` → datetime
- Saves to `outputs/01_raw.parquet`

### **[2] Validation** (`src/validation.py`)
- Checks required columns (TMP, Permeability TC, Flux, Temperature)
- Applies **bandwidth filters** (configurable in `config.yaml`):
  - TMP: 0-8 bar
  - Flux: 0-30 L/m²/h
  - Recovery: 0-100%
  - Removes outliers, logs rejection statistics
- Detects timestamp gaps >5 min
- Saves to `outputs/02_validated.parquet` + `validation_report.json`

### **[3] Preprocessing** (`src/preprocessing.py`)
- **Gap filling**: Forward-fill or interpolate short gaps (<5 min)
- **Downsampling**: LTTB algorithm (10,000 points for viz)
- **SMA calculation**: Rolling averages (50-point window)
- Saves `03_processed.parquet` (full) and `03_processed_viz.parquet` (downsampled)

### **[4] Feature Engineering** (`src/feature_engineering.py`)
- **Derivatives**: dTMP/dt, dFlux/dt (rate of change)
- **Rolling statistics**: 6h/12h/24h mean & std for TMP, Flux, Recovery
- **Temperature normalization**: Correct for viscosity changes
- **Efficiency metrics**: Hydraulic efficiency, crossflow efficiency
- Saves to `outputs/04_features.parquet` (~20 new features)

### **[5] Fouling Metrics** (`src/fouling_metrics.py`)
- **Global TMP slope**: Linear regression (bar/hour, bar/day, R²)
- **Rolling TMP slopes**: 6h, 12h, 24h windows
- **Fouling classification**: Low/Medium/High/Critical based on slope thresholds
- **Permeability decline**: % decline over 24h window
- Saves to `outputs/05_fouling_metrics.parquet` + `fouling_metadata.json`

### **[6] Cleaning Analysis** (`src/cleaning_analysis.py`)
- **Cleaning detection**:
  - TMP drops >20% within 30 min (configurable)
  - Chemical timer increments
- **Cycle segmentation**: Splits data into filtration cycles between cleanings
- **Per-cycle metrics**:
  - Duration, TMP start/end, TMP slope, permeability decline
  - Average flux, recovery, retention
  - Cleaning effectiveness (% TMP recovery)
- Saves `outputs/06_cycles.parquet` + `06_cycle_summary.parquet` + `cycles_summary.json`

### **[7] KPI Engine** (`src/kpi_engine.py`)
- **KPI calculation**:
  - Average recovery rate, specific energy consumption
  - Average cycle duration, cleaning frequency
  - Current TMP status
- **Alert generation** (configurable thresholds):
  - TMP slope: >0.01 bar/hour (warning), >0.05 (critical)
  - Permeability decline: >10% (warning), >20% (critical)
  - High TMP: >7.5 bar (approaching max)
- **TMP forecasting**:
  - Linear/exponential models
  - 7-day prediction with confidence intervals
  - Time-to-threshold calculation (when will TMP reach 8 bar?)
- Saves `outputs/07_kpis.json`, `07_alerts.json`, `07_forecast.json`

### **[8] Dashboard** (`src/dashboard_app.py`)
- **Individual plots**: Each parameter (TMP, Flux, Recovery, etc.) with:
  - Raw data + SMA overlay
  - Temperature on secondary axis
  - Min/max annotations
  - Interactive zoom/pan, time range selection (1h, 6h, 12h, 1d, 1w, all)
- **TMP forecast plot**: Current data + linear fit + 7-day prediction
- **Cycle comparison**: Bar chart of TMP slope per cycle
- All plots: Interactive HTML with Plotly (rangeslider, rangeselector)

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

### **Bandwidth Filters** (validation)
```yaml
validation:
  bandwidth_filters:
    TMP:
      min: 0
      max: 8  # bar
```

### **Fouling Thresholds**
```yaml
fouling:
  tmp_slope:
    thresholds:
      low: 0.005      # bar/hour
      medium: 0.02
      high: 0.05
```

### **Cleaning Detection**
```yaml
cleaning_detection:
  tmp_drop:
    threshold_percent: 20  # % TMP drop
    window_minutes: 30
```

### **Alert Thresholds**
```yaml
kpis:
  alerts:
    tmp_slope:
      warning: 0.01   # bar/hour
      critical: 0.05
```

### **Visualization Settings**
```yaml
visualization:
  colors: ["#1f77b4", "#2ca02c", "#d62728", ...]
  plot_sizes:
    individual_height: 700
```

---

## 📁 Project Structure

```
CapNF/
├── main.py                     # Pipeline orchestrator
├── config.yaml                 # Configuration file
├── requirements.txt
├── README.md
│
├── Data/                       # Place TSV files here
│   ├── 2026-01-28_DataLog.tsv
│   └── ...
│
├── src/                        # Pipeline modules
│   ├── __init__.py
│   ├── ingestion.py            # [1] Load TSV files
│   ├── validation.py           # [2] Quality checks, filtering
│   ├── preprocessing.py        # [3] Downsampling, SMA
│   ├── feature_engineering.py  # [4] Derivatives, rolling stats
│   ├── fouling_metrics.py      # [5] TMP slopes, fouling analysis
│   ├── cleaning_analysis.py    # [6] Cycle detection, segmentation
│   ├── kpi_engine.py           # [7] KPIs, alerts, forecasts
│   ├── dashboard_app.py        # [8] Visualization
│   │
│   └── utils/                  # Shared utilities
│       ├── data_io.py          # Parquet I/O
│       ├── time_utils.py       # Timestamp conversion
│       ├── plotting.py         # Plotly helpers
│       └── models.py           # Data classes (Cycle, Alert, KPI, Forecast)
│
├── outputs/                    # Pipeline outputs (auto-generated)
│   ├── 01_raw.parquet
│   ├── 02_validated.parquet
│   ├── 03_processed.parquet
│   ├── 03_processed_viz.parquet
│   ├── 04_features.parquet
│   ├── 05_fouling_metrics.parquet
│   ├── 06_cycles.parquet
│   ├── 06_cycle_summary.parquet
│   ├── 07_kpis.json
│   ├── 07_alerts.json
│   ├── 07_forecast.json
│   ├── validation_report.json
│   ├── fouling_metadata.json
│   ├── cycles_summary.json
│   └── pipeline.log
│
└── combined_data_plots/        # Interactive HTML dashboards
    ├── TMP.html                # Individual parameter plots
    ├── Flux.html
    ├── Recovery.html
    ├── TMP_forecast.html       # TMP with prediction
    ├── cycle_comparison.html   # Cycle-by-cycle analysis
    └── ...
```

---

## 🔧 Advanced Usage

### **Run Individual Modules**

```bash
# Run only ingestion
python -m src.ingestion

# Run only fouling metrics
python -m src.fouling_metrics

# Run validation with custom config
python -c "from src.validation import run_validation; run_validation('my_config.yaml')"
```

### **Command-Line Options**

```bash
# Use custom config file
python main.py --config custom_config.yaml

# Skip visualization (faster)
python main.py --skip-viz

# Fast mode (less detailed analysis)
python main.py --fast

# Quiet mode (less logging)
python main.py --quiet
```

### **Load Intermediate Data**

```python
from src.utils.data_io import load_parquet

# Load processed data
df = load_parquet("outputs/05_fouling_metrics.parquet")

# Load cycle summary
cycles_df = load_parquet("outputs/06_cycle_summary.parquet")
```

### **Customize Pipeline Execution**

Edit `config.yaml`:

```yaml
pipeline:
  run_modules:
    ingestion: true
    validation: true
    preprocessing: true
    feature_engineering: false  # Skip this stage
    fouling_metrics: true
    cleaning_analysis: true
    kpi_engine: true
    dashboard: true
```

---

## 📈 Example Outputs

### **Validation Report**
```json
{
  "total_rows_input": 160532,
  "total_rows_output": 158247,
  "rows_removed": 2285,
  "outliers_per_parameter": {
    "TMP": 342,
    "Flux": 1120,
    "Recovery": 823
  },
  "warnings": ["Detected 3 timestamp gaps"]
}
```

### **Fouling Metadata**
```json
{
  "global_tmp_slope_per_hour": 0.003214,
  "global_tmp_slope_per_day": 0.0771,
  "tmp_slope_r_squared": 0.9847,
  "fouling_classification": "low"
}
```

### **Alert Example**
```json
{
  "alert_type": "tmp_slope",
  "severity": "warning",
  "value": 0.0154,
  "threshold": 0.01,
  "message": "WARNING: TMP slope 0.015400 bar/hour exceeds threshold 0.01"
}
```

### **Forecast Example**
```json
{
  "parameter": "TMP",
  "current_value": 3.42,
  "predicted_value": 4.78,
  "forecast_horizon_days": 7.0,
  "time_to_threshold": 42.3,
  "threshold_value": 8.0,
  "message": "TMP forecasted to reach 4.78 bar in 7 days (within safe range)"
}
```

---

## 🛠️ Troubleshooting

### **No TSV files found**
- Ensure TSV files are in `Data/` folder
- Check file extension is `.tsv` (not `.txt` or `.csv`)

### **Missing required columns**
- TSV must have: `TimeStamp`, `TMP`, `Permeability TC`, `Flux`, `01-TIT-01`
- Check column names match exactly (case-sensitive)

### **Import errors**
```bash
pip install --upgrade -r requirements.txt
```

### **Memory issues (large datasets)**
- Increase downsampling in `config.yaml`:
  ```yaml
  preprocessing:
    downsample_limits:
      individual_plots: 5000  # Reduce from 10000
  ```

---

## 📚 Technical Details

- **Data format**: Parquet (compressed, 10-50x faster than CSV)
- **Sampling frequency**: 5 seconds (configurable)
- **Fouling indicator**: TMP slope (bar/hour) via linear regression
- **Cleaning detection**: TMP drop >20% + chemical timer increment
- **Forecasting**: Linear/exponential models with scipy
- **Visualization**: Plotly (interactive HTML, no server required)

---

## 📝 Notes

- **Backward compatible**: Still generates `combined_data_filtered.xlsx` and `combined_data_unfiltered.csv` for legacy workflows
- **Modular design**: Each stage can run independently for testing/debugging
- **Performance**: Parquet + downsampling handles 160k+ rows efficiently
- **Extensible**: Easy to add new features, metrics, or visualizations

---

## 🎓 Citation

**PWN CapNF System**  
Membrane Filtration Analytics Pipeline v1.0  
Developed for nanofiltration performance monitoring and predictive maintenance

---

## 📧 Support

For questions or issues, check:
1. Configuration file (`config.yaml`) for correct settings
2. Log file (`outputs/pipeline.log`) for error details
3. Validation report (`outputs/validation_report.json`) for data quality issues

---

**Happy analyzing! 🚰📊**
