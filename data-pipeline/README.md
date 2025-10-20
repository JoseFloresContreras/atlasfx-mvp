# Data Pipeline Documentation

## Overview

The data pipeline processes raw Level 1 tick data from Dukascopy through multiple stages to produce clean, aggregated, and featurized data ready for machine learning models.

## Pipeline Architecture

```
Raw Tick Data (CSV)
    ↓
[1] MERGE - Combine CSV files per symbol
    ↓
[2] CLEAN (ticks) - Detect gaps and clean tick data
    ↓
[3] AGGREGATE - Time-based aggregation (OHLC, VWAP, OFI, etc.)
    ↓
[4] SPLIT - Train/Val/Test split (70/15/15%)
    ↓
[5] WINSORIZE - Outlier handling
    ↓
[6] CLEAN (aggregated) - Clean aggregated data
    ↓
[7] FEATURIZE - Calculate technical indicators and features
    ↓
[8] CLEAN (featurized) - Clean featurized data
    ↓
[9] NORMALIZE - Z-score normalization with clipping
    ↓
[10] VISUALIZE - Generate plots and analysis
    ↓
Final Dataset
```

## Module Descriptions

### Core Modules

#### `pipeline.py`
**Purpose**: Orchestrates the entire pipeline execution  
**Key Functions**:
- `load_pipeline_config()`: Load YAML configuration
- `run_pipeline()`: Execute pipeline steps in order
- `validate_and_order_steps()`: Ensure correct execution order

#### `merge.py`
**Purpose**: Merge multiple CSV files per symbol into single Parquet files  
**Input**: Raw CSV tick data files in folders  
**Output**: `{symbol}-pair_ticks.parquet` or `{symbol}-instrument_ticks.parquet`  
**Key Columns**: timestamp, askPrice, bidPrice, askVolume, bidVolume

#### `clean.py`
**Purpose**: Analyze and clean data, detect gaps  
**Stages**:
- `ticks`: Clean raw tick data
- `aggregated`: Clean after aggregation
- `featurized`: Clean after feature engineering
**Key Functions**:
- `analyze_gaps()`: Detect and report time gaps
- `clean_data()`: Remove NaN, duplicates, outliers

#### `aggregate.py`
**Purpose**: Aggregate tick data into time-based windows  
**Time Windows**: 30S, 1min, 5min, 15min, 30min, 1h, 4h, 1D, 1W  
**Output Columns**: Based on configured aggregators (see aggregators.py)

#### `aggregators.py`
**Purpose**: Individual aggregation functions  
**Available Aggregators**:
- **Basic**: `mean`, `high`, `low`, `close`, `open`
- **Volume**: `tick_count`, `volume`
- **Microstructure**: `vwap` (Volume-Weighted Average Price), `ofi` (Order Flow Imbalance), `micro_price`
- **Market**: `spread`, `volatility`

#### `split.py`
**Purpose**: Split data into train/validation/test sets  
**Method**: Temporal split (preserves time order)  
**Default Ratios**: 70% train, 15% validation, 15% test

#### `winsorize.py`
**Purpose**: Handle outliers using winsorization  
**Method**: Clip extreme values at specified percentiles  
**Configurable**: Per-column percentile thresholds

#### `featurize.py`
**Purpose**: Orchestrate feature engineering  
**Key Functions**:
- `load_featurizers()`: Dynamically load featurizer functions
- `run_featurize()`: Apply featurizers to data

#### `featurizers.py`
**Purpose**: Individual feature calculation functions  
**Available Featurizers**:

**Price Features**:
- `log_pct_change`: Log percentage changes
- `vwap_deviation`: Deviation from VWAP

**Technical Indicators**:
- `rsi`: Relative Strength Index
- `bollinger_width`: Bollinger Band width
- `atr`: Average True Range
- `bipower_variation_features`: Bipower variation measures

**Volatility**:
- `volatility_ratio`: Short/long window volatility ratio
- `return_skewness`: Distribution skewness
- `return_kurtosis`: Distribution kurtosis

**Microstructure**:
- `ofi_rolling_mean`: Order flow imbalance rolling average
- `csi`: Cross-Sectional Imbalance
- `micro_price`: Microstructure-based price

**Correlation**:
- `pair_correlations`: Rolling correlations between currency pairs

**Temporal**:
- `sydney_session`, `tokyo_session`, `london_session`, `ny_session`: Trading session indicators
- `minute_of_day`, `day_of_week`: Time features

#### `normalize.py`
**Purpose**: Normalize features using Z-score normalization  
**Method**: 
1. Calculate mean and std from training set only
2. Apply to train/val/test sets
3. Clip extreme values beyond threshold
**Output**: Normalized data + statistics pickle file

#### `visualize.py`
**Purpose**: Generate visualizations for data analysis  
**Output**: PNG plots in visualization directory

### Utility Modules

#### `logger.py`
**Purpose**: Custom logging utility  
**Features**:
- File and console logging
- Color-coded log levels
- Automatic log directory creation

#### `pipeline.yaml`
**Purpose**: Pipeline configuration file  
**Sections**:
- `steps`: Which steps to execute
- `merge`: Input data paths
- `aggregate`: Time windows and aggregators
- `split`: Train/val/test ratios
- `featurize`: Features to calculate
- `normalize`: Clipping threshold

## Configuration

### Example pipeline.yaml

```yaml
steps: [merge, clean_ticks, aggregate, split, winsorize, clean_aggregated, featurize, clean_featurized, normalize, visualize]

output_directory: data

merge:
  time_column: timestamp
  pairs:
    - symbol: eurusd
      folder_path: data/raw-tick-data/raw data/eurusd
    - symbol: gbpusd
      folder_path: data/raw-tick-data/raw data/gbpusd
  instruments:
    - symbol: xauusd
      folder_path: data/raw-tick-data/raw instruments/xauusd

aggregate:
  time_window: 5min
  output_filename: forex_data
  aggregators:
    - tick_count
    - high
    - low
    - close
    - volume
    - vwap
    - ofi
    - micro_price

split:
  train: 0.7
  val: 0.15
  test: 0.15

featurize:
  featurizers:
    sydney_session:
    tokyo_session:
    london_session:
    ny_session:
    vwap_deviation:
    atr:
      window_size: 14
    rsi:
      window_size: 14
    pair_correlations:
      window_size: 30
      pairs:
        - [eurusd, gbpusd]
        - [eurusd, audusd]

normalize:
  clip_threshold: 4.0
```

## Running the Pipeline

### Basic Usage

```bash
cd data-pipeline
python pipeline.py
```

The pipeline will:
1. Load configuration from `pipeline.yaml`
2. Execute steps in order
3. Save outputs to the configured `output_directory`
4. Log progress to `logs/` directory

### Running Individual Steps

You can configure which steps to run by editing the `steps` list in `pipeline.yaml`:

```yaml
# Run only aggregation onwards
steps: [aggregate, split, winsorize, clean_aggregated, featurize, clean_featurized, normalize]

# Run only feature engineering
steps: [featurize, clean_featurized, normalize]

# Full pipeline
steps: [merge, clean_ticks, aggregate, split, winsorize, clean_aggregated, featurize, clean_featurized, normalize, visualize]
```

## Data Schema

### Raw Tick Data (after merge)
```
timestamp: int64 (Unix milliseconds)
askPrice: float64
bidPrice: float64
askVolume: float64
bidVolume: float64
```

### Aggregated Data (after aggregate)
```
start_time: datetime64[ns, UTC] (index)
tick_count: int64
high: float64
low: float64
close: float64
volume: float64
vwap: float64
ofi: float64
micro_price: float64
{symbol}_high: float64 (for each symbol)
{symbol}_low: float64
{symbol}_close: float64
... (repeated for each symbol)
```

### Featurized Data (after featurize)
All aggregated columns plus:
```
{symbol}_close | log_pct_change: float64
{symbol}_close | vwap_deviation: float64
{symbol}_close | rsi_14: float64
{symbol}_close | bollinger_width_20: float64
{symbol}_close | atr_14: float64
{symbol}_close | volatility_ratio: float64
{symbol}_close | return_skewness_60: float64
{symbol}_close | return_kurtosis_60: float64
{symbol}_close | ofi_rolling_mean_14: float64
{symbol}_close | csi_14: float64
{pair1}_{pair2}_corr_30: float64 (for configured pairs)
sydney_session: int8
tokyo_session: int8
london_session: int8
ny_session: int8
minute_of_day: int16
day_of_week: int8
... (many more features)
```

### Normalized Data (after normalize)
Same columns as featurized data, but all numeric values normalized to approximately N(0, 1) with clipping.

## Output Files

The pipeline generates files in the `output_directory`:

```
data/
├── {symbol}-pair_ticks.parquet              # Merged tick data
├── {symbol}-instrument_ticks.parquet
├── {time_window}_{output_filename}.parquet  # Aggregated data
├── {time_window}_{output_filename}_train.parquet  # Split datasets
├── {time_window}_{output_filename}_val.parquet
├── {time_window}_{output_filename}_test.parquet
├── normalization_stats_{time_window}_{clip}.pkl  # Normalization statistics
└── visualizations/                          # Generated plots
    ├── tick_count_distribution.png
    ├── close_price_timeseries.png
    └── ...
```

## Testing

### Run Tests

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_aggregators.py -v
```

### Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── unit/
│   ├── test_aggregators.py       # Aggregator function tests
│   ├── test_featurizers.py       # Featurizer function tests (TODO)
│   └── test_utilities.py         # Utility function tests (TODO)
└── integration/
    └── test_pipeline.py           # End-to-end pipeline tests (TODO)
```

## Known Issues and Limitations

### Current Issues
1. **Lookahead Bias Risk**: Some features may inadvertently use future information. Needs thorough audit.
2. **Column Naming**: Inconsistent naming between test fixtures and actual data (volume vs askVolume/bidVolume)
3. **Missing Type Hints**: While imports exist, not all function signatures are fully typed
4. **No Data Validation**: No Pydantic schemas or formal data contracts

### Limitations
1. **Memory Usage**: Entire dataset loaded in memory during processing
2. **pandas_ta Dependency**: External library for some indicators, adds dependency risk
3. **No Streaming**: Batch processing only, not suitable for real-time data
4. **Single-threaded**: No parallelization of independent operations

## Improvement Roadmap

### Phase 1: Infrastructure ✅
- [x] Add missing dependencies
- [x] Create pyproject.toml
- [x] Setup test framework
- [x] Initial unit tests

### Phase 2: Code Quality (In Progress)
- [ ] Complete type hints on all functions
- [ ] Add Pydantic schemas for data validation
- [ ] Improve error handling
- [ ] Add comprehensive tests (target: 80% coverage)

### Phase 3: Feature Validation
- [ ] Audit all features for lookahead bias
- [ ] Document each feature calculation
- [ ] Add temporal validation tests
- [ ] Create feature registry

### Phase 4: Optimization
- [ ] Add streaming/batching support
- [ ] Parallelize independent operations
- [ ] Optimize memory usage
- [ ] Profile and optimize bottlenecks

## References

### Academic Papers
- Lopez de Prado (2018) - Advances in Financial Machine Learning
- Cartea et al. (2015) - Algorithmic and High-Frequency Trading
- Easley et al. (2012) - Flow Toxicity and Liquidity

### Code Style
- PEP 8 (Python Style Guide)
- Type hints: PEP 484
- Docstrings: Google Style

## Contributing

When modifying the pipeline:

1. **Add tests** for new features
2. **Document** new functions and parameters
3. **Validate** features for lookahead bias
4. **Update** this README with any changes
5. **Run** full test suite before committing

## Support

For issues or questions:
1. Check existing documentation (AUDIT_REPORT.md, ARCHITECTURE.md)
2. Review test cases for usage examples
3. Check logs in `logs/` directory for runtime errors
4. Consult inline code documentation
