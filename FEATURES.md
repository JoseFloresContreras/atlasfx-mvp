# AtlasFX Feature Engineering Documentation

**Version:** 1.0  
**Last Updated:** October 17, 2025  
**Status:** Audit Phase - Needs Validation

---

## Overview

This document catalogs all features used in the AtlasFX trading system. Each feature is documented with:
- Mathematical formula
- Purpose and intuition
- Implementation details
- Lookahead bias status (✅ Safe, ⚠️ Needs Review, ❌ Has Bias)
- Validation status

---

## Feature Categories

1. [Price-Based Features](#1-price-based-features)
2. [Volume-Based Features](#2-volume-based-features)
3. [Market Microstructure Features](#3-market-microstructure-features)
4. [Technical Indicators](#4-technical-indicators)
5. [Volatility Features](#5-volatility-features)
6. [Session Indicators](#6-session-indicators)
7. [Temporal Features](#7-temporal-features)
8. [Cross-Asset Features](#8-cross-asset-features)

---

## 1. Price-Based Features

### 1.1 Open, High, Low, Close (OHLC)

**Formula:**
```
Open_t = First price in interval [t, t+Δt)
High_t = Max price in interval [t, t+Δt)
Low_t = Min price in interval [t, t+Δt)
Close_t = Last price in interval [t, t+Δt)
```

**Purpose:** Standard price aggregation for time-series analysis.

**Lookahead Status:** ✅ Safe  
**Implementation:** `aggregate.py` (line ~150-200)  
**Notes:** Computed from tick data within non-overlapping windows.

---

### 1.2 Mid Price

**Formula:**
```
Mid_t = (Bid_t + Ask_t) / 2
```

**Purpose:** Fair value estimate between bid and ask.

**Lookahead Status:** ✅ Safe  
**Implementation:** Implicit in tick data processing  
**Notes:** Used as base for other calculations.

---

### 1.3 Log Price Change

**Formula:**
```
LogPctChange_t = log(Close_t / Close_{t-1})
```

**Purpose:** Normalized returns for statistical modeling.

**Lookahead Status:** ✅ Safe  
**Implementation:** `featurizers.py::log_pct_change()` (line 22-53)  
**Notes:** Handles zero values by replacing with NaN.

---

## 2. Volume-Based Features

### 2.1 Volume

**Formula:**
```
Volume_t = Σ (trade_size_i) for all trades in [t, t+Δt)
```

**Purpose:** Trading activity indicator.

**Lookahead Status:** ✅ Safe  
**Implementation:** `aggregate.py::volume_aggregator()`  
**Notes:** Summed from tick-level volumes.

---

### 2.2 VWAP (Volume-Weighted Average Price)

**Formula:**
```
VWAP_t = (Σ price_i × volume_i) / Σ volume_i
```

**Purpose:** Average price weighted by trading volume; measures true average execution price.

**Lookahead Status:** ✅ Safe  
**Implementation:** `aggregate.py::vwap_aggregator()`  
**Notes:** Calculated within each aggregation window.

---

### 2.3 VWAP Deviation

**Formula:**
```
VWAPDeviation_t = (Close_t - VWAP_t) / VWAP_t
```

**Purpose:** Measures price distance from volume-weighted average; mean reversion indicator.

**Lookahead Status:** ✅ Safe  
**Implementation:** `featurizers.py::vwap_deviation()` (line ~90)  
**Notes:** Normalized by VWAP to be scale-invariant.

---

## 3. Market Microstructure Features

### 3.1 Spread

**Formula:**
```
Spread_t = Ask_t - Bid_t
RelativeSpread_t = (Ask_t - Bid_t) / Mid_t
```

**Purpose:** Liquidity indicator; high spreads suggest low liquidity or high volatility.

**Lookahead Status:** ✅ Safe  
**Implementation:** `aggregate.py::spread_aggregator()`  
**Notes:** Aggregated as mean spread within window.

---

### 3.2 Micro Price

**Formula:**
```
MicroPrice_t = (Bid_t × AskSize_t + Ask_t × BidSize_t) / (BidSize_t + AskSize_t)
```

**Purpose:** Order book imbalance-adjusted fair price; predicts short-term price movement.

**Lookahead Status:** ✅ Safe (if sizes are simultaneous)  
**Implementation:** `aggregate.py::micro_price_aggregator()`  
**Notes:** ⚠️ Verify that bid/ask sizes are aligned with prices (no lookahead).

---

### 3.3 OFI (Order Flow Imbalance)

**Formula:**
```
OFI_t = Σ (BidSize_i - AskSize_i) for all updates in [t, t+Δt)
```

**Purpose:** Measures buying vs. selling pressure; predictive of short-term price direction.

**Lookahead Status:** ✅ Safe (if properly windowed)  
**Implementation:** `aggregate.py::ofi_aggregator()`  
**Notes:** ⚠️ Critical to verify no future information is used.

---

### 3.4 Tick Count

**Formula:**
```
TickCount_t = Number of ticks (price updates) in [t, t+Δt)
```

**Purpose:** Activity level indicator; high tick count suggests increased market attention.

**Lookahead Status:** ✅ Safe  
**Implementation:** `aggregate.py::tick_count_aggregator()`  
**Notes:** Simple count, no lookahead risk.

---

## 4. Technical Indicators

### 4.1 RSI (Relative Strength Index)

**Formula:**
```
RS = AvgGain_n / AvgLoss_n
RSI = 100 - (100 / (1 + RS))

where:
AvgGain_n = SMA(max(PriceChange, 0), n)
AvgLoss_n = SMA(max(-PriceChange, 0), n)
```

**Purpose:** Momentum oscillator; indicates overbought (>70) or oversold (<30) conditions.

**Lookahead Status:** ✅ Safe (uses rolling window)  
**Implementation:** `featurizers.py::rsi()` (line ~250)  
**Notes:** Window size = 14 periods (standard).

---

### 4.2 Bollinger Bands Width

**Formula:**
```
BB_Width_t = (UpperBand_t - LowerBand_t) / MiddleBand_t

where:
MiddleBand = SMA(Close, n)
UpperBand = MiddleBand + k × StdDev(Close, n)
LowerBand = MiddleBand - k × StdDev(Close, n)
```

**Purpose:** Volatility indicator; wide bands suggest high volatility, narrow bands suggest low volatility.

**Lookahead Status:** ✅ Safe (uses rolling window)  
**Implementation:** `featurizers.py::bollinger_width()` (line ~280)  
**Notes:** Window size = 20, k = 2 (standard).

---

### 4.3 ATR (Average True Range)

**Formula:**
```
TrueRange_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)
ATR_t = SMA(TrueRange, n)
```

**Purpose:** Volatility measure; accounts for gaps between periods.

**Lookahead Status:** ✅ Safe (uses rolling window)  
**Implementation:** `featurizers.py::atr()` (line ~200)  
**Notes:** Window size = 14 (standard).

---

## 5. Volatility Features

### 5.1 Realized Volatility

**Formula:**
```
RealizedVol_t = sqrt(Σ (log_return_i)^2) for i in [t-n, t]
```

**Purpose:** Historical volatility estimate.

**Lookahead Status:** ✅ Safe (backward-looking)  
**Implementation:** `aggregate.py::volatility_aggregator()`  
**Notes:** Standard deviation of log returns within window.

---

### 5.2 Bipower Variation

**Formula:**
```
BV_t = (π/2) × Σ |r_i| × |r_{i-1}| for i in [t-n, t]
```

**Purpose:** Robust volatility estimate; less sensitive to jumps than squared returns.

**Lookahead Status:** ✅ Safe (uses past returns only)  
**Implementation:** `featurizers.py::bipower_variation_features()` (line ~320)  
**Notes:** Window size = 30 (configurable).

---

### 5.3 Volatility Ratio

**Formula:**
```
VolRatio_t = RealizedVol(short_window) / RealizedVol(long_window)
```

**Purpose:** Detects regime changes (increasing or decreasing volatility).

**Lookahead Status:** ✅ Safe (uses rolling windows)  
**Implementation:** `featurizers.py::volatility_ratio()` (line ~350)  
**Notes:** Short window = 14, long window = 60.

---

### 5.4 Return Skewness & Kurtosis

**Formula:**
```
Skewness_t = E[(r - μ)^3] / σ^3
Kurtosis_t = E[(r - μ)^4] / σ^4
```

**Purpose:** Detects asymmetry (skewness) and tail risk (kurtosis) in return distribution.

**Lookahead Status:** ✅ Safe (uses rolling window)  
**Implementation:** `featurizers.py::return_skewness()`, `return_kurtosis()` (line ~380-420)  
**Notes:** Window size = 60.

---

## 6. Session Indicators

### 6.1 Trading Session Flags

**Formula:**
```
Sydney:  Active when local time in Sydney is [7:00, 16:00)
Tokyo:   Active when local time in Tokyo is [9:00, 15:00)
London:  Active when local time in London is [8:00, 16:30)
NY:      Active when local time in New York is [8:00, 17:00)
```

**Purpose:** Capture session-specific volatility and liquidity patterns.

**Lookahead Status:** ✅ Safe (based on timestamp only)  
**Implementation:** `featurizers.py::sydney_session()`, `tokyo_session()`, etc. (line 55-150)  
**Notes:** Timezone conversions from UTC; excludes weekends.

---

## 7. Temporal Features

### 7.1 Minute of Day

**Formula:**
```
MinuteOfDay_t = (Hour_t × 60 + Minute_t) / 1440
```

**Purpose:** Cyclical time feature; captures intraday patterns.

**Lookahead Status:** ✅ Safe (derived from timestamp)  
**Implementation:** `featurizers.py::minute_of_day()` (line ~450)  
**Notes:** Normalized to [0, 1]; may use sin/cos encoding for cyclical nature.

---

### 7.2 Day of Week

**Formula:**
```
DayOfWeek_t = [Monday, Tuesday, ..., Friday] (one-hot encoded)
```

**Purpose:** Captures day-specific patterns (e.g., Monday reversals, Friday profit-taking).

**Lookahead Status:** ✅ Safe (derived from timestamp)  
**Implementation:** `featurizers.py::day_of_week()` (line ~470)  
**Notes:** One-hot encoded (5 dimensions for Mon-Fri).

---

## 8. Cross-Asset Features

### 8.1 CSI (Cross-Sectional Imbalance)

**Formula:**
```
CSI_t(pair) = (OFI_t(pair) - mean(OFI_t)) / std(OFI_t)
```

**Purpose:** Relative imbalance of one pair compared to all pairs; identifies which pairs are seeing disproportionate buying/selling.

**Lookahead Status:** ⚠️ **Needs Review**  
**Implementation:** `featurizers.py::csi()` (line ~500)  
**Concern:** If mean/std are computed across all pairs at time t, this may introduce subtle lookahead (parallel information). Need to verify calculation uses only past data.

---

### 8.2 Pair Correlations

**Formula:**
```
Corr_t(pair1, pair2) = Pearson(returns_pair1[t-n:t], returns_pair2[t-n:t])
```

**Purpose:** Rolling correlation between currency pairs; captures regime shifts (risk-on vs. risk-off).

**Lookahead Status:** ✅ Safe (uses rolling window)  
**Implementation:** `featurizers.py::pair_correlations()` (line ~550)  
**Notes:** Window size = 30; computed for predefined pairs (e.g., EURUSD-GBPUSD).

---

### 8.3 OFI Rolling Mean

**Formula:**
```
OFI_MA_t = SMA(OFI, n)
```

**Purpose:** Smoothed order flow to reduce noise.

**Lookahead Status:** ✅ Safe (uses rolling window)  
**Implementation:** `featurizers.py::ofi_rolling_mean()` (line ~600)  
**Notes:** Window size = 14.

---

## Feature Summary Table

| Feature Name | Category | Lookahead Status | Window Size | Priority |
|--------------|----------|------------------|-------------|----------|
| OHLC | Price | ✅ Safe | N/A | High |
| Mid Price | Price | ✅ Safe | N/A | High |
| Log Pct Change | Price | ✅ Safe | 1 | High |
| Volume | Volume | ✅ Safe | N/A | High |
| VWAP | Volume | ✅ Safe | Agg window | High |
| VWAP Deviation | Volume | ✅ Safe | Agg window | Medium |
| Spread | Microstructure | ✅ Safe | Agg window | High |
| Micro Price | Microstructure | ⚠️ Needs Review | N/A | High |
| OFI | Microstructure | ⚠️ Needs Review | Agg window | High |
| Tick Count | Microstructure | ✅ Safe | Agg window | Medium |
| RSI | Technical | ✅ Safe | 14 | Medium |
| Bollinger Width | Technical | ✅ Safe | 20 | Medium |
| ATR | Technical | ✅ Safe | 14 | High |
| Realized Vol | Volatility | ✅ Safe | Agg window | High |
| Bipower Variation | Volatility | ✅ Safe | 30 | Medium |
| Volatility Ratio | Volatility | ✅ Safe | 14/60 | Medium |
| Return Skew/Kurt | Volatility | ✅ Safe | 60 | Low |
| Session Flags | Session | ✅ Safe | N/A | High |
| Minute of Day | Temporal | ✅ Safe | N/A | Medium |
| Day of Week | Temporal | ✅ Safe | N/A | Medium |
| CSI | Cross-Asset | ⚠️ Needs Review | 14 | Medium |
| Pair Correlations | Cross-Asset | ✅ Safe | 30 | High |
| OFI Rolling Mean | Microstructure | ✅ Safe | 14 | Medium |

---

## Validation Checklist

### Critical Features (Must Validate)

- [ ] **OFI:** Verify order book updates are properly sequenced
- [ ] **Micro Price:** Confirm bid/ask sizes match timestamps
- [ ] **CSI:** Review cross-sectional mean/std calculation (potential lookahead)
- [ ] **VWAP:** Confirm aggregation window boundaries
- [ ] **Session Flags:** Verify timezone conversions and weekend handling

### Feature Quality Checks

- [ ] No NaN values in first k rows (where k = max window size)
- [ ] All features normalized to similar scales (post-normalization)
- [ ] No infinite values or extreme outliers (post-winsorization)
- [ ] Feature distributions stable across train/val/test splits
- [ ] Rolling windows respect temporal boundaries (no future data)

### Statistical Validation

- [ ] Stationarity tests (ADF) for all features
- [ ] Correlation matrix heatmap (identify redundant features)
- [ ] Feature importance from baseline model (Random Forest)
- [ ] Mutual information analysis (feature-target relationships)

---

## Feature Engineering Best Practices

### 1. Temporal Consistency

**Rule:** All rolling windows must use `closed='left'` to exclude the current timestamp.

**Example:**
```python
# ✅ Correct
df['feature'] = df['value'].rolling(window=14, closed='left').mean()

# ❌ Wrong (includes current timestamp)
df['feature'] = df['value'].rolling(window=14, closed='both').mean()
```

### 2. Cross-Sectional Features

**Rule:** When computing cross-sectional statistics (mean, std across assets), ensure all assets are aligned at the same timestamp.

**Example:**
```python
# ✅ Correct
def compute_csi(df, timestamp):
    # Use only data up to (but not including) timestamp
    past_data = df[df['timestamp'] < timestamp]
    cross_mean = past_data.groupby('pair')['ofi'].last().mean()
    return (current_ofi - cross_mean) / cross_std
```

### 3. Lag Features

**Rule:** When using lagged values, explicitly shift to avoid off-by-one errors.

**Example:**
```python
# ✅ Correct
df['return_lag1'] = df['return'].shift(1)

# ❌ Wrong (no shift)
df['return_lag1'] = df['return']
```

### 4. Testing for Lookahead Bias

**Method:** Implement the "information timeline" test:

1. For each feature at time t, list all data points used
2. Verify all data points have timestamps < t
3. For rolling windows, verify window ends at t-1, not t

---

## Feature Roadmap

### Phase 1: MVP (Current)
- Use all existing features (post-validation)
- Add missing dependencies (pandas_ta)
- Fix any identified lookahead bias

### Phase 2: Enhancement
- [ ] Alternative Data: Economic calendar events (NFP, FOMC)
- [ ] Sentiment: News sentiment scores (FinBERT)
- [ ] Order Book Depth: L2 data features (if available)
- [ ] Volatility Forecasts: GARCH predictions

### Phase 3: Advanced
- [ ] Graph Neural Networks: Currency network topology
- [ ] Autoencoders: Learned features from raw ticks
- [ ] Attention Weights: From TFT as features for SAC

---

## Feature Selection Strategy

### Baseline Features (Must Have)
1. OHLC (Close used for returns)
2. Volume, VWAP
3. OFI, Micro Price
4. ATR (volatility)
5. Session flags
6. Temporal features

### Secondary Features (Nice to Have)
7. RSI, Bollinger Width
8. Bipower variation
9. Pair correlations
10. Return moments (skew, kurt)

### Experimental Features (Test Impact)
11. CSI (if validated)
12. Volatility ratio
13. OFI rolling mean

**Selection Method:**
1. Train baseline model with only "Must Have" features
2. Add "Nice to Have" features one-by-one and measure validation performance
3. Keep features that improve Sharpe ratio by ≥5%
4. Test "Experimental" features last

---

## References

**Market Microstructure:**
- Easley et al. (2012) - "Flow Toxicity and Liquidity in a High-Frequency World"
- Cont et al. (2014) - "The Price Impact of Order Book Events"

**Technical Indicators:**
- Wilder (1978) - "New Concepts in Technical Trading Systems" (RSI)
- Bollinger (2002) - "Bollinger on Bollinger Bands"

**Volatility:**
- Andersen & Bollerslev (1998) - "Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts"
- Barndorff-Nielsen & Shephard (2004) - "Power and Bipower Variation with Stochastic Volatility and Jumps"

---

**Document Status:** Audit Phase - Requires validation of flagged features  
**Next Action:** Implement automated lookahead bias tests for all features
