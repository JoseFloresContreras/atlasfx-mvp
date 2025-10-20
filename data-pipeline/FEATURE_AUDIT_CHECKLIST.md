# Feature Lookahead Bias Audit Checklist

**Purpose**: Systematically audit all features in the data pipeline to ensure no future information is used in feature calculations.

**Critical Importance**: Lookahead bias will cause artificially good backtesting results but catastrophic failure in live trading.

## Audit Process

For each feature, verify:

1. ✅ **Time Window**: Uses only historical data up to current timestamp
2. ✅ **Rolling Calculations**: Window boundaries are strictly in the past
3. ✅ **Shift Operations**: All shifts are backwards (positive values)
4. ✅ **Future Data**: No access to future rows via indexing
5. ✅ **Target Variable**: Feature doesn't depend on what we're trying to predict

## Feature Audit Status

### Aggregator Features (aggregators.py)

| Feature | Status | Notes | Validated By | Date |
|---------|--------|-------|--------------|------|
| `mean` | ⏳ Pending | Calculates mean of mid-price in window | - | - |
| `high` | ⏳ Pending | Max mid-price in window | - | - |
| `low` | ⏳ Pending | Min mid-price in window | - | - |
| `close` | ⏳ Pending | Last mid-price in window | - | - |
| `open` | ⏳ Pending | First mid-price in window | - | - |
| `tick_count` | ✅ Safe | Counts ticks, no future data | - | - |
| `volume` | ⏳ Pending | Sum of volumes in window | - | - |
| `spread` | ⏳ Pending | Mean spread in window | - | - |
| `volatility` | ⏳ Pending | Std of price changes in window | - | - |
| `vwap` | ⏳ Pending | Volume-weighted average price | - | - |
| `ofi` | ⚠️ Review | Order Flow Imbalance - needs careful review | - | - |
| `micro_price` | ⚠️ Review | Microstructure price - needs careful review | - | - |

**Aggregator Assessment**:
- Most aggregators are inherently safe as they aggregate over a **completed** time window
- Key question: Is the window boundary defined correctly?
- Concern: Are windows aligned with when decisions would be made?

**Action Items**:
1. Verify window boundaries don't include "future" relative to decision point
2. Document exact timing of when each aggregated bar is "complete"
3. Add test: If we're at timestamp T, verify no data from T+1 is used

### Featurizer Features (featurizers.py)

#### Price Transform Features

| Feature | Status | Assessment | Validated By | Date |
|---------|--------|------------|--------------|------|
| `log_pct_change` | ⚠️ **HIGH RISK** | Uses `.diff()` - verify shift direction | - | - |

**Concern**: 
```python
result_df[f'{col} | log_pct_change'] = np.log(values).diff()
```
- `.diff()` calculates `current - previous` by default
- This is SAFE if used correctly (uses past value)
- But need to verify it's not inverted

**Validation Needed**:
- [ ] Test: log_pct_change at time T uses data from T and T-1 only
- [ ] Verify: No forward-looking shift

#### Session Features

| Feature | Status | Assessment | Validated By | Date |
|---------|--------|------------|--------------|------|
| `sydney_session` | ✅ Safe | Time-based flag, deterministic | - | - |
| `tokyo_session` | ✅ Safe | Time-based flag, deterministic | - | - |
| `london_session` | ✅ Safe | Time-based flag, deterministic | - | - |
| `ny_session` | ✅ Safe | Time-based flag, deterministic | - | - |

**Assessment**: Session features are safe as they depend only on timestamp, not future prices.

#### Technical Indicators

| Feature | Status | Assessment | Validated By | Date |
|---------|--------|------------|--------------|------|
| `vwap_deviation` | ⏳ Pending | Deviation from VWAP - verify VWAP is not forward-looking | - | - |
| `ofi_rolling_mean` | ⚠️ Review | Rolling mean of OFI - verify window | - | - |
| `atr` | ⚠️ Review | Average True Range - verify pandas_ta implementation | - | - |
| `rsi` | ⚠️ Review | RSI - verify pandas_ta implementation | - | - |
| `bollinger_width` | ⚠️ Review | Bollinger Bands - verify pandas_ta implementation | - | - |

**Concern - pandas_ta**: We're using an external library. Need to verify:
1. pandas_ta doesn't look ahead by default
2. All window calculations use only past data
3. No alignment issues that could cause future leakage

**Validation Needed**:
- [ ] Review pandas_ta source code for each indicator
- [ ] Test: Verify indicators at time T don't use data from T+1
- [ ] Consider: Implement critical indicators natively for full control

#### Volatility Features

| Feature | Status | Assessment | Validated By | Date |
|---------|--------|------------|--------------|------|
| `bipower_variation_features` | ⚠️ **HIGH RISK** | Complex calculation - needs detailed review | - | - |
| `volatility_ratio` | ⚠️ Review | Ratio of two rolling volatilities - verify windows | - | - |
| `return_skewness` | ⚠️ Review | Rolling skewness - verify window | - | - |
| `return_kurtosis` | ⚠️ Review | Rolling kurtosis - verify window | - | - |

**Concern - Rolling Windows**:
```python
# SAFE (looks backward)
df['feature'] = df['price'].rolling(window=20, min_periods=1).mean()

# UNSAFE (if using future data)
df['feature'] = df['price'].rolling(window=20, min_periods=1, center=True).mean()
```

**Validation Needed**:
- [ ] Verify: No `center=True` in any rolling operations
- [ ] Test: Rolling windows at time T use only T and earlier
- [ ] Document: Exact window specification for each feature

#### Microstructure Features

| Feature | Status | Assessment | Validated By | Date |
|---------|--------|------------|--------------|------|
| `csi` | ⚠️ **HIGH RISK** | Cross-Sectional Imbalance - complex, needs review | - | - |
| `micro_price` | ⚠️ **HIGH RISK** | Uses volume imbalances - verify calculation | - | - |

**Concern**: Market microstructure features often involve complex formulas. Need to verify:
1. Order of operations is correct
2. No inadvertent use of future information
3. Volume calculations don't peek ahead

#### Correlation Features

| Feature | Status | Assessment | Validated By | Date |
|---------|--------|------------|--------------|------|
| `pair_correlations` | ⚠️ **HIGH RISK** | Rolling correlations between pairs - verify windows | - | - |

**Concern**:
```python
# Need to verify this doesn't align future data
rolling_corr = pair1.rolling(window).corr(pair2.rolling(window))
```

**Validation Needed**:
- [ ] Test: Correlation at time T uses only data up to T
- [ ] Verify: Both series aligned correctly without forward shift
- [ ] Check: pandas .corr() behavior with time series

#### Temporal Features

| Feature | Status | Assessment | Validated By | Date |
|---------|--------|------------|--------------|------|
| `minute_of_day` | ✅ Safe | Deterministic from timestamp | - | - |
| `day_of_week` | ✅ Safe | Deterministic from timestamp | - | - |

**Assessment**: Temporal features are safe as they're deterministic from the timestamp.

## High-Risk Features Requiring Immediate Audit

### Priority 1 (Critical)
1. **`log_pct_change`** - Verify .diff() doesn't look forward
2. **`bipower_variation_features`** - Complex calculation, high risk
3. **`pair_correlations`** - Correlation alignment issues possible
4. **`csi`** (Cross-Sectional Imbalance) - Complex microstructure calculation
5. **`micro_price`** - Microstructure pricing formula

### Priority 2 (High)
6. **`ofi_rolling_mean`** - Rolling mean verification
7. **`volatility_ratio`** - Two rolling windows need verification
8. **`atr`, `rsi`, `bollinger_width`** - pandas_ta implementation review
9. **`return_skewness`, `return_kurtosis`** - Rolling moment calculations

### Priority 3 (Medium)
10. All aggregator features - Verify window boundaries
11. **`vwap_deviation`** - Depends on VWAP which needs verification

## Validation Tests to Add

### Test 1: Forward Data Inaccessibility
```python
def test_no_forward_data_access(feature_function):
    """
    Test that feature at time T cannot access data from T+1.
    
    Strategy:
    1. Calculate feature on full dataset
    2. Calculate feature on dataset truncated at time T
    3. Verify feature value at T is identical
    """
    full_data = load_test_data()
    truncated_data = full_data.iloc[:split_point]
    
    full_features = feature_function(full_data, config)
    truncated_features = feature_function(truncated_data, config)
    
    # Feature at split_point-1 should be identical
    assert full_features.iloc[split_point-1] == truncated_features.iloc[-1]
```

### Test 2: Rolling Window Boundaries
```python
def test_rolling_window_boundaries(feature_function):
    """
    Test that rolling window features only use historical data.
    
    Strategy:
    1. Create synthetic data with known pattern
    2. Add future shock at time T+1
    3. Verify feature at time T is unaffected
    """
    data = create_synthetic_data()
    data_with_shock = data.copy()
    data_with_shock.iloc[shock_index:] *= 2  # Double all future values
    
    features_before = feature_function(data, config)
    features_with_shock = feature_function(data_with_shock, config)
    
    # Features before shock point should be identical
    assert (features_before.iloc[:shock_index] == 
            features_with_shock.iloc[:shock_index]).all()
```

### Test 3: Shift Direction
```python
def test_diff_shift_direction():
    """
    Test that .diff() calculations are backward-looking.
    """
    data = pd.Series([1, 2, 3, 4, 5])
    diff = data.diff()
    
    # diff[i] should be data[i] - data[i-1]
    # NOT data[i+1] - data[i]
    assert diff.iloc[1] == 1  # 2 - 1
    assert diff.iloc[2] == 1  # 3 - 2
    assert pd.isna(diff.iloc[0])  # No previous value
```

### Test 4: pandas_ta Indicator Validation
```python
def test_pandas_ta_no_lookahead():
    """
    Test that pandas_ta indicators don't look ahead.
    
    Strategy:
    1. Create data with known properties
    2. Calculate indicator
    3. Verify indicator at T doesn't change when future data added
    """
    historical_data = create_test_price_series()
    historical_rsi = ta.rsi(historical_data, length=14)
    
    extended_data = pd.concat([historical_data, future_data])
    extended_rsi = ta.rsi(extended_data, length=14)
    
    # Historical RSI values should not change
    assert (historical_rsi == extended_rsi.iloc[:len(historical_data)]).all()
```

## Documentation Requirements

For each feature, document:

### Required Information
1. **Formula**: Exact mathematical formula
2. **Time Window**: Which timestamps are used
3. **Dependencies**: Which columns/features are inputs
4. **Shift/Lag**: Explicit lag applied
5. **Edge Cases**: Behavior at start of data
6. **Validation**: Test proving no lookahead

### Example Documentation

```markdown
## RSI (Relative Strength Index)

**Formula**: RSI = 100 - (100 / (1 + RS))
where RS = Average Gain / Average Loss over N periods

**Time Window**: 
- Uses prices from [T-N, T]
- Default N = 14 periods

**Dependencies**:
- Input: Close prices
- Calculated: Backward-looking only

**Shift/Lag**:
- No explicit shift needed (rolling calculation is backward)

**Edge Cases**:
- First N-1 values are NaN (insufficient history)

**Validation**:
- Test: test_rsi_no_lookahead (passes)
- Verified: RSI at time T unchanged by future data
```

## Audit Workflow

### Step 1: Code Review
1. Open featurizer source code
2. Trace data flow line by line
3. Identify any forward-looking operations
4. Flag suspicious patterns

### Step 2: Test Creation
1. Write specific test for the feature
2. Test with synthetic data
3. Test with real data subset
4. Verify temporal correctness

### Step 3: Documentation
1. Document formula
2. Document time window
3. Document validation method
4. Add to feature registry

### Step 4: Sign-off
1. Code reviewed: ✅
2. Tests passing: ✅
3. Documentation complete: ✅
4. Reviewer signature: _______
5. Date: _______

## Red Flags to Watch For

### Dangerous Patterns
```python
# ❌ BAD: Center=True looks at future
df['ma'] = df['price'].rolling(20, center=True).mean()

# ❌ BAD: Negative shift looks forward
df['next_price'] = df['price'].shift(-1)

# ❌ BAD: Using future index
df['feature'] = df['price'].iloc[i+1]

# ❌ BAD: Forward-filling from future
df['feature'] = df['price'].shift(-1).fillna(method='bfill')
```

### Safe Patterns
```python
# ✅ GOOD: Default rolling is backward
df['ma'] = df['price'].rolling(20).mean()

# ✅ GOOD: Positive shift looks backward
df['prev_price'] = df['price'].shift(1)

# ✅ GOOD: Using past index only
df['feature'] = df['price'].iloc[:i]

# ✅ GOOD: Forward-fill from past
df['feature'] = df['price'].shift(1).fillna(method='ffill')
```

## Completion Criteria

**Pipeline is validated when**:
- [ ] All features audited (100%)
- [ ] All high-risk features have tests (100%)
- [ ] All tests passing
- [ ] All features documented
- [ ] All features signed off by reviewer
- [ ] Audit report published

**Timeline**: 2-3 weeks for complete audit

## References

- Lopez de Prado (2018) - Chapter 7: Cross-Validation in Finance
- Prado (2018) - Chapter 5: Fractionally Differentiated Features
- Bailey et al. (2014) - The Deflated Sharpe Ratio

---

**Status**: Audit not yet started  
**Next Action**: Begin with Priority 1 features  
**Owner**: TBD  
**Target Completion**: TBD
