# Architectural Decision Records (ADR)

This document records key architectural and design decisions made during the development of AtlasFX. Each decision includes context, rationale, alternatives considered, and consequences.

## Table of Contents

1. [ADR-001: Three-Component Architecture (VAE + TFT + SAC)](#adr-001-three-component-architecture-vae--tft--sac)
2. [ADR-002: SAC over TD3/PPO for Trading](#adr-002-sac-over-td3ppo-for-trading)
3. [ADR-003: Level 1 Tick Data from Dukascopy](#adr-003-level-1-tick-data-from-dukascopy)
4. [ADR-004: Temporal Train/Val/Test Split](#adr-004-temporal-trainvaltest-split)
5. [ADR-005: Multi-Output Aggregator Design](#adr-005-multi-output-aggregator-design)
6. [ADR-006: Type-Safe Configuration with YAML + Pydantic](#adr-006-type-safe-configuration-with-yaml--pydantic)
7. [ADR-007: Reproducibility via Fixed Seeds](#adr-007-reproducibility-via-fixed-seeds)
8. [ADR-008: Test Coverage Targets](#adr-008-test-coverage-targets)
9. [ADR-009: Monorepo Structure](#adr-009-monorepo-structure)
10. [ADR-010: Pre-commit Hooks for Code Quality](#adr-010-pre-commit-hooks-for-code-quality)

---

## ADR-001: Three-Component Architecture (VAE + TFT + SAC)

**Status**: Accepted  
**Date**: 2025-10-17  
**Deciders**: Jose Flores Contreras

### Context

We need a robust architecture for short-term (1-10 minute) forex trading that:
- Learns compact market state representations
- Provides probabilistic multi-horizon forecasts
- Makes stochastic trading decisions with entropy regularization

### Decision

Adopt a three-stage pipeline:
1. **VAE (Variational Autoencoder)**: Compresses high-dimensional features into low-dimensional latent states
2. **TFT (Temporal Fusion Transformer)**: Uses latent states + temporal covariates for multi-horizon forecasting
3. **SAC (Soft Actor-Critic)**: Uses forecasts + latent states as observations for trading policy

### Rationale

- **VAE**: Provides dimensionality reduction while preserving information via reconstruction loss. β-VAE encourages disentangled representations.
- **TFT**: State-of-the-art for time-series forecasting, handles variable-length sequences, provides uncertainty estimates via quantile regression.
- **SAC**: Maximum entropy RL is robust to hyperparameters, encourages exploration, and naturally handles stochastic policies needed for trading.

### Alternatives Considered

1. **End-to-end RL (skip VAE/TFT)**: Less interpretable, harder to debug, no explicit forecasting
2. **LSTM instead of TFT**: Lower capacity, no attention mechanism, no built-in uncertainty quantification
3. **TD3/PPO instead of SAC**: Less robust to hyperparameters, TD3 is deterministic, PPO is on-policy (sample inefficient)

### Consequences

**Positive**:
- Modular design: Can train and evaluate each component independently
- Interpretability: Can visualize latent states, forecasts, and policy decisions
- Flexibility: Can replace components without full system redesign

**Negative**:
- Increased complexity: Three models to train and tune
- Sequential dependencies: VAE must be trained before TFT, TFT before SAC
- Potential error accumulation across stages

### References

- VAE: Kingma & Welling (2014) - Auto-Encoding Variational Bayes
- TFT: Lim et al. (2021) - Temporal Fusion Transformers
- SAC: Haarnoja et al. (2018) - Soft Actor-Critic

---

## ADR-002: SAC over TD3/PPO for Trading

**Status**: Accepted  
**Date**: 2025-10-17  
**Deciders**: Jose Flores Contreras

### Context

We need to select an RL algorithm for trading decisions. Requirements:
- Sample efficiency (expensive data)
- Stochastic policy (handle uncertainty)
- Robust to hyperparameters
- Off-policy (can use replay buffer)

### Decision

Use **Soft Actor-Critic (SAC)** as the primary RL algorithm.

### Rationale

**Why SAC?**
1. **Maximum Entropy Framework**: Encourages exploration while maximizing reward
2. **Stochastic Policy**: Naturally handles uncertainty in trading decisions
3. **Off-Policy**: Sample efficient, can reuse past data
4. **Twin Q-Functions**: Reduces overestimation bias (like TD3)
5. **Automatic Temperature Tuning**: Self-adjusts exploration vs. exploitation
6. **Robust**: Less sensitive to hyperparameters than TD3/PPO

**Why not TD3?**
- Deterministic policy: Less suitable for stochastic trading environments
- Manual exploration noise tuning required
- Less robust to hyperparameters

**Why not PPO?**
- On-policy: Sample inefficient, discards data after each update
- Requires careful clipping hyperparameter tuning
- Harder to debug when training fails

### Alternatives Considered

1. **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**:
   - Pros: Stable, good for continuous control
   - Cons: Deterministic policy, requires manual exploration noise

2. **PPO (Proximal Policy Optimization)**:
   - Pros: Simple, widely used
   - Cons: On-policy (sample inefficient), harder to tune

3. **DQN variants**:
   - Pros: Simple, well-understood
   - Cons: Discrete actions, less suitable for position sizing

### Consequences

**Positive**:
- Stochastic policy matches trading reality
- Sample efficient training
- Automatic exploration tuning
- Can benchmark against TD3/PPO later

**Negative**:
- More complex than vanilla policy gradient
- Requires tuning temperature parameter (though auto-tuned version exists)
- Needs careful implementation of twin Q-networks

### Implementation Notes

- Use automatic entropy tuning (no manual temperature)
- Twin Q-networks with target networks
- Replay buffer size: 1M transitions
- Batch size: 256
- Update frequency: 1 (update every step)

### References

- Haarnoja et al. (2018) - Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
- Haarnoja et al. (2018) - Soft Actor-Critic Algorithms and Applications

---

## ADR-003: Level 1 Tick Data from Dukascopy

**Status**: Accepted  
**Date**: 2025-10-17  
**Deciders**: Jose Flores Contreras

### Context

We need high-frequency market data for short-term trading (1-10 minute horizons).

### Decision

Use **Level 1 tick data from Dukascopy** as the primary data source.

### Rationale

**Dukascopy Level 1 Data**:
- Tick-by-tick bid/ask/volume for forex pairs
- Historical data available (years of history)
- Free access via Dukascopy API
- Sufficient granularity for 1-10 minute aggregations

**Level 1 vs. Level 2**:
- Level 1: Best bid/ask + volume (sufficient for our use case)
- Level 2: Full order book (overkill, expensive, harder to process)

### Alternatives Considered

1. **OANDA**: Good API, but limited historical data
2. **Interactive Brokers**: Full order book, but expensive
3. **Yahoo Finance**: Free, but only daily/minute bars (no tick data)
4. **Alpaca**: US stocks only, not forex

### Consequences

**Positive**:
- Free, high-quality data
- Historical data for backtesting
- Tick granularity for custom aggregations

**Negative**:
- No Level 2 data (full order book)
- Requires preprocessing (gaps, outliers)
- Limited to Dukascopy instruments

### Data Schema

```yaml
timestamp: datetime64[ns]  # UTC timestamp
bid: float64               # Bid price
ask: float64               # Ask price
volume: float64            # Tick volume (optional)
```

**Invariants**:
- `timestamp` is monotonically increasing
- `bid <= ask` (no crossed spreads)
- `bid > 0 and ask > 0` (no negative prices)

### References

- Dukascopy Data Feed: https://www.dukascopy.com/swiss/english/marketwatch/historical/

---

## ADR-004: Temporal Train/Val/Test Split

**Status**: Accepted  
**Date**: 2025-10-17  
**Deciders**: Jose Flores Contreras

### Context

Time-series data has temporal dependencies. Random splits cause **data leakage** (future data used for training).

### Decision

Use **temporal splits** with no shuffling:
- **Train**: 70% (earliest data)
- **Validation**: 15% (middle period)
- **Test**: 15% (most recent data)

### Rationale

**Why Temporal Split?**
1. **No Data Leakage**: Model never sees future data during training
2. **Realistic Evaluation**: Test set simulates real deployment (predicting unseen future)
3. **Stationarity Check**: If test performance degrades, indicates distribution shift

**Why 70/15/15?**
- Standard split ratios for ML
- Sufficient training data (70%)
- Large enough validation set for hyperparameter tuning (15%)
- Test set matches realistic deployment window (15%)

### Alternatives Considered

1. **Random Split**: ❌ Causes data leakage (future→past)
2. **K-Fold CV**: ❌ Not suitable for time-series (breaks temporal order)
3. **Walk-Forward Split**: ✅ Better, but more complex (reserve for Phase 2)

### Consequences

**Positive**:
- No data leakage
- Realistic evaluation
- Simple to implement

**Negative**:
- Assumes stationarity across train/val/test periods
- Distribution shift can cause performance drop
- Cannot use walk-forward analysis (yet)

### Implementation

```python
def temporal_split(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (no shuffling).
    
    Args:
        data: DataFrame with DatetimeIndex
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
    
    Returns:
        (train, val, test) dataframes
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return (
        data.iloc[:train_end],
        data.iloc[train_end:val_end],
        data.iloc[val_end:],
    )
```

### References

- Lopez de Prado (2018) - Advances in Financial Machine Learning, Ch. 7

---

## ADR-005: Multi-Output Aggregator Design

**Status**: Accepted  
**Date**: 2025-10-17  
**Deciders**: Jose Flores Contreras

### Context

Time-series aggregation functions need to compute multiple metrics per window (e.g., OHLC, VWAP, OFI).

### Decision

Aggregator functions return **dictionaries** with multiple outputs:

```python
def aggregator(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> dict[str, float]:
    """
    Compute aggregated metrics for a time window.
    
    Returns:
        dict: {metric_name: value}
    """
    return {
        'open': data['mid_price'].iloc[0],
        'high': data['mid_price'].max(),
        'low': data['mid_price'].min(),
        'close': data['mid_price'].iloc[-1],
    }
```

### Rationale

**Why Dict[str, float]?**
1. **Flexibility**: Single function can compute multiple related metrics
2. **Efficiency**: One pass through data for multiple metrics
3. **Extensibility**: Easy to add new outputs without changing signature
4. **Type Safety**: Clear contract (str keys, float values)

**Why not Tuple?**
- ❌ No named outputs (positional args are fragile)
- ❌ Hard to extend (need to update all callers)

**Why not Dataclass?**
- ❌ Different aggregators have different outputs
- ❌ Harder to make generic

### Alternatives Considered

1. **Single output per function**: ❌ Inefficient (multiple passes)
2. **Tuple of floats**: ❌ No named outputs
3. **Dataclass**: ❌ Hard to make generic

### Consequences

**Positive**:
- Efficient (one pass for multiple metrics)
- Flexible (easy to extend)
- Type-safe

**Negative**:
- Slightly more verbose than single return value
- Callers must know dictionary keys

### Implementation

```python
# Aggregator function
def ohlc(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> dict[str, float]:
    if data.empty:
        return {'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan}
    
    mid = (data['bid'] + data['ask']) / 2
    return {
        'open': mid.iloc[0],
        'high': mid.max(),
        'low': mid.min(),
        'close': mid.iloc[-1],
    }

# Usage
result = ohlc(start_time, duration, window_data)
open_price = result['open']
high_price = result['high']
```

---

## ADR-006: Type-Safe Configuration with YAML + Pydantic

**Status**: Accepted  
**Date**: 2025-10-17  
**Deciders**: Jose Flores Contreras

### Context

Configuration files need to be:
- Human-readable (for editing)
- Machine-parseable (for code)
- Type-safe (catch errors early)
- Validated (enforce constraints)

### Decision

Use **YAML for configuration files** + **Pydantic for validation**.

### Rationale

**YAML**:
- Human-readable (better than JSON)
- Supports comments (better than JSON)
- Native types (int, float, bool, list, dict)

**Pydantic**:
- Type validation at runtime
- Auto-generates schemas
- Great error messages
- Integrates with FastAPI, PyTorch Lightning, etc.

### Alternatives Considered

1. **JSON**: ❌ No comments, less readable
2. **TOML**: ✅ Good, but less widespread than YAML
3. **Python files**: ❌ Security risk (code execution)
4. **Hydra (by Facebook)**: ❌ Overkill for MVP

### Consequences

**Positive**:
- Type-safe configuration
- Catch errors early (before training)
- Great error messages
- Easy to extend

**Negative**:
- Requires Pydantic schemas (more boilerplate)
- YAML parsing can be slow (but negligible)

### Implementation

```python
# configs/schema.yaml
from pydantic import BaseModel, Field, validator


class TickDataSchema(BaseModel):
    """Schema for Level 1 tick data."""
    
    timestamp: datetime
    bid: float = Field(gt=0, description="Bid price (must be positive)")
    ask: float = Field(gt=0, description="Ask price (must be positive)")
    volume: float = Field(ge=0, description="Volume (non-negative)")
    
    @validator('ask')
    def ask_must_be_gte_bid(cls, v, values):
        """Ensure ask >= bid (no crossed spreads)."""
        if 'bid' in values and v < values['bid']:
            raise ValueError(f"Ask ({v}) must be >= bid ({values['bid']})")
        return v


# Usage
import yaml
from pydantic import ValidationError

with open('configs/data_pipeline.yaml') as f:
    config_dict = yaml.safe_load(f)

try:
    config = TickDataSchema(**config_dict)
except ValidationError as e:
    print(e.json())
```

### References

- Pydantic: https://pydantic-docs.helpmanual.io/

---

## ADR-007: Reproducibility via Fixed Seeds

**Status**: Accepted  
**Date**: 2025-10-17  
**Deciders**: Jose Flores Contreras

### Context

ML experiments must be reproducible for scientific rigor and debugging.

### Decision

Fix random seeds for all sources of randomness:
- Python's `random` module
- NumPy's `np.random`
- PyTorch's `torch.manual_seed`
- CUDA's `torch.cuda.manual_seed_all`

### Rationale

**Why Fixed Seeds?**
1. **Reproducibility**: Same code + same data = same results
2. **Debugging**: Can reproduce bugs reliably
3. **Fair Comparisons**: Ensure differences are due to algorithm, not randomness

### Alternatives Considered

1. **No seed control**: ❌ Non-reproducible, hard to debug
2. **Partial seed control**: ❌ Still non-reproducible (hidden randomness)
3. **Record seeds**: ✅ Good for production, but manual

### Consequences

**Positive**:
- Reproducible experiments
- Easier debugging
- Fair comparisons

**Negative**:
- May hide sensitivity to initialization
- Can give false confidence in results

### Implementation

```python
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make CUDA deterministic (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Usage in training script
set_seed(42)  # Call before any randomness
```

### Notes

- Use different seeds for train/val/test splits
- Document seed in experiment configs
- For production, use random seeds + record them

### References

- PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html

---

## ADR-008: Test Coverage Targets

**Status**: Accepted  
**Date**: 2025-10-17  
**Deciders**: Jose Flores Contreras

### Context

Testing is critical for production-grade code, but 100% coverage is impractical.

### Decision

Set differentiated coverage targets:
- **Global**: ≥70% (all code)
- **Core modules** (data, models, training): ≥85%
- **Utilities**: ≥60%
- **Scripts**: Not required (but encouraged)

### Rationale

**Why 70% Global?**
- Industry standard for good coverage
- High enough to catch most bugs
- Low enough to be achievable

**Why 85% Core?**
- Core modules are business-critical
- Higher stakes (trading decisions depend on them)
- More complex logic (more bugs)

**Why 60% Utils?**
- Utilities are simpler (less critical)
- Often just wrappers around libraries

**Why Not 100%?**
- Diminishing returns (last 10% is expensive)
- Some code is hard to test (e.g., GPU code, plotting)

### Alternatives Considered

1. **100% coverage**: ❌ Impractical, expensive
2. **No targets**: ❌ No accountability
3. **Uniform 80%**: ❌ One-size-fits-all doesn't fit

### Consequences

**Positive**:
- Clear targets for developers
- Prioritizes critical code
- Achievable within timeline

**Negative**:
- May encourage "coverage gaming" (trivial tests)
- Doesn't guarantee test quality

### Implementation

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = [
    "--cov=src/atlasfx",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=70",  # Fail if global coverage < 70%
]
```

### Enforcement

- CI fails if global coverage < 70%
- Code review checks core module coverage
- Coverage report in PR description

---

## ADR-009: Monorepo Structure

**Status**: Accepted  
**Date**: 2025-10-17  
**Deciders**: Jose Flores Contreras

### Context

We need to organize code for data pipeline, models, training, evaluation, and deployment.

### Decision

Use a **monorepo structure** with all code in a single repository.

### Rationale

**Why Monorepo?**
1. **Simplicity**: One repo, one clone, one PR workflow
2. **Atomic Changes**: Can update data + models + training in one commit
3. **Easy Refactoring**: No cross-repo dependencies
4. **Single Source of Truth**: All code versioned together

**Why not Multi-Repo?**
- ❌ Complex: Need to coordinate changes across repos
- ❌ Versioning hell: Which version of data pipeline works with which model?
- ❌ Harder to refactor: Breaking changes require multi-repo PRs

### Alternatives Considered

1. **Multi-repo**: ❌ Too complex for small team
2. **Mono-package**: ❌ Doesn't scale (all code in one package)

### Consequences

**Positive**:
- Simple for small team
- Easy to refactor
- Single CI/CD pipeline

**Negative**:
- Can become large (but manageable with good structure)
- All code lives in one place (but that's also a benefit)

### Structure

```
atlasfx-mvp/
├── src/atlasfx/          # Main package
│   ├── data/             # Data pipeline
│   ├── models/           # VAE, TFT, SAC
│   ├── training/         # Training loops
│   ├── evaluation/       # Metrics, backtesting
│   ├── environments/     # Trading environment
│   ├── agents/           # RL agent wrappers
│   └── utils/            # Utilities
├── tests/                # Tests mirror src/ structure
├── docs/                 # Documentation
├── configs/              # Configuration files
├── scripts/              # CLI scripts
├── notebooks/            # Jupyter notebooks
└── experiments/          # Experiment results
```

---

## ADR-010: Pre-commit Hooks for Code Quality

**Status**: Accepted  
**Date**: 2025-10-21  
**Deciders**: Jose Flores Contreras

### Context

Code quality issues (formatting, linting, type errors) should be caught **before** commit, not in CI.

### Decision

Use **pre-commit hooks** to enforce:
- Code formatting (black, isort)
- Linting (ruff)
- Type checking (mypy)
- Security checks (bandit)
- Documentation (pydocstyle)

### Rationale

**Why Pre-commit?**
1. **Fast Feedback**: Catch errors immediately (not in CI)
2. **Consistent Style**: Enforced automatically
3. **Prevent Breakage**: Can't commit broken code
4. **Developer-Friendly**: Auto-fixes most issues

**Why not CI-only?**
- ❌ Slow feedback loop (minutes vs. seconds)
- ❌ Wastes CI resources
- ❌ Annoying for developers (commit → push → CI fails → fix → repeat)

### Alternatives Considered

1. **Manual checks**: ❌ Developers forget
2. **CI-only**: ❌ Slow feedback
3. **IDE integration**: ✅ Good, but not enforced

### Consequences

**Positive**:
- Fast feedback (seconds)
- Consistent style (auto-enforced)
- Fewer CI failures
- Better developer experience

**Negative**:
- Requires setup (one-time cost)
- Can be slow for large commits (rare)
- Can be bypassed with `--no-verify` (trust developers)

### Implementation

See `.pre-commit-config.yaml` for full configuration.

### Usage

```bash
# Install hooks (one-time)
pre-commit install

# Run manually
pre-commit run --all-files

# Bypass (emergency only)
git commit --no-verify
```

---

## Future ADRs

Topics for future decision records:
- ADR-011: Model checkpointing strategy
- ADR-012: Experiment tracking (MLflow vs. W&B)
- ADR-013: Backtesting framework design
- ADR-014: Data versioning with DVC
- ADR-015: Deployment strategy (Docker, Kubernetes)

---

**Maintained by**: Jose Flores Contreras  
**Last Updated**: 2025-10-21
