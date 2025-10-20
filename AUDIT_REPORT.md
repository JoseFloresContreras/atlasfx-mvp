# AtlasFX MVP - Repository Audit Report
**Date:** October 17, 2025  
**Reviewer:** Staff Engineer  
**Status:** Early MVP Phase - Repository Assessment

---

## Executive Summary

This audit evaluates the current AtlasFX repository to determine what components are suitable for the MVP, what should be refactored, and what should be discarded. The project aims to build a professional-grade algorithmic trading system using **VAE + TFT + SAC** architecture for short-term (1-10 minute) forex predictions on Level 1 tick data from Dukascopy.

### Key Findings
- ✅ **Solid data pipeline foundation** with modular structure
- ⚠️ **Wrong RL algorithm** (TD3 instead of SAC as specified)
- ⚠️ **Missing core components** (VAE, TFT not implemented)
- ❌ **No tests, type hints, or data contracts**
- ❌ **No reproducibility setup** (seeds, versioning, experiment tracking)
- ⚠️ **Label leakage risks** in environment design

---

## Repository Structure Overview

```
atlasfx-mvp/
├── agent/
│   └── TD3/               # Twin Delayed DDPG implementation (~700 LOC)
│       ├── TD3.py         # Actor-Critic networks with BatchNorm/Dropout
│       ├── DDPG.py        # Alternative DDPG implementation
│       ├── OurDDPG.py     # Custom DDPG variant
│       ├── env.py         # Forex trading environment (~400 LOC)
│       ├── utils.py       # Replay buffer
│       └── main.py        # Training loop
├── data-pipeline/         # Data processing pipeline (~4,800 LOC)
│   ├── pipeline.py        # Orchestration logic
│   ├── pipeline.yaml      # Configuration
│   ├── merge.py           # Raw tick data merging
│   ├── clean.py           # Data cleaning with gap analysis
│   ├── aggregate.py       # Time-series aggregation (OHLC, VWAP, OFI, etc.)
│   ├── featurizers.py     # Feature engineering (~1,200 LOC)
│   ├── split.py           # Train/val/test split
│   ├── winsorize.py       # Outlier handling
│   ├── normalize.py       # Z-score normalization
│   ├── visualize.py       # Data visualization
│   └── logger.py          # Custom logging
└── test.ipynb             # Jupyter notebook with EDA (5.6 MB, many cells)
```

**Total Code:** ~5,750 lines of Python

---

## Detailed Component Analysis

### 1. Data Pipeline (data-pipeline/) ✅ **MOSTLY REUSABLE**

#### Strengths
1. **Modular Architecture**
   - Well-separated concerns (merge, clean, aggregate, featurize, normalize)
   - YAML-based configuration for reproducibility
   - Pipeline orchestration with dependency management
   - Comprehensive logging

2. **Quality Features**
   - Tick count, OHLC, volume, VWAP, OFI (Order Flow Imbalance), micro price
   - Technical indicators: RSI, Bollinger Bands, ATR, volatility ratio
   - Market microstructure: bipower variation, CSI (Cross-Sectional Imbalance)
   - Session indicators (Sydney, Tokyo, London, NY)
   - Correlation features between pairs
   - Temporal features (minute of day, day of week)

3. **Data Validation**
   - Gap detection and analysis
   - Outlier handling (winsorization)
   - Normalization with clipping

#### Issues to Fix

1. **Missing Type Hints** ⚠️
   - No type annotations (violates doctoral standards)
   - No Pydantic/dataclass contracts for data schemas

2. **No Tests** ❌
   - Zero unit tests
   - No integration tests
   - No data validation tests

3. **Reproducibility Gaps** ⚠️
   - No data versioning (DVC)
   - No feature store concept
   - No experiment tracking (MLflow, Weights & Biases)
   - Random seed management unclear

4. **Label Leakage Risk** 🚨
   - Feature calculations may look ahead
   - Need strict temporal validation
   - Train/val/test split by time, but cross-contamination risk

5. **Dependency on pandas_ta** ⚠️
   - Uses external library (line 16 in featurizers.py)
   - Not in requirements.txt (inconsistency)

#### Verdict: **REFACTOR & REUSE**
- Keep modular structure
- Add type hints and data contracts
- Add comprehensive tests
- Review all features for lookahead bias
- Add experiment tracking
- Document each feature's purpose and validation

---

### 2. Agent (agent/TD3/) ⚠️ **WRONG ALGORITHM**

#### What's Implemented
- **TD3 (Twin Delayed DDPG)** - Off-policy actor-critic method
- Actor network: 2 hidden layers (256 units), BatchNorm, Dropout (0.1)
- Critic network: Twin Q-networks with same architecture
- Replay buffer with standard experience replay
- Trading environment with CFD positions

#### Critical Issues

1. **Wrong Algorithm** 🚨
   - Project specifies **SAC (Soft Actor-Critic)**, not TD3
   - TD3 is deterministic, SAC is stochastic with entropy regularization
   - SAC is more suitable for exploration in trading

2. **No VAE or TFT** ❌
   - Missing VAE for state representation learning
   - Missing TFT (Temporal Fusion Transformer) for forecasting
   - Architecture should be: VAE → TFT → SAC
   - Current: Raw features → TD3

3. **Environment Design Flaws** ⚠️
   - CFD position tracking seems ad-hoc
   - Reward design not documented or justified
   - Transaction costs model unclear
   - Slippage not modeled
   - Market impact not considered

4. **No Tests** ❌
   - No unit tests for networks
   - No environment validation tests
   - No reward function tests

5. **Code Quality** ⚠️
   - No type hints
   - Minimal documentation
   - Hard-coded hyperparameters in main.py
   - No configuration management

6. **Reproducibility** ❌
   - Seeds set but not validated
   - No deterministic mode for PyTorch
   - No version pinning (PyTorch, Gym, etc.)

#### Verdict: **DISCARD & REIMPLEMENT**
- TD3 implementation is clean but wrong algorithm
- Trading environment has potential but needs redesign
- Start fresh with SAC + proper architecture
- Reuse replay buffer utility (utils.py)

---

### 3. Jupyter Notebook (test.ipynb) ℹ️ **EXPLORATORY WORK**

#### Contents
- Hour-of-day analysis for tick counts and volume
- CSI (Cross-Sectional Imbalance) distributions
- Lagged correlation analysis
- Learning curve visualization

#### Verdict: **EXTRACT & DOCUMENT**
- Extract insights into documentation
- Convert useful analyses to scripts
- Delete notebook from main repo (move to experiments/)
- Use proper experiment tracking instead

---

## Architecture Gaps: VAE + TFT + SAC

The repository is **missing the core architecture** specified in the requirements:

### Required Components (Not Implemented)

1. **VAE (Variational Autoencoder)** ❌
   - **Purpose:** Learn compressed latent representations of market state
   - **Input:** Raw features from data pipeline
   - **Output:** Latent state vector (e.g., 32-128 dimensions)
   - **Missing:** Entire component

2. **TFT (Temporal Fusion Transformer)** ❌
   - **Purpose:** Multi-horizon forecasting of price movements (1-10 min)
   - **Input:** VAE latent states + temporal covariates
   - **Output:** Probabilistic forecasts (quantiles, point estimates)
   - **Missing:** Entire component

3. **SAC (Soft Actor-Critic)** ❌
   - **Purpose:** Stochastic policy learning with entropy regularization
   - **Input:** VAE states + TFT forecasts
   - **Output:** Trading actions (position sizing, direction)
   - **Current:** TD3 instead (deterministic, no entropy)

### Correct Data Flow

```
Raw Tick Data (Level 1)
    ↓
Data Pipeline (merge, clean, aggregate, featurize)
    ↓
VAE Encoder
    ↓
Latent State Representation (z ~ N(μ, σ²))
    ↓
TFT Transformer
    ↓
Multi-Step Forecasts (1-10 min) + Uncertainty
    ↓
SAC Agent (Actor-Critic + Entropy)
    ↓
Trading Actions (position, direction, size)
    ↓
Environment Execution
    ↓
Reward (PnL, Sharpe, drawdown penalties)
    ↓
Replay Buffer → Update VAE, TFT, SAC
```

---

## Risk Assessment

### High Priority Risks 🚨

1. **Label Leakage**
   - Features may use future information
   - Train/val/test split needs temporal validation
   - Rolling window features need careful bounds checking

2. **Reproducibility Crisis**
   - No experiment tracking
   - No data versioning
   - No model registry
   - Hard to reproduce results

3. **Wrong Foundation**
   - Building on TD3 when SAC is required
   - Missing VAE and TFT means rebuilding from scratch

4. **No Testing Culture**
   - Zero tests increases bug risk
   - Hard to refactor safely
   - No validation of financial calculations

### Medium Priority Risks ⚠️

5. **Type Safety**
   - No type hints makes refactoring risky
   - No data contracts (Pydantic, dataclasses)

6. **Environment Realism**
   - CFD modeling seems simplistic
   - No slippage, market impact, or realistic execution
   - Reward function not validated

7. **Scalability**
   - All data loaded in memory
   - No streaming/batching for large datasets

---

## Recommendations for MVP

### Phase 1: Foundation (Weeks 1-2)

1. **Setup Infrastructure**
   - [ ] Add `pyproject.toml` with Poetry/PDM for dependency management
   - [ ] Add `pytest` with coverage requirements (≥80%)
   - [ ] Add `mypy` for type checking
   - [ ] Add `ruff` or `black` + `isort` for formatting
   - [ ] Setup pre-commit hooks
   - [ ] Add GitHub Actions CI/CD

2. **Data Pipeline Refactor**
   - [ ] Add type hints to all functions
   - [ ] Create Pydantic schemas for data contracts
   - [ ] Add unit tests for each module
   - [ ] Audit features for lookahead bias
   - [ ] Document each feature's purpose and validation
   - [ ] Add DVC for data versioning

3. **Project Structure**
   ```
   atlasfx-mvp/
   ├── src/
   │   ├── data/          # Data pipeline (refactored)
   │   ├── models/        # VAE, TFT, SAC implementations
   │   ├── environments/  # Trading environment
   │   ├── utils/         # Shared utilities
   │   └── config/        # Configuration schemas
   ├── tests/             # Comprehensive test suite
   ├── experiments/       # Experiment tracking (MLflow)
   ├── data/              # Data (DVC tracked)
   ├── docs/              # Documentation
   ├── notebooks/         # Exploratory work (not for production)
   ├── pyproject.toml     # Dependencies and config
   ├── README.md          # Professional README
   └── .github/
       └── workflows/     # CI/CD pipelines
   ```

### Phase 2: Core Architecture (Weeks 3-4)

4. **Implement VAE**
   - [ ] Design encoder/decoder architecture
   - [ ] Add KL divergence regularization
   - [ ] Test on real data
   - [ ] Validate latent space quality
   - [ ] Add reconstruction metrics

5. **Implement TFT**
   - [ ] Use PyTorch Forecasting library or custom implementation
   - [ ] Multi-horizon forecasting (1, 5, 10 min)
   - [ ] Quantile regression for uncertainty
   - [ ] Attention visualization
   - [ ] Backtest forecast accuracy

6. **Implement SAC**
   - [ ] Stochastic policy network
   - [ ] Twin Q-critics
   - [ ] Entropy regularization
   - [ ] Target networks
   - [ ] Replay buffer integration

### Phase 3: Integration & Validation (Weeks 5-6)

7. **End-to-End Pipeline**
   - [ ] Connect VAE → TFT → SAC
   - [ ] Training loop with checkpointing
   - [ ] Hyperparameter tuning (Optuna)
   - [ ] Experiment tracking (MLflow/W&B)

8. **Backtesting & Validation**
   - [ ] Walk-forward validation
   - [ ] Out-of-sample testing
   - [ ] Performance metrics (Sharpe, Sortino, max drawdown)
   - [ ] Transaction cost analysis
   - [ ] Risk-adjusted returns

9. **Documentation**
   - [ ] Architecture decision records (ADRs)
   - [ ] API documentation (Sphinx)
   - [ ] User guide
   - [ ] Research notes and references

---

## What to Keep vs. Discard

### ✅ Keep & Refactor

1. **Data Pipeline Structure**
   - Modular design is excellent
   - YAML configuration approach
   - Logging infrastructure
   - Feature engineering ideas (need validation)

2. **Trading Environment Concept**
   - CFD position tracking (refine)
   - Episode sampling approach
   - Observation space design (augment with VAE/TFT)

3. **Replay Buffer (utils.py)**
   - Standard implementation, reusable

### ⚠️ Refactor Heavily

4. **Pipeline Modules**
   - Add types, tests, documentation
   - Validate for lookahead bias
   - Add data contracts

5. **Environment Rewards**
   - Document current design
   - Research best practices
   - Add multiple reward formulations

### ❌ Discard

6. **TD3 Implementation**
   - Wrong algorithm (need SAC)
   - Start fresh with proper architecture

7. **test.ipynb**
   - Move insights to documentation
   - Use proper experiment tracking

8. **Actor/Critic from TD3**
   - Architecture decisions (BatchNorm, Dropout) need justification
   - SAC architecture will differ

---

## Technical Debt Summary

| Category | Severity | Items | Effort |
|----------|----------|-------|--------|
| Architecture | 🚨 Critical | Missing VAE, TFT, wrong RL algorithm | 4-6 weeks |
| Testing | 🚨 Critical | Zero tests, no CI/CD | 2-3 weeks |
| Type Safety | ⚠️ High | No type hints, no contracts | 1-2 weeks |
| Reproducibility | ⚠️ High | No versioning, tracking, seeds | 1-2 weeks |
| Label Leakage | 🚨 Critical | Features need audit | 1-2 weeks |
| Documentation | ⚠️ High | Minimal docs, no ADRs | 1-2 weeks |
| Code Quality | ⚠️ Medium | Inconsistent style, deps | 1 week |

**Total Estimated Effort:** 12-18 weeks for production-grade MVP

---

## Immediate Next Steps

### Week 1 Tasks

1. **Setup Modern Python Project**
   ```bash
   # Add pyproject.toml with Poetry
   poetry init
   poetry add pandas numpy torch gymnasium pytest mypy ruff
   poetry add --group dev black isort pytest-cov
   ```

2. **Create Test Infrastructure**
   ```bash
   mkdir -p tests/{unit,integration}
   touch tests/conftest.py
   # Add pytest configuration
   ```

3. **Add Type Hints to Data Pipeline**
   - Start with merge.py, clean.py
   - Add return types and parameter types
   - Run mypy to validate

4. **Document Current State**
   - Create ARCHITECTURE.md
   - Create FEATURES.md (list all features with formulas)
   - Create EXPERIMENTS.md (log book template)

5. **Audit Features for Lookahead Bias**
   - Review each featurizer
   - Check rolling window boundaries
   - Validate temporal ordering

### Week 2 Tasks

6. **Add Unit Tests**
   - Test data loaders
   - Test feature calculators
   - Test normalization
   - Target: 80% coverage

7. **Setup Experiment Tracking**
   - Add MLflow or Weights & Biases
   - Create experiment templates
   - Log hyperparameters and metrics

8. **Research & Design**
   - VAE architecture for forex data
   - TFT implementation options
   - SAC hyperparameters for trading

---

## Questions for User

1. **Data Availability:** Do you have access to Dukascopy Level 1 tick data? What time range?

2. **Computational Resources:** What GPU resources are available for training?

3. **Timeline:** Is the 12-18 week estimate acceptable for a production-grade MVP?

4. **Priorities:** Should we focus on MVP speed or correctness? (I recommend correctness)

5. **Team:** Will you be working solo or with other developers?

6. **Deployment:** Is this for personal trading or institutional use? (affects compliance)

---

## Conclusion

The current repository has a **solid data pipeline foundation** but is **missing the core ML architecture** (VAE + TFT + SAC). The TD3 agent must be discarded and rebuilt from scratch with the correct algorithms.

**Recommendation:** Start fresh with a properly structured project, reuse and refactor the data pipeline, and implement the full architecture with tests, types, and documentation from day one.

**Risk Level:** 🟡 Medium-High (manageable with proper planning)

**MVP Viability:** ✅ Achievable in 12-18 weeks with disciplined execution

---

**Next Action:** Review this audit with the user and get alignment on priorities and timeline before proceeding with implementation.
