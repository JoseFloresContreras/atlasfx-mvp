# AtlasFX MVP

**Professional Algorithmic Trading System with Deep Learning**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-audit%20phase-yellow.svg)]()

---

## 🎯 Overview

AtlasFX is a professional-grade algorithmic trading system designed for short-term (1-10 minute) forex prediction on Level 1 tick data. The system integrates three state-of-the-art deep learning architectures:

1. **VAE (Variational Autoencoder)** - Learns compressed latent representations of market state
2. **TFT (Temporal Fusion Transformer)** - Provides multi-horizon probabilistic forecasts
3. **SAC (Soft Actor-Critic)** - Makes trading decisions with entropy-regularized RL

The project emphasizes **doctoral-level rigor**: reproducibility, modularity, comprehensive testing, type safety, and professional documentation.

---

## 🏗️ Architecture

```
Raw Tick Data (Dukascopy Level 1)
    ↓
Data Pipeline (Clean, Aggregate, Featurize)
    ↓
VAE (Latent State Representation)
    ↓
TFT (Multi-Horizon Forecasts + Uncertainty)
    ↓
SAC (Stochastic Trading Policy)
    ↓
Trading Environment (CFD Execution)
    ↓
Risk-Adjusted Returns
```

**Key Features:**
- ✅ Modular, type-safe Python codebase
- ✅ Comprehensive test suite (target: 80% coverage)
- ✅ Reproducible experiments with MLflow/W&B
- ✅ Data versioning with DVC
- ✅ CI/CD with GitHub Actions
- ✅ Professional documentation

---

## 📊 Current Status: Audit Phase

This repository is currently undergoing a comprehensive audit to determine what components are suitable for the MVP. See [AUDIT_REPORT.md](AUDIT_REPORT.md) for the full assessment.

**Summary of Findings:**
- ✅ **Data Pipeline:** Solid foundation, needs refactoring (add types, tests, validation)
- ⚠️ **Agent:** Uses TD3 instead of required SAC; needs reimplementation
- ❌ **Missing:** VAE and TFT components not yet implemented
- 🔧 **Next Steps:** Follow the [MVP_ACTION_PLAN.md](MVP_ACTION_PLAN.md) for implementation

---

## 📁 Repository Structure

```
atlasfx-mvp/
├── agent/                    # Current: TD3 implementation (to be replaced with SAC)
├── data-pipeline/            # Data processing (to be refactored)
├── AUDIT_REPORT.md          # Comprehensive repository assessment ✅
├── ARCHITECTURE.md          # System architecture documentation ✅
├── FEATURES.md              # Feature engineering documentation ✅
├── MVP_ACTION_PLAN.md       # 12-18 week implementation roadmap ✅
└── README.md                # This file
```

**Future Structure (after restructuring):**
```
atlasfx-mvp/
├── src/atlasfx/             # Main package
│   ├── data/                # Data pipeline (refactored)
│   ├── models/              # VAE, TFT, SAC implementations
│   ├── environments/        # Trading environment
│   ├── agents/              # RL agent wrapper
│   ├── utils/               # Shared utilities
│   └── config/              # Configuration schemas
├── tests/                   # Comprehensive test suite
├── experiments/             # MLflow/W&B logs
├── data/                    # Data (DVC tracked)
├── docs/                    # Sphinx documentation
├── scripts/                 # Training/evaluation scripts
├── pyproject.toml           # Dependencies and config
└── README.md
```

---

## 🚀 Quick Start (Post-Restructure)

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM
- Access to Dukascopy tick data

### Installation

```bash
# Clone the repository
git clone https://github.com/JoseFloresContreras/atlasfx-mvp.git
cd atlasfx-mvp

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Install pre-commit hooks
pre-commit install
```

### Running the Pipeline

```bash
# 1. Process raw tick data
python scripts/run_data_pipeline.py --config configs/data_pipeline.yaml

# 2. Train VAE
python scripts/train_vae.py --config configs/vae.yaml

# 3. Train TFT
python scripts/train_tft.py --config configs/tft.yaml

# 4. Train SAC agent
python scripts/train_sac.py --config configs/sac.yaml

# 5. Backtest on test set
python scripts/backtest.py --checkpoint models/sac_best.pth
```

### Running Tests

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run specific test module
pytest tests/unit/data/test_aggregators.py -v

# Run integration tests
pytest tests/integration/ -v
```

---

## 📚 Documentation

- **[AUDIT_REPORT.md](AUDIT_REPORT.md)** - Comprehensive analysis of current codebase
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed system architecture and design decisions
- **[FEATURES.md](FEATURES.md)** - Complete feature engineering documentation
- **[MVP_ACTION_PLAN.md](MVP_ACTION_PLAN.md)** - 12-18 week implementation roadmap

---

## 🔬 Research & References

### Papers

**VAE:**
- Kingma & Welling (2014) - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- Higgins et al. (2017) - [β-VAE: Learning Basic Visual Concepts](https://openreview.net/forum?id=Sy2fzU9gl)

**TFT:**
- Lim et al. (2021) - [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363)

**SAC:**
- Haarnoja et al. (2018) - [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
- Haarnoja et al. (2018) - [SAC Algorithms and Applications](https://arxiv.org/abs/1812.05905)

**Trading & Market Microstructure:**
- Cartea et al. (2015) - Algorithmic and High-Frequency Trading
- Lopez de Prado (2018) - Advances in Financial Machine Learning
- Easley et al. (2012) - Flow Toxicity and Liquidity in a High-Frequency World

---

## 🛠️ Development

### Code Quality Standards

- **Type Safety:** All code must have type hints; validated with `mypy`
- **Testing:** Minimum 80% test coverage; unit + integration tests
- **Formatting:** Black + isort + ruff for consistent style
- **Documentation:** Docstrings for all public APIs
- **Pre-commit:** All checks must pass before commit

### Workflow

1. Create feature branch from `main`
2. Implement changes with tests
3. Run linters and tests locally
4. Submit PR with detailed description
5. Pass CI/CD checks
6. Get code review approval
7. Merge to `main`

### Testing Strategy

```python
# Unit tests: Test individual functions in isolation
def test_vwap_calculation(sample_data):
    result = compute_vwap(sample_data)
    assert result.shape == expected_shape
    assert not result.isna().any()

# Integration tests: Test component interactions
def test_data_pipeline_end_to_end(raw_data):
    pipeline = DataPipeline(config)
    result = pipeline.run(raw_data)
    assert result.is_valid()
    
# Property-based tests: Test invariants
@given(st.lists(st.floats(min_value=0)))
def test_vwap_bounds(prices, volumes):
    vwap = compute_vwap(prices, volumes)
    assert min(prices) <= vwap <= max(prices)
```

---

## 📈 Performance Goals (MVP)

### Training Metrics
- VAE reconstruction error < 0.1
- TFT forecast RMSE < baseline (persistence model)
- SAC training stability (no divergence)

### Backtesting Metrics
- **Sharpe Ratio:** > 1.0 (risk-adjusted returns)
- **Maximum Drawdown:** < 20%
- **Win Rate:** > 50%
- **Profit Factor:** > 1.5

### Code Quality Metrics
- Test coverage ≥ 80%
- Zero mypy errors
- Zero critical security vulnerabilities
- Documentation completeness: 100%

---

## 🚧 Known Limitations

**Current (Audit Phase):**
- Missing VAE and TFT implementations
- TD3 agent needs replacement with SAC
- No comprehensive tests
- No type hints
- Potential lookahead bias in features (needs validation)

**Post-MVP:**
- No slippage modeling (will add in Phase 2)
- No market impact modeling
- Simplified transaction costs
- Single-pair focus (multi-asset in Phase 3)

---

## 🤝 Contributing

This is currently a private project in early development. Contributions are not yet accepted, but feedback on the architecture and approach is welcome.

**Contact:** [Jose Flores Contreras](https://github.com/JoseFloresContreras)

---

## 📄 License

[MIT License](LICENSE) - See LICENSE file for details

---

## 🎓 Academic Rigor

This project adheres to doctoral-level standards:

1. **Reproducibility:** All experiments tracked, seeds fixed, data versioned
2. **Validation:** Cross-validation, walk-forward analysis, out-of-sample testing
3. **Documentation:** Every design decision justified with references
4. **Testing:** Comprehensive test suite with property-based tests
5. **Code Quality:** Type-safe, linted, reviewed, CI/CD validated

**No shortcuts. No hacks. Production-grade from day one.**

---

## 🗓️ Roadmap

### Phase 1: Foundation (Weeks 1-2) - 🔄 In Progress
- [x] Repository audit complete
- [x] Architecture documentation
- [x] Action plan created
- [ ] Project restructuring
- [ ] CI/CD setup
- [ ] Testing infrastructure

### Phase 2: Data Pipeline (Weeks 3-4)
- [ ] Refactor with type hints
- [ ] Add comprehensive tests
- [ ] Validate features (no lookahead bias)
- [ ] Setup DVC for data versioning

### Phase 3: VAE (Weeks 5-6)
- [ ] Implement encoder/decoder
- [ ] Pre-training on features
- [ ] Latent space analysis

### Phase 4: TFT (Weeks 7-8)
- [ ] Implement TFT components
- [ ] Multi-horizon forecasting
- [ ] Uncertainty quantification

### Phase 5: SAC (Weeks 9-10)
- [ ] Implement actor-critic
- [ ] Trading environment refactor
- [ ] RL training loop

### Phase 6: Integration (Weeks 11-12)
- [ ] End-to-end pipeline
- [ ] Initial backtesting
- [ ] Performance analysis

### Phase 7: Optimization (Weeks 13-14)
- [ ] Hyperparameter tuning
- [ ] Architecture search
- [ ] Final model selection

### Phase 8: Validation (Weeks 15-18)
- [ ] Walk-forward validation
- [ ] Stress testing
- [ ] Final documentation
- [ ] MVP completion

---

## 📊 Project Stats

- **Total Lines of Code:** ~5,750 (pre-refactor)
- **Test Coverage:** 0% (to be added)
- **Documentation:** 50,000+ words (audit + architecture + features + plan)
- **Timeline:** 12-18 weeks to MVP
- **Status:** Audit Phase Complete ✅

---

**Last Updated:** October 17, 2025  
**Version:** 1.0 - Audit Phase  
**Next Milestone:** Project restructuring (Week 1)

---

*"The only way to write good software is to write good software."*  
— Discipline, rigor, and professionalism above all.
