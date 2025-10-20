# AtlasFX MVP - Action Plan

**Version:** 1.0  
**Created:** October 17, 2025  
**Timeline:** 12-18 weeks  
**Status:** Planning Phase

---

## Executive Summary

This action plan outlines the step-by-step roadmap to build a production-grade MVP of AtlasFX, a professional algorithmic trading system using VAE + TFT + SAC architecture. The plan prioritizes correctness, reproducibility, and professional standards over speed.

---

## Timeline Overview

```
Weeks 1-2:  Foundation & Infrastructure Setup
Weeks 3-4:  Data Pipeline Refactor & Validation
Weeks 5-6:  VAE Implementation & Pre-training
Weeks 7-8:  TFT Implementation & Pre-training
Weeks 9-10: SAC Implementation & RL Training
Weeks 11-12: Integration & Initial Backtesting
Weeks 13-14: Hyperparameter Tuning & Optimization
Weeks 15-16: Validation & Documentation
Weeks 17-18: Buffer for Issues & Final Testing
```

---

## Phase 1: Foundation (Weeks 1-2)

### Week 1: Project Setup & Infrastructure

#### Day 1-2: Modern Python Project Structure

**Tasks:**
1. Create new project structure
   ```
   atlasfx-mvp/
   ├── src/
   │   ├── atlasfx/
   │   │   ├── __init__.py
   │   │   ├── data/          # Data pipeline
   │   │   ├── models/        # VAE, TFT, SAC
   │   │   ├── environments/  # Trading env
   │   │   ├── agents/        # RL agent wrapper
   │   │   ├── utils/         # Shared utilities
   │   │   └── config/        # Configuration
   ├── tests/
   │   ├── unit/
   │   ├── integration/
   │   └── conftest.py
   ├── experiments/           # MLflow/W&B logs
   ├── data/                  # DVC tracked
   ├── notebooks/             # Exploratory only
   ├── docs/                  # Sphinx documentation
   ├── scripts/               # Training/evaluation scripts
   ├── pyproject.toml
   ├── README.md
   ├── .pre-commit-config.yaml
   └── .github/workflows/
   ```

2. Setup dependency management
   ```bash
   # Initialize Poetry project
   poetry init
   
   # Core dependencies
   poetry add python="^3.10"
   poetry add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   poetry add pandas numpy pyarrow
   poetry add gymnasium stable-baselines3
   poetry add pydantic pyyaml
   poetry add mlflow wandb
   poetry add pytorch-forecasting  # For TFT
   
   # Development dependencies
   poetry add --group dev pytest pytest-cov pytest-mock
   poetry add --group dev mypy types-pyyaml types-requests
   poetry add --group dev ruff black isort
   poetry add --group dev pre-commit
   poetry add --group dev sphinx sphinx-rtd-theme
   ```

3. Configure code quality tools
   ```toml
   # pyproject.toml
   [tool.pytest.ini_options]
   minversion = "7.0"
   addopts = "-ra -q --cov=src --cov-report=html --cov-report=term"
   testpaths = ["tests"]
   
   [tool.mypy]
   python_version = "3.10"
   strict = true
   warn_return_any = true
   warn_unused_configs = true
   
   [tool.ruff]
   line-length = 100
   target-version = "py310"
   
   [tool.black]
   line-length = 100
   target-version = ['py310']
   ```

**Deliverables:**
- [ ] Project structure created
- [ ] Dependencies installed and locked
- [ ] Code quality tools configured

**Success Criteria:**
- `poetry install` runs without errors
- `pytest` runs (0 tests, but no errors)
- `mypy src` runs (no files yet, but no errors)

---

#### Day 3-4: CI/CD & Git Workflow

**Tasks:**
1. Setup pre-commit hooks
   ```yaml
   # .pre-commit-config.yaml
   repos:
     - repo: https://github.com/pre-commit/pre-commit-hooks
       hooks:
         - id: trailing-whitespace
         - id: end-of-file-fixer
         - id: check-yaml
         - id: check-added-large-files
     - repo: https://github.com/psf/black
       hooks:
         - id: black
     - repo: https://github.com/pycqa/isort
       hooks:
         - id: isort
     - repo: https://github.com/charliermarsh/ruff-pre-commit
       hooks:
         - id: ruff
     - repo: https://github.com/pre-commit/mirrors-mypy
       hooks:
         - id: mypy
   ```

2. Setup GitHub Actions CI
   ```yaml
   # .github/workflows/ci.yml
   name: CI
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - uses: actions/setup-python@v4
           with:
             python-version: '3.10'
         - name: Install dependencies
           run: |
             pip install poetry
             poetry install
         - name: Run linters
           run: |
             poetry run ruff check src tests
             poetry run black --check src tests
             poetry run mypy src
         - name: Run tests
           run: poetry run pytest
   ```

3. Setup branch protection rules
   - Require PR reviews
   - Require CI to pass
   - No force push to main

**Deliverables:**
- [ ] Pre-commit hooks installed
- [ ] GitHub Actions CI configured
- [ ] Branch protection rules set

**Success Criteria:**
- Pre-commit hooks run on commit
- CI pipeline runs on PR
- Cannot merge without passing tests

---

#### Day 5-7: Configuration System & Logging

**Tasks:**
1. Create Pydantic configuration schemas
   ```python
   # src/atlasfx/config/schemas.py
   from pydantic import BaseModel, Field
   from typing import List, Dict, Optional
   
   class DataPipelineConfig(BaseModel):
       """Configuration for data pipeline."""
       time_window: str = Field(..., description="Aggregation window (e.g., 5min)")
       symbols: List[str] = Field(..., description="List of trading pairs")
       train_split: float = Field(0.7, ge=0, le=1)
       val_split: float = Field(0.15, ge=0, le=1)
       test_split: float = Field(0.15, ge=0, le=1)
       
   class VAEConfig(BaseModel):
       """Configuration for VAE model."""
       input_dim: int = Field(..., description="Feature dimension")
       latent_dim: int = Field(64, description="Latent space dimension")
       hidden_dims: List[int] = Field([128, 64], description="Hidden layer dimensions")
       beta: float = Field(1.0, ge=0, description="KL divergence weight")
       learning_rate: float = Field(1e-3, gt=0)
       batch_size: int = Field(256, gt=0)
       
   class TFTConfig(BaseModel):
       """Configuration for TFT model."""
       hidden_dim: int = Field(128, description="Hidden state dimension")
       num_heads: int = Field(4, gt=0, description="Attention heads")
       num_layers: int = Field(2, gt=0, description="Encoder layers")
       forecast_horizons: List[int] = Field([1, 5, 10], description="Forecast steps")
       quantiles: List[float] = Field([0.1, 0.5, 0.9], description="Quantiles")
       
   class SACConfig(BaseModel):
       """Configuration for SAC agent."""
       state_dim: int = Field(..., description="State space dimension")
       action_dim: int = Field(..., description="Action space dimension")
       hidden_dim: int = Field(256, description="Hidden layer dimension")
       alpha: float = Field(0.2, gt=0, description="Entropy coefficient")
       gamma: float = Field(0.99, ge=0, le=1, description="Discount factor")
       tau: float = Field(0.005, gt=0, description="Soft update rate")
       learning_rate: float = Field(3e-4, gt=0)
       batch_size: int = Field(256, gt=0)
       replay_buffer_size: int = Field(1_000_000, gt=0)
   ```

2. Create experiment configuration
   ```python
   # src/atlasfx/config/experiment.py
   class ExperimentConfig(BaseModel):
       """Configuration for full experiment."""
       name: str = Field(..., description="Experiment name")
       seed: int = Field(42, description="Random seed")
       device: str = Field("cuda", description="Device (cuda/cpu)")
       
       data: DataPipelineConfig
       vae: VAEConfig
       tft: TFTConfig
       sac: SACConfig
       
       @classmethod
       def from_yaml(cls, path: str) -> "ExperimentConfig":
           """Load config from YAML file."""
           with open(path) as f:
               config_dict = yaml.safe_load(f)
           return cls(**config_dict)
   ```

3. Setup structured logging
   ```python
   # src/atlasfx/utils/logging.py
   import logging
   import sys
   from pathlib import Path
   
   def setup_logger(name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
       """Setup structured logger with file and console handlers."""
       logger = logging.getLogger(name)
       logger.setLevel(level)
       
       # Console handler
       console_handler = logging.StreamHandler(sys.stdout)
       console_handler.setLevel(level)
       console_formatter = logging.Formatter(
           '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
       )
       console_handler.setFormatter(console_formatter)
       
       # File handler
       log_dir.mkdir(parents=True, exist_ok=True)
       file_handler = logging.FileHandler(log_dir / f"{name}.log")
       file_handler.setLevel(level)
       file_handler.setFormatter(console_formatter)
       
       logger.addHandler(console_handler)
       logger.addHandler(file_handler)
       
       return logger
   ```

**Deliverables:**
- [ ] Pydantic configuration schemas
- [ ] YAML config loader
- [ ] Structured logging system

**Success Criteria:**
- Can load config from YAML without errors
- Config validation catches invalid values
- Logs written to both console and file

---

### Week 2: Testing Infrastructure & Documentation

#### Day 8-10: Testing Framework

**Tasks:**
1. Create test fixtures
   ```python
   # tests/conftest.py
   import pytest
   import pandas as pd
   import numpy as np
   from pathlib import Path
   
   @pytest.fixture
   def sample_tick_data():
       """Generate sample tick data for testing."""
       n = 1000
       timestamps = pd.date_range('2024-01-01', periods=n, freq='1s')
       data = pd.DataFrame({
           'timestamp': timestamps.astype(int) // 10**6,  # ms
           'bid': np.random.randn(n).cumsum() + 1.1000,
           'ask': np.random.randn(n).cumsum() + 1.1010,
           'volume': np.random.randint(1, 100, n)
       })
       return data
   
   @pytest.fixture
   def sample_aggregated_data():
       """Generate sample aggregated data."""
       n = 100
       timestamps = pd.date_range('2024-01-01', periods=n, freq='5min')
       data = pd.DataFrame({
           'start_time': timestamps.astype(int) // 10**6,
           'open': np.random.randn(n).cumsum() + 1.1000,
           'high': np.random.randn(n).cumsum() + 1.1050,
           'low': np.random.randn(n).cumsum() + 1.0950,
           'close': np.random.randn(n).cumsum() + 1.1020,
           'volume': np.random.randint(100, 1000, n),
       })
       return data
   
   @pytest.fixture
   def temp_data_dir(tmp_path):
       """Create temporary data directory."""
       data_dir = tmp_path / "data"
       data_dir.mkdir()
       return data_dir
   ```

2. Write example tests
   ```python
   # tests/unit/data/test_aggregators.py
   import pytest
   import pandas as pd
   from atlasfx.data.aggregators import compute_vwap
   
   def test_vwap_calculation(sample_tick_data):
       """Test VWAP calculation."""
       result = compute_vwap(sample_tick_data, window='5min')
       
       # Check output shape
       assert len(result) > 0
       
       # Check VWAP is between min and max price
       assert (result['vwap'] >= result['low']).all()
       assert (result['vwap'] <= result['high']).all()
       
       # Check no NaN values
       assert not result['vwap'].isna().any()
   
   def test_vwap_with_zero_volume(sample_tick_data):
       """Test VWAP handles zero volume."""
       data = sample_tick_data.copy()
       data.loc[0:10, 'volume'] = 0
       
       result = compute_vwap(data, window='5min')
       
       # Should handle gracefully (no errors)
       assert len(result) > 0
   ```

3. Setup coverage reporting
   ```bash
   # Run tests with coverage
   poetry run pytest --cov=src --cov-report=html --cov-report=term
   
   # View coverage report
   open htmlcov/index.html
   ```

**Deliverables:**
- [ ] Test fixtures for common data structures
- [ ] Example unit tests
- [ ] Coverage reporting configured

**Success Criteria:**
- Tests run without errors
- Coverage report generated
- Target: 80% coverage (will grow as code is added)

---

#### Day 11-14: Documentation & Initial Migration

**Tasks:**
1. Write comprehensive README
   - Project overview
   - Installation instructions
   - Quickstart guide
   - Development setup

2. Setup Sphinx documentation
   ```bash
   cd docs
   sphinx-quickstart
   # Configure autodoc, napoleon, rtd theme
   ```

3. Document architecture decisions
   - Why VAE over other autoencoders?
   - Why TFT over LSTM/Transformer?
   - Why SAC over TD3/PPO?

4. Begin migrating data pipeline
   - Copy existing modules to new structure
   - Add type hints
   - Add docstrings
   - Write tests

**Deliverables:**
- [ ] Professional README.md
- [ ] Sphinx docs initialized
- [ ] Architecture Decision Records (ADRs)
- [ ] Data pipeline modules migrated (partial)

**Success Criteria:**
- README has all essential information
- Docs build without errors (`make html`)
- At least 1 module fully migrated with tests

---

## Phase 2: Data Pipeline (Weeks 3-4)

### Week 3: Data Pipeline Refactor

#### Day 15-17: Merge, Clean, Aggregate

**Tasks:**
1. Refactor merge.py with types and tests
2. Refactor clean.py with gap detection
3. Refactor aggregate.py with all aggregators
4. Add Pydantic schemas for data validation

**Deliverables:**
- [ ] Merge module with 100% type coverage
- [ ] Clean module with gap analysis tests
- [ ] Aggregate module with unit tests for each aggregator
- [ ] Data schemas validated with Pydantic

---

#### Day 18-21: Featurizers & Validation

**Tasks:**
1. Refactor featurizers.py
2. Implement lookahead bias tests
3. Add feature validation pipeline
4. Document each feature with formula

**Deliverables:**
- [ ] All featurizers type-hinted and tested
- [ ] Automated lookahead bias detection
- [ ] Feature validation report

---

### Week 4: Data Validation & Versioning

#### Day 22-24: Feature Validation

**Tasks:**
1. Run full pipeline on sample data
2. Generate feature correlation matrix
3. Check for NaN/inf values
4. Validate temporal consistency

**Deliverables:**
- [ ] Feature validation report
- [ ] Correlation heatmap
- [ ] NaN/inf summary

---

#### Day 25-28: Data Versioning & Experiment Tracking

**Tasks:**
1. Setup DVC for data versioning
2. Setup MLflow for experiment tracking
3. Create experiment templates
4. Document data lineage

**Deliverables:**
- [ ] DVC initialized with remote storage
- [ ] MLflow tracking server running
- [ ] First experiment logged

---

## Phase 3: VAE Implementation (Weeks 5-6)

### Week 5: VAE Architecture

#### Day 29-31: Encoder & Decoder

**Tasks:**
1. Implement VAE encoder
2. Implement VAE decoder
3. Add reparameterization trick
4. Add KL divergence loss

**Code Structure:**
```python
# src/atlasfx/models/vae.py
class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        ...
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns mean and log_var."""
        ...
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        ...
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        ...
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns reconstruction, mu, log_var."""
        ...
    
    def loss_function(self, x: torch.Tensor, recon_x: torch.Tensor, 
                      mu: torch.Tensor, log_var: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute VAE loss."""
        ...
```

**Deliverables:**
- [ ] VAE implementation
- [ ] Unit tests for each method
- [ ] Loss function validated

---

#### Day 32-35: VAE Training

**Tasks:**
1. Create VAE training script
2. Add checkpointing
3. Add early stopping
4. Log metrics to MLflow

**Deliverables:**
- [ ] Training script
- [ ] Trained VAE on sample data
- [ ] Validation metrics logged

---

### Week 6: VAE Evaluation

#### Day 36-38: Latent Space Analysis

**Tasks:**
1. Visualize latent space (PCA, t-SNE)
2. Measure reconstruction quality
3. Evaluate disentanglement
4. Test on validation set

**Deliverables:**
- [ ] Latent space visualizations
- [ ] Reconstruction error < 0.1
- [ ] Validation report

---

#### Day 39-42: VAE Fine-tuning

**Tasks:**
1. Hyperparameter tuning (beta, hidden_dims)
2. Try different architectures
3. Select best model
4. Save final checkpoint

**Deliverables:**
- [ ] Tuning results logged
- [ ] Best model selected
- [ ] Model checkpoint saved

---

## Phase 4: TFT Implementation (Weeks 7-8)

### Week 7: TFT Architecture

#### Day 43-45: TFT Components

**Tasks:**
1. Implement Variable Selection Networks
2. Implement LSTM Encoder
3. Implement Multi-Head Attention
4. Implement Quantile Regression Heads

**Deliverables:**
- [ ] TFT components implemented
- [ ] Unit tests for each component
- [ ] Forward pass validated

---

#### Day 46-49: TFT Training

**Tasks:**
1. Create TFT training script
2. Train on VAE latent states
3. Validate forecast accuracy
4. Log metrics to MLflow

**Deliverables:**
- [ ] Training script
- [ ] Trained TFT on sample data
- [ ] Forecast RMSE reported

---

### Week 8: TFT Evaluation

#### Day 50-52: Forecast Validation

**Tasks:**
1. Multi-horizon forecast evaluation
2. Quantile coverage analysis
3. Attention weight visualization
4. Walk-forward validation

**Deliverables:**
- [ ] Forecast accuracy report
- [ ] Quantile coverage plots
- [ ] Attention visualizations

---

#### Day 53-56: TFT Fine-tuning

**Tasks:**
1. Hyperparameter tuning
2. Try different architectures
3. Select best model
4. Save final checkpoint

**Deliverables:**
- [ ] Tuning results logged
- [ ] Best model selected
- [ ] Model checkpoint saved

---

## Phase 5: SAC Implementation (Weeks 9-10)

### Week 9: SAC Architecture

#### Day 57-59: Actor & Critic Networks

**Tasks:**
1. Implement Gaussian policy (actor)
2. Implement twin Q-critics
3. Add target networks
4. Add entropy tuning

**Deliverables:**
- [ ] SAC components implemented
- [ ] Unit tests for networks
- [ ] Forward/backward pass validated

---

#### Day 60-63: Trading Environment

**Tasks:**
1. Refactor trading environment
2. Add realistic transaction costs
3. Implement reward function
4. Add environment tests

**Deliverables:**
- [ ] Environment implemented
- [ ] Reward function documented
- [ ] Environment tests pass

---

### Week 10: SAC Training

#### Day 64-66: RL Training Loop

**Tasks:**
1. Implement SAC update rule
2. Add replay buffer
3. Create training script
4. Train on sample episodes

**Deliverables:**
- [ ] Training script
- [ ] SAC training on sample data
- [ ] Rewards plotted

---

#### Day 67-70: Initial Backtesting

**Tasks:**
1. Run backtest on validation set
2. Compute performance metrics
3. Analyze trade statistics
4. Log results to MLflow

**Deliverables:**
- [ ] Backtest results
- [ ] Sharpe ratio computed
- [ ] Trade analysis report

---

## Phase 6: Integration & Validation (Weeks 11-18)

### Weeks 11-12: End-to-End Integration
- [ ] Connect VAE → TFT → SAC pipeline
- [ ] Run full training loop
- [ ] Validate on test set
- [ ] Generate performance report

### Weeks 13-14: Hyperparameter Optimization
- [ ] Setup Optuna for HPO
- [ ] Define search space
- [ ] Run optimization trials
- [ ] Select best hyperparameters

### Weeks 15-16: Final Validation
- [ ] Walk-forward validation
- [ ] Stress testing
- [ ] Robustness checks
- [ ] Final documentation

### Weeks 17-18: Buffer & Polish
- [ ] Address any issues found
- [ ] Final code review
- [ ] Documentation completeness check
- [ ] Prepare final presentation

---

## Success Metrics

### Technical Metrics
- [ ] Test coverage ≥ 80%
- [ ] No mypy errors
- [ ] CI/CD passing
- [ ] All features validated (no lookahead bias)

### Model Metrics
- [ ] VAE reconstruction error < 0.1
- [ ] TFT forecast RMSE < baseline
- [ ] SAC Sharpe ratio > 1.0 on validation
- [ ] Maximum drawdown < 20%

### Process Metrics
- [ ] All experiments logged in MLflow
- [ ] Data versioned with DVC
- [ ] Code reviewed before merge
- [ ] Documentation complete

---

## Risk Mitigation

### Technical Risks
1. **Risk:** Feature lookahead bias  
   **Mitigation:** Automated tests, manual audits

2. **Risk:** Overfitting to training data  
   **Mitigation:** Walk-forward validation, regularization

3. **Risk:** Numerical instability in training  
   **Mitigation:** Gradient clipping, batch normalization

### Project Risks
1. **Risk:** Timeline slippage  
   **Mitigation:** 2-week buffer, weekly checkpoints

2. **Risk:** Scope creep  
   **Mitigation:** Strict MVP definition, defer enhancements

3. **Risk:** Resource constraints  
   **Mitigation:** Cloud GPU (Colab, Lambda Labs)

---

## Next Steps

1. **Review & Approve:** Get stakeholder sign-off on this plan
2. **Setup Repository:** Create new repo structure (Week 1, Day 1)
3. **Begin Implementation:** Follow daily tasks sequentially
4. **Weekly Check-ins:** Review progress every Friday
5. **Adapt as Needed:** Adjust plan based on learnings

---

**Document Status:** Planning Phase  
**Next Review:** After Week 2 completion  
**Owner:** Staff Engineer
