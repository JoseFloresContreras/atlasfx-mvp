# Contributing to AtlasFX

Thank you for your interest in contributing to AtlasFX! This document provides guidelines and workflows for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Git Workflow](#git-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Commit Message Convention](#commit-message-convention)

## Code of Conduct

This project adheres to doctoral-level standards of professionalism, rigor, and respect. All contributors are expected to:

- Write production-grade code with comprehensive tests
- Document all design decisions and architectural choices
- Provide constructive feedback in code reviews
- Maintain reproducibility and scientific rigor
- Follow established conventions and standards

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) CUDA 11.8+ for GPU training

### Initial Setup

```bash
# 1. Clone the repository
git clone https://github.com/JoseFloresContreras/atlasfx-mvp.git
cd atlasfx-mvp

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies (development mode)
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Verify installation
pytest tests/ -v
black --check src/
ruff check src/
mypy src/atlasfx
```

### Environment Configuration

For reproducibility, always set random seeds:

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

## Git Workflow

### Branch Strategy

We use a feature-branch workflow with the following main branches:

- **`main`**: Production-ready code, always deployable
- **`dev`**: Integration branch for ongoing development
- **`experiments/<name>`**: Experimental features and model iterations
- **`refactor/<component>`**: Major refactoring efforts (e.g., `refactor-vae`, `refactor-tft`)
- **`feature/<name>`**: New features or enhancements
- **`fix/<name>`**: Bug fixes
- **`docs/<name>`**: Documentation updates

### Creating a Feature Branch

```bash
# 1. Update your local main branch
git checkout main
git pull origin main

# 2. Create a feature branch
git checkout -b feature/add-vae-encoder

# 3. Work on your changes
# ... make changes ...

# 4. Commit your changes (see Commit Convention below)
git add .
git commit -m "feat(models): add VAE encoder with beta-VAE loss"

# 5. Push to remote
git push origin feature/add-vae-encoder

# 6. Create a Pull Request on GitHub
```

### Working with Experimental Branches

For ML experiments (hyperparameter tuning, architecture search):

```bash
# Create an experiment branch
git checkout -b experiments/vae-beta-tuning

# Track experiments in experiments/ directory
# Commit results, not just code
git add experiments/vae-beta-tuning/
git commit -m "exp: test beta values [1.0, 2.0, 4.0] for VAE"

# Merge successful experiments back to dev
git checkout dev
git merge experiments/vae-beta-tuning
```

## Coding Standards

### Code Quality Requirements

All code must meet the following standards:

1. **Type Hints**: All functions must have type annotations
2. **Docstrings**: Google-style docstrings for all public APIs
3. **Testing**: Minimum 80% test coverage for new code
4. **Linting**: Pass black, ruff, and mypy without errors
5. **No Dead Code**: Remove commented-out code and unused imports

### Code Style

We use:
- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **Ruff** for linting
- **mypy** for type checking

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
ruff check src/ tests/ --fix

# Type check
mypy src/atlasfx
```

### Example: Well-Formatted Function

```python
from typing import Optional

import numpy as np
import pandas as pd


def compute_vwap(
    prices: np.ndarray,
    volumes: np.ndarray,
    window: Optional[int] = None,
) -> np.ndarray:
    """
    Compute Volume-Weighted Average Price (VWAP).

    Args:
        prices: Array of prices (N,)
        volumes: Array of volumes (N,)
        window: Rolling window size. If None, compute cumulative VWAP.

    Returns:
        np.ndarray: VWAP values (N,)

    Raises:
        ValueError: If prices and volumes have different lengths
        ValueError: If any volume is negative

    Example:
        >>> prices = np.array([100.0, 101.0, 99.0])
        >>> volumes = np.array([10, 20, 15])
        >>> vwap = compute_vwap(prices, volumes)
        >>> vwap
        array([100.  , 100.67,  99.89])
    """
    if len(prices) != len(volumes):
        raise ValueError("Prices and volumes must have same length")
    if np.any(volumes < 0):
        raise ValueError("Volumes cannot be negative")

    if window is None:
        # Cumulative VWAP
        cumulative_pv = np.cumsum(prices * volumes)
        cumulative_volume = np.cumsum(volumes)
        return cumulative_pv / cumulative_volume
    else:
        # Rolling VWAP
        return pd.Series(prices * volumes).rolling(window).sum() / pd.Series(volumes).rolling(window).sum()
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests (fast, isolated)
│   ├── data/
│   ├── models/
│   └── training/
├── integration/       # Integration tests (slower, end-to-end)
│   └── test_pipeline.py
├── fixtures/          # Test data fixtures
│   ├── sample_tick_data.csv
│   └── sample_features.parquet
└── conftest.py        # Shared fixtures and configuration
```

### Writing Tests

```python
import pytest
import numpy as np
from atlasfx.data.aggregators import compute_vwap


class TestVWAP:
    """Tests for VWAP calculation."""

    def test_vwap_basic(self):
        """Test basic VWAP calculation."""
        prices = np.array([100.0, 101.0, 99.0])
        volumes = np.array([10, 20, 15])
        result = compute_vwap(prices, volumes)
        
        expected = np.array([100.0, 100.666667, 99.888889])
        np.testing.assert_array_almost_equal(result, expected)

    def test_vwap_empty_input(self):
        """Test VWAP with empty arrays."""
        with pytest.raises(ValueError):
            compute_vwap(np.array([]), np.array([]))

    def test_vwap_negative_volume(self):
        """Test VWAP rejects negative volumes."""
        prices = np.array([100.0, 101.0])
        volumes = np.array([10, -5])
        
        with pytest.raises(ValueError, match="negative"):
            compute_vwap(prices, volumes)

    @pytest.mark.parametrize("window", [2, 3, 5])
    def test_vwap_rolling(self, window):
        """Test rolling VWAP with different windows."""
        prices = np.array([100.0, 101.0, 99.0, 102.0, 98.0])
        volumes = np.array([10, 20, 15, 25, 30])
        
        result = compute_vwap(prices, volumes, window=window)
        assert len(result) == len(prices)
        assert not np.isnan(result[-1])  # Last value should be valid
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/data/test_aggregators.py -v

# Run with coverage
pytest tests/ --cov=src/atlasfx --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -m "not slow"

# Run with specific markers
pytest tests/ -m "unit"
```

### Coverage Requirements

- **Global**: Minimum 70% coverage
- **Core modules** (data, models, training): Minimum 85% coverage
- **Utilities**: Minimum 60% coverage

## Documentation

### Docstring Style

We use Google-style docstrings:

```python
def function_name(arg1: type1, arg2: type2) -> return_type:
    """
    Short description (one line).

    Longer description if needed. Can span multiple paragraphs.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ExceptionType: When this exception is raised

    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
```

### Documentation Files

Update documentation when:
- Adding new features: Update `README.md` and `docs/ARCHITECTURE.md`
- Making architectural decisions: Add entry to `DECISIONS.md`
- Changing APIs: Update docstrings and examples

## Pull Request Process

### Before Submitting

1. **Run all checks locally**:
   ```bash
   # Format and lint
   black src/ tests/
   ruff check src/ tests/ --fix
   mypy src/atlasfx
   
   # Run tests
   pytest tests/ -v --cov=src/atlasfx
   ```

2. **Update documentation**:
   - Update README if user-facing changes
   - Add docstrings to new functions
   - Update ARCHITECTURE.md if design changes

3. **Write tests**:
   - Add unit tests for new functions
   - Add integration tests for new features
   - Ensure coverage requirements are met

### PR Template

Title format: `<type>(<scope>): <description>`

Example: `feat(models): add TFT with attention mechanisms`

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added and passing
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review performed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added with >80% coverage
- [ ] All pre-commit hooks pass

## Screenshots (if applicable)
Add screenshots for UI changes

## Additional Notes
Any additional context or notes for reviewers
```

### Code Review Process

1. **Self-Review**: Review your own code before requesting review
2. **Automated Checks**: Ensure CI passes (linting, type checking, tests)
3. **Peer Review**: At least one approval required
4. **Address Feedback**: Respond to all review comments
5. **Merge**: Squash and merge once approved

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no logic change)
- **refactor**: Code refactoring (no feature change)
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **chore**: Maintenance tasks (dependencies, config)
- **ci**: CI/CD changes
- **exp**: Experiment results or configurations

### Scopes

- **data**: Data pipeline
- **models**: Model implementations (VAE, TFT, SAC)
- **training**: Training utilities
- **evaluation**: Evaluation and metrics
- **env**: Trading environment
- **agents**: RL agents
- **config**: Configuration
- **utils**: Utilities

### Examples

```bash
# Feature addition
git commit -m "feat(models): implement VAE encoder with beta-VAE loss"

# Bug fix
git commit -m "fix(data): correct lookahead bias in feature calculation"

# Documentation
git commit -m "docs(architecture): add VAE latent space analysis"

# Refactoring
git commit -m "refactor(training): extract training loop to separate class"

# Experiment
git commit -m "exp(vae): test beta values [1.0, 2.0, 4.0] for latent disentanglement"

# Breaking change
git commit -m "feat(data)!: change aggregation API to support multi-output

BREAKING CHANGE: Aggregators now return dict instead of single value"
```

## Questions?

For questions or clarifications:
- Open an issue on GitHub
- Contact: [Jose Flores Contreras](https://github.com/JoseFloresContreras)

---

**Remember**: Quality over quantity. Write code you'd be proud to defend in a PhD thesis defense.
