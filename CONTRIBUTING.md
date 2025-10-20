# Contributing to Atlas FX MVP

Thank you for considering contributing to this project! This document provides guidelines for contributing code, documentation, and other improvements.

## Code of Conduct

- Be respectful and constructive in all interactions
- Focus on the code and ideas, not individuals
- Welcome newcomers and help them learn

## Getting Started

### Prerequisites

1. Python 3.8 or higher
2. Git for version control
3. Virtual environment tool (venv, conda, etc.)

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/JoseFloresContreras/atlasfx-mvp.git
cd atlasfx-mvp

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r data-pipeline/requirements.txt
pip install -r agent/TD3/requirements.txt
```

## Code Style Guidelines

### Python

We follow PEP 8 with these modifications:
- **Line length**: Maximum 120 characters (instead of 79)
- **Indentation**: 4 spaces (no tabs for new code)
- **String quotes**: Use single quotes unless double quotes avoid escaping

### Documentation

- **All functions** must have docstrings with:
  - Brief description
  - Args section with types
  - Returns section with type
  - Raises section if applicable
  
Example:
```python
def process_data(input_file: str, output_dir: str) -> pd.DataFrame:
    """
    Process raw data and save results.
    
    Args:
        input_file (str): Path to input file
        output_dir (str): Directory for output files
        
    Returns:
        pd.DataFrame: Processed dataframe
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If data format is invalid
    """
    pass
```

### Import Organization

Order imports as follows:
1. Standard library imports
2. Third-party imports
3. Local application imports

Separate each group with a blank line.

```python
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from logger import log
from utils import helper_function
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write clear, focused commits
- Test your changes thoroughly
- Update documentation as needed

### 3. Test Your Changes

```bash
# For data pipeline
cd data-pipeline
python -m py_compile *.py  # Check syntax

# For agent
cd agent/TD3
python -m py_compile *.py  # Check syntax
```

### 4. Commit Guidelines

Write meaningful commit messages:

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example:
```
feat: add RSI indicator to featurizers

Added Relative Strength Index calculation with configurable
window size. Includes proper NaN handling for incomplete windows.

Closes #123
```

### 5. Submit Pull Request

1. Push your branch to GitHub
2. Create a Pull Request with:
   - Clear title and description
   - Reference related issues
   - List of changes made
   - Testing performed

## Project-Specific Guidelines

### Data Pipeline

When adding new pipeline steps:

1. Create new module in `data-pipeline/`
2. Follow existing naming patterns (verb-based: merge, clean, etc.)
3. Implement `run_<step_name>(config)` function
4. Add configuration schema to `pipeline.yaml`
5. Update `pipeline.py` to include new step
6. Add docstrings for all functions
7. Use the logger for all output: `from logger import log`

### Agent Development

When modifying the agent:

1. Maintain compatibility with gym API
2. Test with both train and validation environments
3. Document any hyperparameter changes
4. Preserve original TD3 implementation structure

### Featurizers and Aggregators

When adding new featurizers/aggregators:

1. Follow the existing function signature
2. Handle empty DataFrames gracefully
3. Return NaN for missing/invalid data
4. Add comprehensive docstring
5. Include example configuration in `pipeline.yaml`

Example featurizer:
```python
def my_indicator(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate custom indicator.
    
    Args:
        dataframe: Input dataframe with time index
        config: Configuration dictionary with:
            - window_size (int): Rolling window size
            
    Returns:
        pd.DataFrame: DataFrame with indicator columns
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    window_size = config.get('window_size', 14)
    # Your implementation here
    
    return result_df
```

## What to Contribute

### Good First Contributions

- Fix typos in documentation
- Improve error messages
- Add code comments for complex logic
- Update README with examples
- Fix PEP 8 violations

### Intermediate Contributions

- Add new featurizers or aggregators
- Improve logging and error handling
- Optimize performance
- Add input validation

### Advanced Contributions

- Add new pipeline steps
- Improve agent architecture
- Add testing framework
- Optimize memory usage
- Add new algorithms

## Testing

Currently, testing is done manually. When contributing:

1. Test with sample data
2. Verify error handling
3. Check edge cases
4. Ensure no regression in existing functionality

Future: We plan to add automated testing with pytest.

## Documentation

When contributing, update relevant documentation:

- `README.md` for user-facing changes
- `CODE_QUALITY.md` for code standards
- Inline comments for complex algorithms
- Docstrings for all public functions

## Questions?

If you have questions:

1. Check existing documentation
2. Review similar implementations in the codebase
3. Open an issue for clarification
4. Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

Thank you for contributing to Atlas FX MVP!
