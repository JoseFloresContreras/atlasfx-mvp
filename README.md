# Atlas FX MVP

A comprehensive forex trading system combining data processing pipeline and reinforcement learning agents.

## Project Structure

```
atlasfx-mvp/
├── data-pipeline/          # Data processing pipeline
│   ├── pipeline.py         # Main orchestration script
│   ├── pipeline.yaml       # Pipeline configuration
│   ├── merge.py           # CSV merging module
│   ├── clean.py           # Data cleaning module
│   ├── aggregate.py       # Time-series aggregation
│   ├── split.py           # Train/val/test splitting
│   ├── winsorize.py       # Outlier handling
│   ├── featurize.py       # Feature engineering
│   ├── featurizers.py     # Feature computation functions
│   ├── aggregators.py     # Aggregation functions
│   ├── normalize.py       # Data normalization
│   ├── visualize.py       # Data visualization
│   └── logger.py          # Custom logging module
│
└── agent/
    └── TD3/               # Twin Delayed DDPG agent
        ├── main.py        # Training script
        ├── env.py         # Forex trading environment
        ├── TD3.py         # TD3 algorithm implementation
        ├── DDPG.py        # DDPG baseline
        ├── OurDDPG.py     # Tuned DDPG variant
        ├── utils.py       # Replay buffer utilities
        └── README.md      # TD3 algorithm documentation

```

## Data Pipeline

The data pipeline processes raw forex tick data through multiple stages:

1. **Merge**: Combine CSV files from different sources
2. **Clean (Ticks)**: Clean raw tick data, handle gaps
3. **Aggregate**: Resample to different time windows (5min, 1H, etc.)
4. **Split**: Divide into train/validation/test sets
5. **Winsorize**: Handle outliers using percentile clipping
6. **Clean (Aggregated)**: Second cleaning pass on aggregated data
7. **Featurize**: Generate technical indicators and features
8. **Clean (Featurized)**: Final cleaning pass
9. **Normalize**: Standardize features using z-score normalization
10. **Visualize**: Generate data quality reports

### Running the Pipeline

```bash
cd data-pipeline
python pipeline.py
```

Configuration is managed through `pipeline.yaml`.

### Requirements

```bash
pip install -r data-pipeline/requirements.txt
```

## Trading Agent

The TD3 (Twin Delayed Deep Deterministic Policy Gradients) agent learns to trade forex pairs using reinforcement learning.

### Environment

- **State Space**: Market features + portfolio positions + profits
- **Action Space**: Continuous actions for position sizing
- **Reward**: Based on portfolio value changes

### Training

```bash
cd agent/TD3
python main.py --env ForexTradingEnv --policy TD3
```

## Development

### Code Quality Standards

This project follows Python best practices:

- PEP 8 style guide (with 120 character line limit)
- Comprehensive docstrings for all functions
- Type hints where applicable
- Structured error handling and logging
- No hardcoded credentials or sensitive data

### Git Workflow

Excluded from version control (see `.gitignore`):
- `__pycache__/` and `.pyc` files
- `data/` directory (raw and processed data)
- `logs/` directory
- Virtual environments
- IDE-specific files

## License

See individual component licenses:
- TD3 implementation: See `agent/TD3/LICENSE`
- Data pipeline: Custom implementation

## Citation

If using the TD3 implementation, please cite:

```bibtex
@inproceedings{fujimoto2018addressing,
  title={Addressing Function Approximation Error in Actor-Critic Methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle={International Conference on Machine Learning},
  pages={1582--1591},
  year={2018}
}
```

## Contributing

When contributing:
1. Follow existing code style and structure
2. Add docstrings to all new functions
3. Update relevant documentation
4. Test changes thoroughly before submitting
5. Keep commits focused and descriptive
