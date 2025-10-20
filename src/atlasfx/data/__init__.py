"""
Data management module for AtlasFX.

This module provides functionality for loading, processing, validating,
and preparing financial tick data for model training.

Submodules:
- loaders: Data loading and merging utilities
- aggregators: Time-based aggregation functions
- aggregation: Aggregation orchestration
- cleaning: Data cleaning utilities
- normalization: Feature normalization
- winsorization: Outlier handling
- featurization: Feature engineering orchestration
- featurizers: Feature calculation functions
- splitters: Train/val/test splitting
"""

__all__ = [
    "loaders",
    "aggregators",
    "aggregation",
    "cleaning",
    "normalization",
    "winsorization",
    "featurization",
    "featurizers",
    "splitters",
]
