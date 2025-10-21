"""
Data validation module for AtlasFX.

Validates Level 1 tick data and aggregated features according to the schema
defined in configs/schema.yaml.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from atlasfx.utils.logging import log


class ValidationError(Exception):
    """Custom exception for data validation errors."""

    pass


class DataValidator:
    """
    Validator for tick data and aggregated features.

    Validates data against schema defined in configs/schema.yaml.
    """

    def __init__(self, schema_path: str = "configs/schema.yaml") -> None:
        """
        Initialize validator with schema.

        Args:
            schema_path: Path to schema YAML file
        """
        self.schema = self._load_schema(schema_path)

    def _load_schema(self, schema_path: str) -> dict[str, Any]:
        """
        Load validation schema from YAML file.

        Args:
            schema_path: Path to schema file

        Returns:
            dict: Schema configuration

        Raises:
            FileNotFoundError: If schema file not found
        """
        try:
            with open(schema_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            error_msg = f"Schema file not found: {schema_path}"
            log.error(error_msg)
            raise FileNotFoundError(error_msg)

    def validate_tick_data(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate Level 1 tick data.

        Checks:
        - Required columns present
        - Data types correct
        - Value constraints satisfied
        - Cross-column constraints (bid <= ask)
        - Timestamp monotonicity

        Args:
            df: DataFrame with tick data

        Returns:
            tuple: (is_valid, list_of_errors)
        """
        errors = []
        schema = self.schema.get("tick_data", {})
        required_columns = schema.get("required_columns", {})

        # Check required columns
        for col_name, col_spec in required_columns.items():
            if col_name not in df.columns:
                errors.append(f"Missing required column: {col_name}")
                continue

            col = df[col_name]

            # Check nullable
            if not col_spec.get("nullable", False) and col.isna().any():
                errors.append(f"Column '{col_name}' contains NaN values but is not nullable")

            # Check data type
            expected_type = col_spec.get("type", "")
            if "float" in expected_type and not pd.api.types.is_float_dtype(col):
                errors.append(
                    f"Column '{col_name}' has wrong type: "
                    f"expected {expected_type}, got {col.dtype}"
                )
            elif "int" in expected_type and not pd.api.types.is_integer_dtype(col):
                errors.append(
                    f"Column '{col_name}' has wrong type: "
                    f"expected {expected_type}, got {col.dtype}"
                )
            elif "datetime" in expected_type and not pd.api.types.is_datetime64_any_dtype(col):
                errors.append(
                    f"Column '{col_name}' has wrong type: "
                    f"expected {expected_type}, got {col.dtype}"
                )

            # Check constraints
            constraints = col_spec.get("constraints", [])
            for constraint in constraints:
                if isinstance(constraint, dict):
                    for key, value in constraint.items():
                        if key == "min_value":
                            # Only check numeric columns with numeric constraints
                            if pd.api.types.is_numeric_dtype(col) and isinstance(
                                value, (int, float)
                            ):
                                if (col < value).any():
                                    errors.append(
                                        f"Column '{col_name}' has values < {value} "
                                        f"(min: {col.min()})"
                                    )
                        elif key == "max_value":
                            # Only check numeric columns with numeric constraints
                            if pd.api.types.is_numeric_dtype(col) and isinstance(
                                value, (int, float)
                            ):
                                if (col > value).any():
                                    errors.append(
                                        f"Column '{col_name}' has values > {value} "
                                        f"(max: {col.max()})"
                                    )
                        elif key == "monotonic_increasing":
                            if value and not col.is_monotonic_increasing:
                                errors.append(
                                    f"Column '{col_name}' is not monotonically increasing"
                                )
                        # Ignore timezone and other non-validation constraints

        # Check cross-column constraints
        cross_constraints = schema.get("cross_column_constraints", [])
        for constraint in cross_constraints:
            name = constraint.get("name", "unknown")
            rule = constraint.get("rule", "")
            severity = constraint.get("severity", "error")

            if name == "no_crossed_spreads":
                if "bid" in df.columns and "ask" in df.columns:
                    crossed = df["ask"] < df["bid"]
                    if crossed.any():
                        msg = f"Found {crossed.sum()} crossed spreads (ask < bid)"
                        if severity == "error":
                            errors.append(msg)
                        else:
                            log.warning(msg)

            elif name == "reasonable_spread":
                if "bid" in df.columns and "ask" in df.columns:
                    mid = (df["ask"] + df["bid"]) / 2
                    spread_pct = (df["ask"] - df["bid"]) / mid
                    large_spreads = spread_pct > 0.01
                    if large_spreads.any():
                        msg = (
                            f"Found {large_spreads.sum()} bars with spread > 1% "
                            f"(max: {spread_pct.max():.2%})"
                        )
                        if severity == "error":
                            errors.append(msg)
                        else:
                            log.warning(msg)

        # Check quality
        quality_checks = schema.get("quality_checks", [])
        for check in quality_checks:
            name = check.get("name", "unknown")
            severity = check.get("severity", "error")

            if name == "no_duplicates":
                if "timestamp" in df.columns:
                    duplicates = df["timestamp"].duplicated().sum()
                    if duplicates > 0:
                        msg = f"Found {duplicates} duplicate timestamps"
                        if severity == "error":
                            errors.append(msg)
                        else:
                            log.warning(msg)

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_ohlc_data(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate aggregated OHLC data.

        Checks:
        - OHLC columns present
        - High >= Low, High >= Open/Close
        - Positive prices
        - Monotonic timestamps

        Args:
            df: DataFrame with OHLC data

        Returns:
            tuple: (is_valid, list_of_errors)
        """
        errors = []
        schema = self.schema.get("ohlc_data", {})
        required_columns = schema.get("required_columns", {})

        # Check required columns
        for col_name in required_columns.keys():
            if col_name not in df.columns:
                errors.append(f"Missing required column: {col_name}")

        # Check OHLC relationships
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            # High >= Low
            invalid_high_low = df["high"] < df["low"]
            if invalid_high_low.any():
                errors.append(
                    f"Found {invalid_high_low.sum()} bars where high < low"
                )

            # High >= Open
            invalid_high_open = df["high"] < df["open"]
            if invalid_high_open.any():
                errors.append(
                    f"Found {invalid_high_open.sum()} bars where high < open"
                )

            # High >= Close
            invalid_high_close = df["high"] < df["close"]
            if invalid_high_close.any():
                errors.append(
                    f"Found {invalid_high_close.sum()} bars where high < close"
                )

            # Low <= Open
            invalid_low_open = df["low"] > df["open"]
            if invalid_low_open.any():
                errors.append(
                    f"Found {invalid_low_open.sum()} bars where low > open"
                )

            # Low <= Close
            invalid_low_close = df["low"] > df["close"]
            if invalid_low_close.any():
                errors.append(
                    f"Found {invalid_low_close.sum()} bars where low > close"
                )

        # Check positive prices
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                negative = df[col] <= 0
                if negative.any():
                    errors.append(
                        f"Found {negative.sum()} non-positive values in '{col}'"
                    )

        # Check timestamp monotonicity
        if "timestamp" in df.columns:
            if not df["timestamp"].is_monotonic_increasing:
                errors.append("Timestamps are not monotonically increasing")

        is_valid = len(errors) == 0
        return is_valid, errors

    def validate_feature_matrix(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """
        Validate feature matrix for VAE input.

        Checks:
        - Required features present
        - No infinite values
        - Returns in reasonable range
        - Non-negative volume/spread

        Args:
            df: DataFrame with feature matrix

        Returns:
            tuple: (is_valid, list_of_errors)
        """
        errors = []
        schema = self.schema.get("feature_matrix", {})
        required_columns = schema.get("required_columns", {})

        # Check required columns
        for col_name in required_columns.keys():
            if col_name not in df.columns:
                errors.append(f"Missing required column: {col_name}")

        # Check for infinite values
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                errors.append(f"Column '{col}' contains infinite values")

        # Check returns range (warning only)
        if "returns" in df.columns:
            extreme_returns = df["returns"].abs() > 1.0
            if extreme_returns.any():
                log.warning(
                    f"Found {extreme_returns.sum()} extreme returns (>100%): "
                    f"max={df['returns'].max():.2%}, min={df['returns'].min():.2%}"
                )

        # Check spread non-negative
        if "spread" in df.columns:
            negative_spread = df["spread"] < 0
            if negative_spread.any():
                errors.append(
                    f"Found {negative_spread.sum()} negative spread values"
                )

        # Check volume non-negative
        if "volume" in df.columns:
            negative_volume = df["volume"] < 0
            if negative_volume.any():
                errors.append(
                    f"Found {negative_volume.sum()} negative volume values"
                )

        is_valid = len(errors) == 0
        return is_valid, errors


def validate_dataframe(
    df: pd.DataFrame,
    data_type: str = "tick_data",
    schema_path: str = "configs/schema.yaml",
) -> None:
    """
    Validate DataFrame and raise exception if invalid.

    Args:
        df: DataFrame to validate
        data_type: Type of data ('tick_data', 'ohlc_data', 'feature_matrix')
        schema_path: Path to schema file

    Raises:
        ValidationError: If validation fails
    """
    validator = DataValidator(schema_path)

    if data_type == "tick_data":
        is_valid, errors = validator.validate_tick_data(df)
    elif data_type == "ohlc_data":
        is_valid, errors = validator.validate_ohlc_data(df)
    elif data_type == "feature_matrix":
        is_valid, errors = validator.validate_feature_matrix(df)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    if not is_valid:
        error_msg = f"Validation failed for {data_type}:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        log.error(error_msg)
        raise ValidationError(error_msg)

    log.info(f"✅ Validation passed for {data_type} ({len(df)} rows)")


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate tick data")
    parser.add_argument("--sample", type=str, help="Path to sample CSV file")
    parser.add_argument(
        "--type",
        type=str,
        default="tick_data",
        choices=["tick_data", "ohlc_data", "feature_matrix"],
        help="Type of data to validate",
    )
    args = parser.parse_args()

    if args.sample:
        df = pd.read_csv(args.sample, parse_dates=["timestamp"])
        try:
            validate_dataframe(df, data_type=args.type)
            print(f"✅ Validation passed for {args.type}")
        except ValidationError as e:
            print(f"❌ Validation failed:\n{e}")
            exit(1)
    else:
        print("No sample file provided. Use --sample <path>")
