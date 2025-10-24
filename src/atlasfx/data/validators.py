"""
AtlasFX Data Validation Module (v3.1)
-------------------------------------
Validates all data structures (tick, OHLC, feature) against
the formal schema defined in configs/schema.yaml.

Supports:
✅ Schema-driven validation
✅ Severity-based error reporting
✅ Cross-column & quality rules
✅ CLI and programmatic use
"""

from __future__ import annotations

import ast
import builtins
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from atlasfx.utils.logging import log


# =============================================================================
# Exceptions
# =============================================================================

class ValidationError(Exception):
    """Raised when a validation check fails with severity 'error'."""
    pass


# =============================================================================
# DataValidator Class
# =============================================================================

class DataValidator:
    """Validates DataFrames against the AtlasFX schema."""

    def __init__(self, schema_path: str = "configs/schema.yaml") -> None:
        self.schema_path = schema_path
        self.schema = self._load_schema(schema_path)

    def _load_schema(self, schema_path: str) -> dict[str, Any]:
        """Load the YAML schema into memory."""
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            msg = f"Schema file not found: {schema_path}"
            log.critical(msg, also_print=True)
            raise

    # -------------------------------------------------------------------------
    # Generic schema-driven validator
    # -------------------------------------------------------------------------
    def validate(self, df: pd.DataFrame, schema_key: str) -> tuple[bool, list[str]]:
        """
        Validate a DataFrame against a schema section (tick_data, ohlc_data, feature_matrix).
        """
        schema = self.schema.get(schema_key, {})
        if not schema:
            raise ValueError(f"Schema section '{schema_key}' not found in {self.schema_path}")

        errors, warnings = [], []

        # --- Validate required columns ---
        required_cols = schema.get("required_columns", {})
        for col_name, col_spec in required_cols.items():
            if col_name not in df.columns:
                errors.append(f"Missing required column: {col_name}")
                continue

            col = df[col_name]
            col_type = col_spec.get("type", "")
            nullable = col_spec.get("nullable", False)

            # Nullability
            if not nullable and col.isna().any():
                errors.append(f"Column '{col_name}' contains NaN but is not nullable")

            # Type validation
            if "float" in col_type and not pd.api.types.is_float_dtype(col):
                errors.append(f"Column '{col_name}' type mismatch: expected float, got {col.dtype}")
            elif "int" in col_type and not pd.api.types.is_integer_dtype(col):
                errors.append(f"Column '{col_name}' type mismatch: expected int, got {col.dtype}")
            elif "datetime" in col_type and not pd.api.types.is_datetime64_any_dtype(col):
                errors.append(f"Column '{col_name}' type mismatch: expected datetime, got {col.dtype}")

            # Value constraints
            for constraint in col_spec.get("constraints", []):
                if not isinstance(constraint, dict):
                    continue
                for key, value in constraint.items():
                    if key == "min_value" and (col < value).any():
                        errors.append(f"Column '{col_name}' has values < {value} (min={col.min()})")
                    elif key == "max_value" and (col > value).any():
                        errors.append(f"Column '{col_name}' has values > {value} (max={col.max()})")
                    elif key == "abs_value_lt" and (col.abs() > value).any():
                        warnings.append(f"Column '{col_name}' exceeds |{value}| threshold")

        # --- Cross-column constraints ---
        for rule in schema.get("cross_column_constraints", []):
            name = rule.get("name", "unnamed_rule")
            expr = rule.get("rule", "")
            severity = rule.get("severity", "error")

            try:
                local_env = {**{c: df[c] for c in df.columns if c in expr}, "np": np}
                result = self._safe_eval(expr, local_env)
                if isinstance(result, pd.Series):
                    failed = (~result).sum()
                    if failed > 0:
                        msg = f"Rule '{name}' failed {failed} times: {expr}"
                        (errors if severity == "error" else warnings).append(msg)
                elif not bool(result):
                    msg = f"Rule '{name}' failed: {expr}"
                    (errors if severity == "error" else warnings).append(msg)
            except Exception as e:
                warnings.append(f"Rule '{name}' could not be evaluated ({e})")

        # --- Quality checks ---
        for qc in schema.get("quality_checks", []):
            name = qc.get("name", "unnamed_check")
            expr = qc.get("rule", "")
            severity = qc.get("severity", "error")

            try:
                local_env = {"df": df, "np": np, "pd": pd}
                result = self._safe_eval(expr, local_env)
                if not bool(result):
                    msg = f"Quality check '{name}' failed: {expr}"
                    (errors if severity == "error" else warnings).append(msg)
            except Exception as e:
                warnings.append(f"Quality check '{name}' evaluation failed ({e})")

        # --- Summary logging ---
        for w in warnings:
            log.warning(f"⚠️ {w}")
        for e in errors:
            log.error(f"❌ {e}")

        is_valid = not errors
        return is_valid, errors + warnings

    # -------------------------------------------------------------------------
    # Specific shortcuts
    # -------------------------------------------------------------------------
    def validate_tick_data(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Shortcut for tick data validation."""
        return self.validate(df, "tick_data")

    def validate_ohlc_data(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Shortcut for OHLC data validation."""
        return self.validate(df, "ohlc_data")

    def validate_feature_matrix(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Shortcut for feature matrix validation."""
        return self.validate(df, "feature_matrix")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_eval(expr: str, local_env: dict[str, Any]) -> Any:
        """Safely evaluate validation expressions."""
        # Restrict eval to arithmetic and logical ops
        allowed_nodes = (
            ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
            ast.Name, ast.Load, ast.Call, ast.Constant, ast.Attribute
        )
        parsed = ast.parse(expr, mode="eval")
        if not all(isinstance(n, allowed_nodes) for n in ast.walk(parsed)):
            raise ValueError(f"Unsafe expression: {expr}")
        return eval(compile(parsed, "<validation>", "eval"), {"__builtins__": {}}, local_env)


# =============================================================================
# Public API
# =============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    data_type: str = "tick_data",
    schema_path: str = "configs/schema.yaml",
) -> None:
    """Validate DataFrame and raise ValidationError if invalid."""
    validator = DataValidator(schema_path)
    if data_type not in {"tick_data", "ohlc_data", "feature_matrix"}:
        raise ValueError(f"Unsupported data type: {data_type}")

    is_valid, errors = validator.validate(df, data_type)
    if not is_valid:
        message = f"Validation failed for {data_type}:\n" + "\n".join(f"  - {e}" for e in errors)
        log.error(message, also_print=True)
        raise ValidationError(message)

    log.info(f"✅ Validation passed for {data_type} ({len(df)} rows)", also_print=True)


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate a dataset using AtlasFX schema.")
    parser.add_argument("--sample", type=str, help="Path to CSV or Parquet sample file.")
    parser.add_argument(
        "--type",
        type=str,
        default="tick_data",
        choices=["tick_data", "ohlc_data", "feature_matrix"],
        help="Schema type to validate.",
    )
    parser.add_argument("--schema", type=str, default="configs/schema.yaml", help="Path to schema file.")
    args = parser.parse_args()

    if not args.sample:
        print("❌ No sample file provided. Use --sample <path>")
        exit(1)

    # Load file
    sample_path = args.sample
    if sample_path.endswith(".csv"):
        df = pd.read_csv(sample_path, parse_dates=["timestamp"])
    elif sample_path.endswith(".parquet"):
        df = pd.read_parquet(sample_path)
    else:
        print(f"❌ Unsupported file format: {sample_path}")
        exit(1)

    try:
        validate_dataframe(df, data_type=args.type, schema_path=args.schema)
        print(f"✅ Validation passed for {args.type}")
    except ValidationError as e:
        print(f"❌ Validation failed:\n{e}")
        exit(1)
