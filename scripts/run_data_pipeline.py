#!/usr/bin/env python3
"""
AtlasFX Data Processing Pipeline (v3.2)
----------------------------------------
Executes the full multi-stage data pipeline dynamically
based on YAML configuration, with built-in validation,
logging controls, and path security.

Key features:
✅ Secure path resolution via atlasfx.utils.pathing
✅ Config validation and dependency checks
✅ Dynamic modular pipeline (step_map)
✅ Granular log-level control
✅ Full error propagation with context
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Any, Optional

# === Imports: AtlasFX modules ===
from atlasfx.data.loaders import run_merge
from atlasfx.data.cleaning import run_clean
from atlasfx.data.aggregation import run_aggregate
from atlasfx.data.splitters import run_split
from atlasfx.data.winsorization import run_winsorize
from atlasfx.data.featurization import run_featurize
from atlasfx.data.normalization import run_normalize
from atlasfx.evaluation.visualizers import run_visualize
from atlasfx.utils.logging import log
from atlasfx.utils.pathing import cd_repo_root, resolve_path, resolve_repo_root


# ============================================================
# CONFIG LOADING AND VALIDATION
# ============================================================

def load_pipeline_config(config_file: Path) -> dict[str, Any]:
    """Load and validate the main pipeline configuration from YAML."""
    try:
        config_file = config_file.resolve(strict=True)
    except FileNotFoundError:
        msg = f"❌ Configuration file not found: {config_file}"
        log.critical(msg, also_print=True)
        raise

    try:
        with open(config_file, "r", encoding="utf-8") as file:
            cfg = yaml.safe_load(file)
    except yaml.YAMLError as e:
        msg = f"❌ Error parsing YAML file: {e}"
        log.critical(msg, also_print=True)
        raise

    required_top_keys = ["steps", "merge", "aggregate", "output_directory"]
    missing = [k for k in required_top_keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys in pipeline config: {missing}")

    out_dir = Path(cfg["output_directory"]).expanduser().resolve()
    if not out_dir.exists():
        os.makedirs(out_dir, exist_ok=True)
        log.info(f"📁 Created output directory: {out_dir}")

    cfg["output_directory"] = str(out_dir)
    return cfg


# ============================================================
# STEP VALIDATION AND DEPENDENCIES
# ============================================================

def validate_and_order_steps(steps_to_execute: list[str]) -> list[str]:
    """Validate and reorder steps according to dependency rules."""
    valid_steps = [
        "merge",
        "clean_ticks",
        "aggregate",
        "split",
        "winsorize",
        "clean_aggregated",
        "featurize",
        "clean_featurized",
        "normalize",
        "visualize",
    ]

    unknown = [s for s in steps_to_execute if s not in valid_steps]
    if unknown:
        log.warning(f"⚠️ Unknown step(s) detected in config: {unknown}")

    ordered = [s for s in valid_steps if s in steps_to_execute]
    for i, step in enumerate(ordered):
        if i > 0:
            prev = valid_steps[valid_steps.index(step) - 1]
            if prev not in ordered:
                log.warning(
                    f"⚠️ Step '{step}' depends on '{prev}', which is missing. "
                    "It may rely on previous outputs."
                )

    if ordered != steps_to_execute:
        log.info(f"🔄 Reordered steps to: {', '.join(ordered)}")

    return ordered


# ============================================================
# INPUT VALIDATION AND DISCOVERY
# ============================================================

def get_expected_input_files(step_name: str, cfg: dict[str, Any]) -> Optional[list[str]]:
    """Determine and validate expected input files for each step."""

    def ensure_files_exist(paths: list[str]) -> list[str]:
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            msg = f"Required input files not found: {missing}"
            log.critical(msg, also_print=True)
            raise FileNotFoundError(msg)
        return sorted(paths)

    out_dir = cfg["output_directory"]

    if step_name == "merge":
        return None

    elif step_name in ["clean_ticks", "aggregate"]:
        pairs = [s["symbol"] for s in cfg["merge"].get("pairs", [])]
        instruments = [s["symbol"] for s in cfg["merge"].get("instruments", [])]
        files = [
            *(os.path.join(out_dir, f"{s}-pair_ticks.parquet") for s in pairs),
            *(os.path.join(out_dir, f"{s}-instrument_ticks.parquet") for s in instruments),
        ]
        return ensure_files_exist(files)

    elif step_name == "split":
        time_window = cfg["aggregate"]["time_window"]
        base_name = cfg["aggregate"]["output_filename"]
        file_path = os.path.join(out_dir, f"{time_window}_{base_name}.parquet")
        return ensure_files_exist([file_path])[0]

    elif step_name in [
        "winsorize",
        "clean_aggregated",
        "featurize",
        "clean_featurized",
        "normalize",
        "visualize",
    ]:
        time_window = cfg["aggregate"]["time_window"]
        base_name = cfg["aggregate"]["output_filename"]
        suffixes = ["_train.parquet", "_val.parquet", "_test.parquet"]
        files = [os.path.join(out_dir, f"{time_window}_{base_name}{s}") for s in suffixes]
        return ensure_files_exist(files)

    raise ValueError(f"Unknown pipeline step: {step_name}")


# ============================================================
# CONFIGURATION BUILDERS FOR EACH STEP
# ============================================================

def generate_merge_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "pairs": cfg["merge"].get("pairs", []),
        "instruments": cfg["merge"].get("instruments", []),
        "output_directory": cfg["output_directory"],
        "time_column": cfg["merge"]["time_column"],
    }


def generate_clean_ticks_config(cfg: dict[str, Any], inputs: list[str]) -> dict[str, Any]:
    return {
        "pipeline_stage": "ticks",
        "stages": {
            "ticks": {
                "input_files": inputs,
                "output_directory": cfg["output_directory"],
                "time_column": cfg["merge"]["time_column"],
            }
        },
    }


def generate_aggregate_config(cfg: dict[str, Any], inputs: list[str]) -> dict[str, Any]:
    return {
        "time_window": cfg["aggregate"]["time_window"],
        "input_files": inputs,
        "output_directory": cfg["output_directory"],
        "aggregators": cfg["aggregate"]["aggregators"],
        "output_filename": cfg["aggregate"]["output_filename"],
    }


def generate_split_config(cfg: dict[str, Any], input_file: str) -> dict[str, Any]:
    return {
        "input_file": input_file,
        "output_directory": cfg["output_directory"],
        "split": cfg.get("split", {"train": 0.7, "val": 0.15, "test": 0.15}),
    }


def generate_winsorize_config(cfg: dict[str, Any], inputs: list[str]) -> dict[str, Any]:
    return {
        "input_files": inputs,
        "output_directory": cfg["output_directory"],
        "winsorization_configs": cfg.get("winsorize", []),
        "time_window": cfg["aggregate"]["time_window"],
    }


def generate_clean_stage_config(stage: str, cfg: dict[str, Any], inputs: list[str]) -> dict[str, Any]:
    return {
        "pipeline_stage": stage,
        "stages": {
            stage: {
                "input_files": inputs,
                "output_directory": cfg["output_directory"],
                "time_column": "start_time",
            }
        },
    }


def generate_featurize_config(cfg: dict[str, Any], inputs: list[str]) -> dict[str, Any]:
    return {
        "input_files": inputs,
        "output_directory": cfg["output_directory"],
        "featurizers": cfg["featurize"]["featurizers"],
    }


def generate_normalize_config(cfg: dict[str, Any], inputs: list[str]) -> dict[str, Any]:
    return {
        "input_files": inputs,
        "output_directory": cfg["output_directory"],
        "time_window": cfg["aggregate"]["time_window"],
        "clip_threshold": cfg["normalize"]["clip_threshold"],
    }


def generate_visualize_config(cfg: dict[str, Any], inputs: list[str]) -> dict[str, Any]:
    return {
        "input_files": inputs,
        "output_directory": cfg["output_directory"],
        "time_column": "start_time",
        "time_window": cfg["aggregate"]["time_window"],
    }


# ============================================================
# CORE EXECUTION
# ============================================================

def run_pipeline_step(step: str, func, cfg: dict[str, Any]) -> None:
    """Execute a pipeline step safely."""
    try:
        log.info("=" * 60, also_print=True)
        log.info(f"🚀 Running {step} step...", also_print=True)
        func(cfg)
        log.info(f"✅ Step '{step}' completed successfully!", also_print=True)
    except Exception as e:
        log.error(f"❌ Step '{step}' failed: {e}", also_print=True)
        raise


def run_pipeline(config_file: str, log_level: str = "INFO") -> None:
    """Execute all pipeline steps sequentially."""
    # --- Ensure we’re running from repo root ---
    cd_repo_root()
    log.set_level(log_level.upper())
    log.info(f"📂 Running from repo root: {resolve_repo_root()}")

    cfg = load_pipeline_config(resolve_path(config_file))
    steps = validate_and_order_steps(cfg.get("steps", []))
    if not steps:
        log.error("❌ No steps specified in configuration.", also_print=True)
        return

    step_map = {
        "merge": (run_merge, generate_merge_config),
        "clean_ticks": (run_clean, generate_clean_ticks_config),
        "aggregate": (run_aggregate, generate_aggregate_config),
        "split": (run_split, generate_split_config),
        "winsorize": (run_winsorize, generate_winsorize_config),
        "clean_aggregated": (run_clean, lambda c, i: generate_clean_stage_config("aggregated", c, i)),
        "featurize": (run_featurize, generate_featurize_config),
        "clean_featurized": (run_clean, lambda c, i: generate_clean_stage_config("featurized", c, i)),
        "normalize": (run_normalize, generate_normalize_config),
        "visualize": (run_visualize, generate_visualize_config),
    }

    executed, success = [], True
    for idx, step in enumerate(steps, 1):
        log.info(f"\n📊 Step {idx}/{len(steps)}: {step.upper()}", also_print=True)
        try:
            inputs = get_expected_input_files(step, cfg)
            func, gen = step_map[step]
            conf = gen(cfg, inputs) if step != "split" else gen(cfg, str(inputs))
            run_pipeline_step(step, func, conf)
            executed.append(step)
        except Exception as e:
            log.error(f"❌ Pipeline failed at '{step}': {e}", also_print=True)
            success = False
            break

    # Summary
    log.info("\n" + "=" * 60, also_print=True)
    log.info("🎯 PIPELINE SUMMARY", also_print=True)
    log.info("=" * 60, also_print=True)
    log.info(f"📋 Executed: {', '.join(executed)}", also_print=True)
    log.info(f"📊 Success: {success}", also_print=True)
    log.info("=" * 60, also_print=True)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main() -> None:
    """CLI entrypoint for the AtlasFX pipeline."""
    parser = argparse.ArgumentParser(description="Run the AtlasFX data pipeline.")
    parser.add_argument("--config", type=str, help="Path to YAML config.", default="configs/data_pipeline.yaml")
    parser.add_argument("--log-level", type=str, default="INFO", help="Set log level (DEBUG, INFO, ERROR).")
    args = parser.parse_args()

    run_pipeline(args.config, log_level=args.log_level)
    log.info("\n📋 Pipeline execution completed!")


if __name__ == "__main__":
    main()
