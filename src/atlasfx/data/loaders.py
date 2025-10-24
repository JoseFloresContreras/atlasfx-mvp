"""
AtlasFX Data Loader & Merger (v3.2)
-----------------------------------
Handles loading, validation, and merging of raw CSV tick data
into canonical Parquet datasets.

Key features:
✅ Robust path resolution via atlasfx.utils.pathing
✅ Memory-efficient float32 typing
✅ TQDM progress tracking
✅ Detailed logging and summary of skipped files
✅ Safe merge and persistence with Snappy compression
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import pandas as pd
from tqdm import tqdm

from atlasfx.utils.logging import log
from atlasfx.utils.pathing import resolve_path, cd_repo_root  # ✅ nuevo


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _resolve_folder_path(folder_path: str | Path) -> Path:
    """Resolve folder path using pathing helper."""
    folder = resolve_path(folder_path)  # ✅ reemplaza todo el bloque anterior

    if not folder.exists():
        msg = f"❌ Folder not found: {folder}"
        log.critical(msg, also_print=True)
        raise FileNotFoundError(msg)

    return folder


def _convert_numeric_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns to float32 to reduce memory usage."""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        df[col] = df[col].astype("float32")
    return df


# ============================================================
# CORE FUNCTIONS
# ============================================================

def load_and_merge_csvs_from_folder(folder_path: str | Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Load and merge all CSV files from a specified folder.

    Args:
        folder_path (str | Path): Path to the folder containing CSV files.

    Returns:
        tuple[pd.DataFrame, list[dict[str, Any]]]:
            - Merged DataFrame of all loaded CSVs.
            - List of skipped files with reasons.
    """
    folder = _resolve_folder_path(folder_path)

    csv_files = sorted(f for f in os.listdir(folder) if f.endswith(".csv"))
    if not csv_files:
        log.warning(f"⚠️  No CSV files found in {folder}")
        return pd.DataFrame(), []

    log.info(f"📁 Found {len(csv_files)} CSV files in {folder}")

    dataframes: list[pd.DataFrame] = []
    skipped_files: list[dict[str, Any]] = []

    for csv_file in tqdm(csv_files, desc="📄 Loading CSV files", unit="file"):
        file_path = folder / csv_file

        try:
            if file_path.stat().st_size == 0:
                skipped_files.append({"file": csv_file, "reason": "Empty file (0 bytes)"})
                continue

            df = pd.read_csv(file_path)
            if df.empty:
                skipped_files.append({"file": csv_file, "reason": "Empty dataframe (0 rows)"})
                continue

            df = _convert_numeric_to_float32(df)
            df["source_file"] = csv_file
            dataframes.append(df)

        except Exception as e:
            skipped_files.append({"file": csv_file, "reason": str(e)})
            log.error(f"❌ Failed to load {csv_file}: {e}", also_print=True)

    if not dataframes:
        log.warning("⚠️  No valid CSV files could be loaded.")
        return pd.DataFrame(), skipped_files

    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df.sort_values(by="timestamp", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    log.info(f"✅ Merged {len(dataframes)} files, total rows: {len(merged_df)}")
    return merged_df, skipped_files


def print_skipped_files_summary(skipped_files: list[dict[str, Any]]) -> None:
    """Print a grouped summary of skipped files and reasons."""
    if not skipped_files:
        log.info("✅ All files processed successfully!")
        return

    log.warning(f"\n⚠️ Skipped Files Summary ({len(skipped_files)} files):")
    log.info("=" * 60)

    grouped: dict[str, list[str]] = {}
    for item in skipped_files:
        grouped.setdefault(item["reason"], []).append(item["file"])

    for reason, files in grouped.items():
        log.warning(f"\n📋 {reason}:")
        for f in files:
            log.warning(f"   • {f}")

    log.info("=" * 60)


# ============================================================
# SYMBOL-LEVEL PROCESSING
# ============================================================

def process_single_symbol(symbol: str, folder_path: str | Path, output_directory: str | Path, suffix: str = "") -> str | None:
    """
    Process a single symbol (e.g., EURUSD or GOLD) and save the merged result.

    Args:
        symbol: Asset symbol.
        folder_path: Folder containing CSV tick data.
        output_directory: Target directory for output files.
        suffix: Optional suffix for file naming (e.g., "-pair", "-instrument").

    Returns:
        str | None: Path of the saved Parquet file, or None if failed.
    """
    try:
        folder = _resolve_folder_path(folder_path)
        output_dir = resolve_path(output_directory)  # ✅ usa resolve_path
        output_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"\n🔄 Processing symbol: {symbol}")
        log.info(f"📂 Folder path: {folder}")

        merged_df, skipped_files = load_and_merge_csvs_from_folder(folder)
        if merged_df.empty:
            log.error(f"❌ No valid data for {symbol}")
            return None

        memory_mb = merged_df.memory_usage(deep=True).sum() / 1024**2
        log.info(f"📊 Shape: {merged_df.shape} | 💾 {memory_mb:.2f} MB")

        print_skipped_files_summary(skipped_files)

        output_path = output_dir / f"{symbol}{suffix}_ticks.parquet"
        merged_df.to_parquet(output_path, engine="pyarrow", compression="snappy")

        log.info(f"✅ Saved {len(merged_df)} rows → {output_path}")
        return str(output_path)

    except Exception as e:
        log.error(f"❌ Error processing symbol {symbol}: {e}", also_print=True)
        raise


# ============================================================
# MERGE CONTROLLER
# ============================================================

def run_merge(config: dict[str, Any]) -> None:
    """
    Run the merge stage according to pipeline configuration.

    Expected config structure:
    {
        "pairs": [{"symbol": "EURUSD", "folder_path": "data/raw/eurusd/"}],
        "instruments": [{"symbol": "GOLD", "folder_path": "data/raw/gold/"}],
        "output_directory": "data/processed/"
    }
    """
    cd_repo_root()  # ✅ reemplaza todo el bloque con os.chdir y repo_root
    log.info("📂 Working directory set to repo root")

    try:
        output_dir = resolve_path(config["output_directory"])  # ✅ resuelve de forma segura
        output_dir.mkdir(parents=True, exist_ok=True)

        pairs = config.get("pairs", [])
        instruments = config.get("instruments", [])

        log.info(f"🚀 Starting merge process → output: {output_dir}")
        log.info(f"💱 Pairs: {len(pairs)} | 📈 Instruments: {len(instruments)}")

        total_processed = 0

        # --- Process currency pairs ---
        for p in pairs:
            result = process_single_symbol(p["symbol"], p["folder_path"], output_dir, suffix="-pair")
            if result:
                total_processed += 1

        # --- Process instruments (e.g., GOLD, OIL) ---
        for i in instruments:
            result = process_single_symbol(i["symbol"], i["folder_path"], output_dir, suffix="-instrument")
            if result:
                total_processed += 1

        log.info("\n🎯 Merge Process Summary:")
        log.info("=" * 50)
        log.info(f"✅ Total processed symbols: {total_processed}")
        log.info(f"📁 Output directory: {output_dir}")
        log.info("=" * 50)

    except Exception as e:
        log.critical(f"❌ CRITICAL ERROR in merge stage: {e}", also_print=True)
        raise
