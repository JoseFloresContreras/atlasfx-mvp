print("✅ loaders.py loaded successfully")

import os
from typing import Any

import pandas as pd
from tqdm import tqdm

from atlasfx.utils.logging import log


def load_and_merge_csvs_from_folder(folder_path: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Load and merge all CSV files from a specified folder.
    """
    import os
    from pathlib import Path

    print("\n=== DEBUG PATH INFO ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Original folder_path arg: {folder_path} (type={type(folder_path)})")

    # Ensure it's a Path object
    folder_path = Path(folder_path)
    if not folder_path.is_absolute():
        folder_path = Path(__file__).resolve().parents[3] / folder_path

    print(f"Resolved absolute folder_path: {folder_path}")
    print(f"Does folder exist? {folder_path.exists()}")
    print("========================\n")

    if not folder_path.exists():
        error_msg = f"Folder '{folder_path}' does not exist"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise FileNotFoundError(error_msg)

    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        log.warning(f"⚠️  No CSV files found in {folder_path}")
        return pd.DataFrame(), []

    log.info(f"📁 Found {len(csv_files)} CSV files in {folder_path}")

    # Initialize variables
    dataframes = []
    skipped_files = []

    # Process each CSV file with progress bar
    for csv_file in tqdm(csv_files, desc="Processing CSV files", unit="file"):
        file_path = os.path.join(folder_path, csv_file)

        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                skipped_files.append({"file": csv_file, "reason": "Empty file (0 bytes)"})
                continue

            # Read CSV file with float32 data types for numeric columns
            df = pd.read_csv(file_path)

            # Convert numeric columns to float32 to reduce memory usage
            numeric_columns = df.select_dtypes(include=["number"]).columns
            for col in numeric_columns:
                df[col] = df[col].astype("float32")

            # Check if dataframe is empty
            if df.empty:
                skipped_files.append({"file": csv_file, "reason": "Empty dataframe (0 rows)"})
                continue

            dataframes.append(df)

        except Exception as e:
            log.critical(f"❌ CRITICAL ERROR: {e}")
            raise e

    if not dataframes:
        log.warning("⚠️  No valid CSV files could be loaded")
        return pd.DataFrame(), skipped_files

    # Merge all dataframes
    log.info("🔗 Merging dataframes...")
    merged_df = pd.concat(dataframes, ignore_index=True)

    return merged_df, skipped_files


def print_skipped_files_summary(skipped_files: list[dict[str, Any]]):
    """
    Print a summary of skipped files.

    Args:
        skipped_files (list[Dict]): List of dictionaries containing skipped file info
    """
    if not skipped_files:
        log.info("✅ All files processed successfully!")
        return

    log.warning(f"\n⚠️  Skipped Files Summary ({len(skipped_files)} files):")
    log.info("=" * 60)

    # Group by reason
    reasons = {}
    for file_info in skipped_files:
        reason = file_info["reason"]
        if reason not in reasons:
            reasons[reason] = []
        reasons[reason].append(file_info["file"])

    # Print grouped summary
    for reason, files in reasons.items():
        log.warning(f"\n📋 {reason}:")
        for file in files:
            log.warning(f"   • {file}")

    log.info("=" * 60)


def process_single_symbol(
    symbol: str, folder_path: str, output_directory: str, suffix: str = ""
) -> str:
    """
    Process a single symbol and save the merged result.

    Args:
        symbol (str): Symbol name
        folder_path (str): Path to the folder containing CSV files
        output_directory (str): Directory to save the output file
        suffix (str): Suffix to add to the output filename (e.g., '-pair', '-instrument')

    Returns:
        str: Output file path if successful, None otherwise
    """
    try:
        log.info(f"\n🔄 Processing symbol: {symbol}")
        log.info(f"📁 Loading CSV files from: {folder_path}")

        # ======================================================
        # 🧭 DEBUGGING SECTION - path resolution diagnostics
        # ======================================================
        import os
        from pathlib import Path

        print("\n=== DEBUG SYMBOL INFO ===")
        print(f"Symbol: {symbol}")
        print(f"Original folder_path arg: {folder_path} (type={type(folder_path)})")
        print(f"Current working directory (os.getcwd): {os.getcwd()}")

        REPO_ROOT = Path(__file__).resolve().parents[3]
        print(f"Repo root (calculated): {REPO_ROOT}")

        folder_path = Path(folder_path)
        if not folder_path.is_absolute():
            folder_path = REPO_ROOT / folder_path

        print(f"Resolved absolute folder_path: {folder_path}")
        print(f"Does folder exist? {folder_path.exists()}")
        print("========================\n")
        # ======================================================

        log.info(f"📂 Normalized folder path: {folder_path}")

        # Load and merge CSV files
        merged_df, skipped_files = load_and_merge_csvs_from_folder(folder_path)

        if merged_df.empty:
            log.error(f"❌ No data was loaded for {symbol}")
            return None

        # Display data summary
        log.info(f"📊 Merged dataframe shape: {merged_df.shape}")
        log.info(f"📋 Columns: {list(merged_df.columns)}")
        memory_mb = merged_df.memory_usage(deep=True).sum() / 1024**2
        log.info(f"💾 Memory usage: {memory_mb:.2f} MB (with float32 optimization)")

        # Print skipped files summary for this symbol
        print_skipped_files_summary(skipped_files)

        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Generate output filename with suffix
        output_filename = f"{symbol}{suffix}_ticks.parquet"
        output_path = os.path.join(output_directory, output_filename)

        # Save merged data with float32 precision
        log.info(f"💾 Saving to: {output_path}")
        merged_df.to_parquet(output_path, engine="pyarrow", compression="snappy")

        log.info(f"✅ Successfully saved {len(merged_df)} rows to {output_path}")
        return output_path

    except Exception as e:
        log.error(f"❌ Error merging {symbol}: {e}")
        raise e


def run_merge(config):
    """
    Run the merge process with the specified configuration.

    Args:
        config (dict[str, Any]): Configuration dictionary containing pairs, instruments and output directory settings
    """
    import os
    from pathlib import Path

    try:
        # --- Ensure relative paths resolve from repo root ---
        REPO_ROOT = (
            Path(__file__).resolve().parents[3]
        )  # → sube desde src/atlasfx/data hasta atlasfx-mvp
        os.chdir(REPO_ROOT)

        print("🔥 run_merge() has started")

        # 🧩 DEBUG BLOCK
        print("\n🔍 DEBUG INSIDE run_merge")
        print(f"📂 Current working directory: {os.getcwd()}")
        print(f"📁 Output directory (raw): {config.get('output_directory')}")
        print(f"🔧 Full config passed in:\n{config}")
        print(f"🔎 Absolute folder paths for pairs:")
        for p in config.get("pairs", []):
            f = Path(p["folder_path"]).resolve()
            print(f"   - {p['symbol']}: {f} (exists={f.exists()})")

        log.info("🚀 Starting CSV Merge Process")
        log.info("=" * 50)

        # --- Normalize output directory ---
        output_directory = Path(config["output_directory"])
        if not output_directory.is_absolute():
            output_directory = REPO_ROOT / output_directory
        output_directory.mkdir(parents=True, exist_ok=True)

        # --- Extract pair/instrument configs ---
        pairs_config = config.get("pairs", [])
        instruments_config = config.get("instruments", [])

        log.info(f"📁 Output directory: {output_directory}")
        log.info(f"🔤 Pairs to process: {len(pairs_config)}")
        log.info(f"📈 Instruments to process: {len(instruments_config)}")

        total_processed = 0

        # --- Process pairs ---
        if pairs_config:
            log.info(f"\n💱 Processing {len(pairs_config)} pairs...")
            for symbol_config in pairs_config:
                symbol = symbol_config["symbol"]
                folder_path = Path(symbol_config["folder_path"])
                if not folder_path.is_absolute():
                    folder_path = REPO_ROOT / folder_path

                output_path = process_single_symbol(
                    symbol, folder_path, output_directory, suffix="-pair"
                )
                if output_path:
                    total_processed += 1

        # --- Process instruments ---
        if instruments_config:
            log.info(f"\n📊 Processing {len(instruments_config)} instruments...")
            for symbol_config in instruments_config:
                symbol = symbol_config["symbol"]
                folder_path = Path(symbol_config["folder_path"])
                if not folder_path.is_absolute():
                    folder_path = REPO_ROOT / folder_path

                output_path = process_single_symbol(
                    symbol, folder_path, output_directory, suffix="-instrument"
                )
                if output_path:
                    total_processed += 1

        # --- Summary ---
        log.info("\n🎯 Merge Process Summary:")
        log.info("=" * 50)
        log.info(f"   💱 Total pairs: {len(pairs_config)}")
        log.info(f"   📈 Total instruments: {len(instruments_config)}")
        log.info(f"   ✅ Successfully processed: {total_processed}")
        log.info(f"   📁 Output directory: {output_directory}")
        log.info("\n🎉 All symbols processed successfully!")
        log.info("=" * 50)

    except Exception as e:
        error_msg = f"Fatal Error: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise
