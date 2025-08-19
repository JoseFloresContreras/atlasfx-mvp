import os
import pandas as pd
import yaml
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
from logger import log

def load_and_merge_csvs_from_folder(folder_path: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Load and merge all CSV files from a specified folder.
    
    Args:
        folder_path (str): Path to the folder containing CSV files
        
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: Merged dataframe and list of skipped files info
    """
    if not os.path.exists(folder_path):
        error_msg = f"Folder '{folder_path}' does not exist"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise FileNotFoundError(error_msg)
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
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
                skipped_files.append({
                    'file': csv_file,
                    'reason': 'Empty file (0 bytes)'
                })
                continue
            
            # Read CSV file with float32 data types for numeric columns
            df = pd.read_csv(file_path)
            
            # Convert numeric columns to float32 to reduce memory usage
            numeric_columns = df.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                df[col] = df[col].astype('float32')
            
            # Check if dataframe is empty
            if df.empty:
                skipped_files.append({
                    'file': csv_file,
                    'reason': 'Empty dataframe (0 rows)'
                })
                continue
            
            # Add source file column
            df['source_file'] = csv_file
            
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

def print_skipped_files_summary(skipped_files: List[Dict[str, Any]]):
    """
    Print a summary of skipped files.
    
    Args:
        skipped_files (List[Dict]): List of dictionaries containing skipped file info
    """
    if not skipped_files:
        log.info("✅ All files processed successfully!")
        return
    
    log.warning(f"\n⚠️  Skipped Files Summary ({len(skipped_files)} files):")
    log.info("=" * 60)
    
    # Group by reason
    reasons = {}
    for file_info in skipped_files:
        reason = file_info['reason']
        if reason not in reasons:
            reasons[reason] = []
        reasons[reason].append(file_info['file'])
    
    # Print grouped summary
    for reason, files in reasons.items():
        log.warning(f"\n📋 {reason}:")
        for file in files:
            log.warning(f"   • {file}")
    
    log.info("=" * 60)

def process_single_symbol(symbol: str, folder_path: str, output_directory: str) -> Tuple[bool, str]:
    """
    Process a single symbol and save the merged result.
    
    Args:
        symbol (str): Symbol name
        folder_path (str): Path to the folder containing CSV files
        output_directory (str): Directory to save the output file
        
    Returns:
        Tuple[bool, str]: (True if successful, output file path) or (False, None)
    """
    try:
        log.info(f"\n🔄 Processing symbol: {symbol}")
        log.info(f"📁 Loading CSV files from: {folder_path}")
        
        # Load and merge CSV files
        merged_df, skipped_files = load_and_merge_csvs_from_folder(folder_path)
        
        if merged_df.empty:
            log.error(f"❌ No data was loaded for {symbol}")
            return False, None
        
        # Display data summary
        log.info(f"📊 Merged dataframe shape: {merged_df.shape}")
        log.info(f"📋 Columns: {list(merged_df.columns)}")
        memory_mb = merged_df.memory_usage(deep=True).sum() / 1024**2
        log.info(f"💾 Memory usage: {memory_mb:.2f} MB (with float32 optimization)")
        
        # Print skipped files summary for this symbol
        print_skipped_files_summary(skipped_files)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Generate output filename
        output_filename = f"{symbol}_ticks.parquet"
        output_path = os.path.join(output_directory, output_filename)
        
        # Save merged data with float32 precision
        log.info(f"💾 Saving to: {output_path}")
        merged_df.to_parquet(output_path, engine='pyarrow', compression='snappy')
        
        log.info(f"✅ Successfully saved {len(merged_df)} rows to {output_path}")
        return output_path
        
    except Exception as e:
        log.error(f"❌ Error merging {symbol}: {e}")
        raise e

def run_merge(config):
    """
    Run the merge process with the specified configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing symbols and output directory settings
    """
    try:
        log.info("🚀 Starting CSV Merge Process")
        log.info("=" * 50)
        
        # Extract configuration values
        symbols_config = config['symbols']
        output_directory = config['output_directory']
        
        log.info(f"📁 Output directory: {output_directory}")
        log.info(f"🔤 Symbols to process: {len(symbols_config)}")
        
        # Process each symbol sequentially
        total_symbols = len(symbols_config)
        
        for symbol_config in symbols_config:
            symbol = symbol_config['symbol']
            folder_path = symbol_config['folder_path']
            
            output_path = process_single_symbol(symbol, folder_path, output_directory)
        
        # Display final summary
        log.info(f"\n🎯 Merge Process Summary:")
        log.info("=" * 50)
        log.info(f"   📊 Total symbols: {total_symbols}")
        log.info(f"   ✅ Successful: {total_symbols}")
        log.info(f"   📁 Output directory: {output_directory}")
        log.info("\n🎉 All symbols processed successfully!")
        
        log.info("=" * 50)
        
    except Exception as e:
        error_msg = f"Fatal Error: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise 