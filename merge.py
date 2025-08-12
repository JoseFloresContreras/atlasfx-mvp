import os
import pandas as pd
import yaml
from tqdm import tqdm
from typing import List, Dict, Tuple, Any

def load_and_merge_csvs_from_folder(folder_path: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Load and merge all CSV files from a specified folder.
    
    Args:
        folder_path (str): Path to the folder containing CSV files
        
    Returns:
        Tuple[pd.DataFrame, List[Dict]]: Merged dataframe and list of skipped files info
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
    
    # Get all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"⚠️  No CSV files found in {folder_path}")
        return pd.DataFrame(), []
    
    print(f"📁 Found {len(csv_files)} CSV files in {folder_path}")
    
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
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
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
            skipped_files.append({
                'file': csv_file,
                'reason': f'Read error: {str(e)}'
            })
            continue
    
    if not dataframes:
        print("⚠️  No valid CSV files could be loaded")
        return pd.DataFrame(), skipped_files
    
    # Merge all dataframes
    print("🔗 Merging dataframes...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    return merged_df, skipped_files

def print_skipped_files_summary(skipped_files: List[Dict[str, Any]]):
    """
    Print a summary of skipped files.
    
    Args:
        skipped_files (List[Dict]): List of dictionaries containing skipped file info
    """
    if not skipped_files:
        print("✅ All files processed successfully!")
        return
    
    print(f"\n⚠️  Skipped Files Summary ({len(skipped_files)} files):")
    print("=" * 60)
    
    # Group by reason
    reasons = {}
    for file_info in skipped_files:
        reason = file_info['reason']
        if reason not in reasons:
            reasons[reason] = []
        reasons[reason].append(file_info['file'])
    
    # Print grouped summary
    for reason, files in reasons.items():
        print(f"\n📋 {reason}:")
        for file in files:
            print(f"   • {file}")
    
    print("=" * 60)

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
        print(f"\n🔄 Processing symbol: {symbol}")
        print(f"📁 Loading CSV files from: {folder_path}")
        
        # Load and merge CSV files
        merged_df, skipped_files = load_and_merge_csvs_from_folder(folder_path)
        
        if merged_df.empty:
            print(f"❌ No data was loaded for {symbol}")
            return False, None
        
        # Display data summary
        print(f"📊 Merged dataframe shape: {merged_df.shape}")
        print(f"📋 Columns: {list(merged_df.columns)}")
        memory_mb = merged_df.memory_usage(deep=True).sum() / 1024**2
        print(f"💾 Memory usage: {memory_mb:.2f} MB")
        
        # Print skipped files summary for this symbol
        print_skipped_files_summary(skipped_files)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Generate output filename
        output_filename = f"{symbol}_ticks.parquet"
        output_path = os.path.join(output_directory, output_filename)
        
        # Save merged data
        print(f"💾 Saving to: {output_path}")
        merged_df.to_parquet(output_path, engine='pyarrow')
        
        print(f"✅ Successfully saved {len(merged_df)} rows to {output_path}")
        return True, output_path
        
    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")
        return False, None

def run_merge(config):
    """
    Run the merge process with the specified configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing symbols and output directory settings
    """
    try:
        print("🚀 Starting CSV Merge Process")
        print("=" * 50)
        
        # Extract configuration values
        symbols_config = config['symbols']
        output_directory = config['output_directory']
        
        print(f"📁 Output directory: {output_directory}")
        print(f"🔤 Symbols to process: {len(symbols_config)}")
        
        # Process each symbol sequentially
        successful_symbols = 0
        total_symbols = len(symbols_config)
        
        for symbol_config in symbols_config:
            symbol = symbol_config['symbol']
            folder_path = symbol_config['folder_path']
            
            success, output_path = process_single_symbol(symbol, folder_path, output_directory)
            if success:
                successful_symbols += 1
        
        # Display final summary
        print(f"\n🎯 Merge Process Summary:")
        print("=" * 50)
        print(f"   📊 Total symbols: {total_symbols}")
        print(f"   ✅ Successful: {successful_symbols}")
        print(f"   ❌ Failed: {total_symbols - successful_symbols}")
        print(f"   📁 Output directory: {output_directory}")
        
        if successful_symbols == total_symbols:
            print("\n🎉 All symbols processed successfully!")
        else:
            print(f"\n⚠️  {total_symbols - successful_symbols} symbols failed to process")
        
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Fatal Error: {e}")
        raise 