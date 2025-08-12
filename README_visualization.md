# Data Visualization and Train/Validation/Test Split Module

This module adds comprehensive data visualization and train/validation/test splitting capabilities to the AtlasFX data processing pipeline.

## Features

### 📊 Data Visualization
- **Basic Statistics**: Comprehensive data overview including row count, column count, memory usage, missing values, and data types
- **Time Series Plots**: Visualize key metrics over time (open, high, low, close, volume, spread)
- **Correlation Heatmap**: Analyze relationships between numeric features
- **Distribution Plots**: Histograms for all numeric columns to understand data distributions
- **Box Plots**: Identify outliers and understand data spread for numeric features

### ✂️ Data Splitting
- **Train/Validation/Test Split**: Automatically split data into three sets
- **Configurable Ratios**: Customize split proportions via configuration
- **Time-Series Aware**: Maintains temporal order during splitting
- **Split Summary**: Detailed statistics for each split

## Configuration

Add the following section to your `pipeline.yaml`:

```yaml
# Step 7: Visualization and Data Split Configuration
visualize:
  split:
    train: 0.7    # 70% for training
    val: 0.15     # 15% for validation
    test: 0.15    # 15% for testing
```

## Pipeline Integration

The visualization step is automatically integrated into the pipeline and can be included in the steps list:

```yaml
steps: [merge, clean_ticks, aggregate, clean_aggregated, augment, clean_augmented, visualize]
```

## Output Files

### Visualizations
All visualization files are saved to `data/visualizations/`:
- `basic_statistics.txt` - Comprehensive data statistics
- `time_series_plots.png` - Time series visualization of key metrics
- `correlation_heatmap.png` - Feature correlation analysis
- `distribution_plots.png` - Histograms of numeric features
- `box_plots.png` - Box plots showing data distribution and outliers
- `split_summary.txt` - Statistics for each data split

### Data Splits
Split files are saved to `data/splits/`:
- `train.parquet` - Training dataset
- `validation.parquet` - Validation dataset
- `test.parquet` - Test dataset

## Dependencies

The following additional dependencies are required:
- `matplotlib>=3.5.0` - For creating plots
- `seaborn>=0.11.0` - For enhanced visualizations
- `scikit-learn>=1.0.0` - For data splitting
- `numpy>=1.21.0` - For numerical operations

## Usage

### Standalone Usage
```python
from visualize import run_visualize

config = {
    'input_file': 'data/your_data.parquet',
    'output_directory': 'data',
    'time_column': 'start_time',
    'split': {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15
    }
}

run_visualize(config)
```

### Pipeline Integration
Simply add `visualize` to your pipeline steps and the module will automatically:
1. Load the final processed data
2. Generate comprehensive visualizations
3. Split the data into train/validation/test sets
4. Save all outputs to the specified directories

## Key Features

- **Automatic Column Limiting**: Handles large datasets by limiting visualization to first 20-30 columns to prevent memory issues
- **Missing Value Handling**: Automatically handles missing values in visualizations
- **Time Series Preservation**: Maintains temporal order during data splitting
- **Comprehensive Reporting**: Detailed statistics and summaries for all operations
- **Error Handling**: Robust error handling with informative messages

## Example Output

After running the visualization step, you'll have:
- 105,120 rows of data split into:
  - 73,583 training samples (70%)
  - 15,769 validation samples (15%)
  - 15,768 test samples (15%)
- 235 features with comprehensive visualizations
- Complete data quality reports and statistics 