# Quick Start Guide

Get up and running with Atlas FX MVP in minutes.

## Prerequisites

- Python 3.8 or higher
- Git
- 4GB+ RAM recommended
- Raw forex tick data (CSV format)

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository (update URL to your actual repository)
git clone https://github.com/JoseFloresContreras/atlasfx-mvp.git
cd atlasfx-mvp

# Run setup script
./setup_dev.sh
```

### Option 2: Manual Setup

```bash
# Clone the repository (update URL to your actual repository)
git clone https://github.com/JoseFloresContreras/atlasfx-mvp.git
cd atlasfx-mvp

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r data-pipeline/requirements.txt
pip install -r agent/TD3/requirements.txt
```

## Data Preparation

1. Place your raw tick data in the appropriate directory:
   ```
   data/
   └── raw-tick-data/
       ├── raw data/
       │   ├── audusd/
       │   │   └── *.csv
       │   ├── eurusd/
       │   │   └── *.csv
       │   └── ...
       └── raw instruments/
           ├── xauusd/
           │   └── *.csv
           └── ...
   ```

2. Expected CSV format:
   - Columns: `timestamp`, `askPrice`, `bidPrice`, `volume` (optional)
   - Timestamp: Unix milliseconds

## Running the Pipeline

### 1. Configure Pipeline

Edit `data-pipeline/pipeline.yaml`:

```yaml
# Choose which steps to run
steps: [merge, clean_ticks, aggregate, split, winsorize, 
        clean_aggregated, featurize, clean_featurized, normalize, visualize]

# Set output directory
output_directory: data

# Configure aggregation
aggregate:
  time_window: 5min  # Options: 30S, 1min, 5min, 15min, 30min, 1h, 4h, 1D, 1W
  output_filename: forex_data
```

### 2. Run Pipeline

```bash
cd data-pipeline
python pipeline.py
```

Expected output:
```
🎯 DYNAMIC FOREX DATA PROCESSING PIPELINE
============================================================
📋 Loading pipeline configuration from pipeline.yaml...
📋 Steps to execute: merge, clean_ticks, aggregate, ...
```

### 3. Monitor Progress

- **Console**: Real-time progress updates
- **Logs**: Detailed logs in `logs/log-YYYYMMDD_HHMMSS.log`
- **Output**: Processed data in `data/` directory

## Training the Agent

### 1. Verify Data

Ensure pipeline produced (example with 1H time window):
```
data/
├── 1H_forex_data_train.parquet
├── 1H_forex_data_val.parquet
└── 1H_forex_data_test.parquet
```

Note: Update the time window in `pipeline.yaml` if you want different granularity (5min, 15min, etc.)

### 2. Configure Agent

Edit `agent/TD3/main.py` if needed:
```python
env = ForexTradingEnv(
    data_path="../../data-pipeline/data/1H_forex_data_train.parquet",
    episode_length=100,
    initial_balance=1,  # Normalized balance for RL training
    transaction_fee=0.0001,
    max_position_size=0.1,
    add_noise=False
)
```

### 3. Train Agent

```bash
cd agent/TD3
python main.py --policy TD3 --seed 0
```

Optional arguments:
```bash
python main.py \
    --policy TD3 \
    --seed 0 \
    --start_timesteps 25000 \
    --eval_freq 5000 \
    --max_timesteps 1000000 \
    --save_model
```

### 4. Monitor Training

Training output:
```
---------------------------------------
Policy: TD3, Env: ForexTradingEnv, Seed: 0
---------------------------------------
Total T: 5000 Episode Num: 10 Episode T: 500 Reward: 123.45
...
```

Results saved in:
- `agent/TD3/results/` - Training metrics
- `agent/TD3/models/` - Saved models (if --save_model)

## Common Issues

### Issue: Module Not Found

**Error**: `ModuleNotFoundError: No module named 'pandas'`

**Solution**: 
```bash
source .venv/bin/activate  # Activate virtual environment
pip install -r data-pipeline/requirements.txt
```

### Issue: File Not Found

**Error**: `FileNotFoundError: 'data/raw-tick-data/...'`

**Solution**: 
1. Check data directory structure
2. Update paths in `pipeline.yaml`
3. Ensure CSV files exist

### Issue: Out of Memory

**Error**: `MemoryError` or system slowdown

**Solution**:
1. Process fewer pairs/instruments
2. Use smaller time windows
3. Reduce episode length in agent
4. Increase system swap space

### Issue: GPU Not Found

**Message**: `Using CPU for training`

**Solution**: 
- PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Training works on CPU (slower)

## Validation

### Verify Pipeline Output

```bash
cd data-pipeline
python -c "
import pandas as pd
df = pd.read_parquet('data/1H_forex_data_train.parquet')
print(f'Shape: {df.shape}')
print(f'Columns: {len(df.columns)}')
print(f'Features: {len([c for c in df.columns if \"[Feature]\" in c])}')
"
```

Expected output:
```
Shape: (50000, 150)
Columns: 150
Features: 120
```

### Test Environment

```bash
cd agent/TD3
python -c "
from env import ForexTradingEnv
env = ForexTradingEnv(data_path='../../data-pipeline/data/1H_forex_data_train.parquet')
obs = env.reset()
print(f'Observation shape: {obs.shape}')
print('✅ Environment working')
"
```

## Next Steps

1. **Experiment with hyperparameters**: Modify `pipeline.yaml` and agent parameters
2. **Add custom features**: Create new featurizers in `data-pipeline/featurizers.py`
3. **Try different policies**: Test DDPG or OurDDPG instead of TD3
4. **Analyze results**: Use visualization outputs in `data/visualizations/`
5. **Read documentation**: See `README.md`, `CONTRIBUTING.md`, `CODE_QUALITY.md`

## Getting Help

- Check existing documentation
- Review code comments
- Open an issue on GitHub
- Check logs in `logs/` directory

## Performance Tips

### Pipeline Optimization
- Use `parquet` format (already default)
- Process data incrementally if memory limited
- Use appropriate time windows (larger = less data)

### Training Optimization
- Use GPU if available
- Adjust batch size based on memory
- Save models periodically with `--save_model`
- Use multiple seeds for robust results

---

**Ready to dive deeper?** Check out the [README.md](README.md) for comprehensive documentation.
