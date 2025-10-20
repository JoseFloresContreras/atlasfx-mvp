#!/bin/bash
# Development Environment Setup Script for Atlas FX MVP

set -e  # Exit on error

echo "🚀 Setting up Atlas FX MVP development environment..."
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Check if Python 3.8 or higher
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "   ✅ Python version is compatible"
else
    echo "   ❌ Python 3.8+ required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
    echo "   ✅ Virtual environment created"
else
    echo ""
    echo "ℹ️  Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q

# Install data-pipeline requirements
if [ -f "data-pipeline/requirements.txt" ]; then
    echo ""
    echo "📦 Installing data-pipeline dependencies..."
    pip install -r data-pipeline/requirements.txt -q
    echo "   ✅ Data-pipeline dependencies installed"
fi

# Install agent requirements
if [ -f "agent/TD3/requirements.txt" ]; then
    echo ""
    echo "📦 Installing agent dependencies..."
    pip install -r agent/TD3/requirements.txt -q
    echo "   ✅ Agent dependencies installed"
fi

# Create necessary directories
echo ""
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p data
echo "   ✅ Directories created"

# Check if data files exist
echo ""
echo "📊 Checking for data files..."
if [ -d "data/raw-tick-data" ]; then
    echo "   ✅ Raw data directory found"
else
    echo "   ℹ️  Raw data directory not found. Please add your data to data/raw-tick-data/"
fi

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Activate the virtual environment: source .venv/bin/activate"
echo "   2. Add your data to: data/raw-tick-data/"
echo "   3. Configure pipeline: data-pipeline/pipeline.yaml"
echo "   4. Run pipeline: cd data-pipeline && python pipeline.py"
echo "   5. Train agent: cd agent/TD3 && python main.py"
echo ""
echo "📚 Documentation:"
echo "   - README.md: Project overview and usage"
echo "   - CONTRIBUTING.md: Development guidelines"
echo "   - CODE_QUALITY.md: Code quality standards"
echo ""
