#!/bin/bash
# Environment setup script for AtlasFX
# This script creates a reproducible development environment

set -e  # Exit on error

echo "🚀 Setting up AtlasFX development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.10"

echo -e "${YELLOW}📋 Checking Python version...${NC}"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo -e "${RED}❌ Error: Python 3.10+ required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python version OK: $(python3 --version)${NC}"

# Create virtual environment
echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists. Skipping creation.${NC}"
else
    python3 -m venv .venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}🔄 Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✅ pip upgraded${NC}"

# Install dependencies
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
pip install -e ".[dev]" > /dev/null 2>&1
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Install pandas-stubs and types for better type checking
echo -e "${YELLOW}📦 Installing type stubs...${NC}"
pip install pandas-stubs types-PyYAML > /dev/null 2>&1
echo -e "${GREEN}✅ Type stubs installed${NC}"

# Install pre-commit hooks
echo -e "${YELLOW}🪝 Installing pre-commit hooks...${NC}"
pre-commit install > /dev/null 2>&1
echo -e "${GREEN}✅ Pre-commit hooks installed${NC}"

# Create necessary directories
echo -e "${YELLOW}📁 Creating project directories...${NC}"
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/features
mkdir -p experiments
mkdir -p models
mkdir -p logs
mkdir -p htmlcov
echo -e "${GREEN}✅ Directories created${NC}"

# Run initial tests to verify setup
echo -e "${YELLOW}🧪 Running initial tests...${NC}"
if pytest tests/ -v --tb=short > /tmp/test_output.log 2>&1; then
    echo -e "${GREEN}✅ All tests passing${NC}"
else
    echo -e "${RED}⚠️  Some tests failed. Check /tmp/test_output.log for details${NC}"
fi

# Check code formatting
echo -e "${YELLOW}🎨 Checking code formatting...${NC}"
if black --check src/ tests/ scripts/ > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Code formatting OK${NC}"
else
    echo -e "${YELLOW}⚠️  Code needs formatting. Run: black src/ tests/ scripts/${NC}"
fi

# Summary
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}✨ Setup complete! AtlasFX is ready to go! ✨${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo ""
echo "To activate the environment in the future, run:"
echo -e "  ${YELLOW}source .venv/bin/activate${NC}"
echo ""
echo "Useful commands:"
echo -e "  ${YELLOW}pytest tests/ -v${NC}              # Run tests"
echo -e "  ${YELLOW}black src/ tests/ scripts/${NC}   # Format code"
echo -e "  ${YELLOW}ruff check src/${NC}               # Lint code"
echo -e "  ${YELLOW}mypy src/atlasfx${NC}              # Type check"
echo -e "  ${YELLOW}pre-commit run --all-files${NC}   # Run all pre-commit hooks"
echo ""
echo "Happy coding! 🎉"
