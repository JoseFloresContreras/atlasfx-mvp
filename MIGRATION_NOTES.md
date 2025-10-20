# Migration Notes

## Repository Restructure - Phase 1 Complete

### What was done:

1. ✅ Created new `src/atlasfx/` package structure following PEP 517/518 standards
2. ✅ Migrated all data-pipeline modules to `src/atlasfx/data/`
3. ✅ Created `scripts/` directory with `run_data_pipeline.py`
4. ✅ Created `configs/` directory with `data_pipeline.yaml`
5. ✅ Created placeholder directories: `notebooks/`, `experiments/`, `data/`, `models/`
6. ✅ Updated all imports to use new module structure
7. ✅ Reorganized tests to mirror `src/` structure
8. ✅ Updated `pyproject.toml` for src layout
9. ✅ Created `.github/workflows/ci.yml`
10. ✅ Added README files to key directories

### File/Folder Naming Conventions Applied:

- ✅ Python modules: snake_case (e.g., `data_pipeline.py`, `visualizers.py`)
- ✅ Package names: lowercase (e.g., `atlasfx`, `data`, `models`)
- ✅ Config files: snake_case with extension (e.g., `data_pipeline.yaml`)
- ✅ Documentation: UPPERCASE for main docs (e.g., `README.md`, `ARCHITECTURE.md`)
- ✅ Scripts: descriptive snake_case with action prefix (e.g., `run_data_pipeline.py`)

### Old data-pipeline directory

⚠️ The old `data-pipeline/` directory is still present for reference but is **deprecated**.
All functionality has been migrated to `src/atlasfx/data/`.

**Action required**: Once the migration is verified and all dependencies are installed,
the `data-pipeline/` directory can be safely removed:

```bash
# After verification, remove old directory
rm -rf data-pipeline/
```

### Dependencies

The following dependencies are required but could not be installed due to network issues:
- tqdm
- pyarrow
- matplotlib
- seaborn
- scikit-learn
- pandas-ta

These should be installed before running tests:
```bash
pip install tqdm pyarrow matplotlib seaborn scikit-learn pandas-ta
```

### Next Steps

1. Install missing dependencies
2. Verify all tests pass: `pytest tests/`
3. Remove old `data-pipeline/` directory
4. Implement model modules in `src/atlasfx/models/`
5. Implement training modules in `src/atlasfx/training/`
6. Add integration tests
