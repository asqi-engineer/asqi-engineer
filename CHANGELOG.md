# Changelog
## [0.5.0] - 2026-04-28

### 🚀 Features

- Enable support for rich schema definition for label map (f58a91e)
- Add docs for labelmaps (f58a91e)
- Add Metrics Registry for metric discovery across LLM, RAG, VLM, Object Detection, and Image Editing SUT types (e594009)
- Add `list_metrics_by_data` to metric registries for finding compatible metrics per dataset/test case type (e594009)
- Add `load_test_cases()` for schema-safe typed dataset loading with Pydantic TestCase validation (0ceb6d4)

### 🐛 Bug Fixes

- Fixed other linting (fcc9e1b)
- Ruff linting (f58a91e)
- Update labelmap schema to be more specific (f58a91e)
- Update the schema definitions (f58a91e)
- Security issue by upgrading pygments to >=2.20.0 (f58a91e)
- Fix lint and test issues (190001c)
- Update import paths to use `asqi_metrics.schemas` across all metrics modules (e594009)

### 📚 Documentation

- Add `datasets.md` with full API reference, usage examples, and schema selection guide for `load_test_cases()` (0ceb6d4)
- Enhance metrics README with detailed usage examples for direct import and dynamic metric discovery (e594009)

### 🧪 Testing

- Test multiple test change scenarios (d430b84)
- Add comprehensive test suite for `load_test_cases()` covering JSONL, JSON, CSV, and Parquet formats with LLM and RAG schemas (0ceb6d4)

### ◀️ Revert

- Revert smoke tests (fd29538)
- Revert tag of 4.10 in uv.lock (190001c)
## [0.4.9] - 2026-04-06

### 🚀 Features

- Enable support for rich schema definition for label map (dd7d2e7)
- Add docs for labelmaps (dd7d2e7)

### 🐛 Bug Fixes

- Ruff linting (dd7d2e7)
- Update labelmap schema to be more specific (dd7d2e7)
- Update the schema definitions (dd7d2e7)
- Security issue by upgrading pygments to >=2.20.0 (dd7d2e7)
