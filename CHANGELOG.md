# Changelog
## [0.5.2] - 2026-05-14

### 🐛 Bug Fixes
- Fix container output parsing failures when a single JSON line exceeds Docker's stream chunk size (16KB) (6d55977)

### 🧪 Testing
- Add edge case tests for chunked log parser (6d55977)

## [0.5.1] - 2026-05-05

### 🚀 Features
- Add logic for building extra_body.metadata in `get_openai_tracking_kwargs()` (6edbe7d)

### 🐛 Bug Fixes
- Update changelog's features and documentation lists for metrics registry and imports (d8af922)
- Refactor code for improved readability and maintainability by applying consistent formatting across multiple files in `src/` (e2ed70f)
- Add security annotation to dataset loading to address linting issue (8170e64)
- Update greenlet package URLs and bump pip version to 26.1 (fb2004b)
- Update sync-to-private-dispatch.yml (6a5b97f)
- Fix type checking in `get_openai_tracking_kwargs()` (6edbe7d)

### 🧪 Testing
- Add tests for `src/utils.py` (eabd6ba)

## [0.5.0] - 2026-04-28

### 🚀 Features

- Enable support for rich schema definition for label map (f58a91e)
- Add docs for labelmaps (f58a91e)
- Add `load_test_cases()` for schema-safe typed dataset loading with Pydantic TestCase validation (0ceb6d4)

### 🐛 Bug Fixes

- Fixed other linting (fcc9e1b)
- Ruff linting (f58a91e)
- Update labelmap schema to be more specific (f58a91e)
- Update the schema definitions (f58a91e)
- Security issue by upgrading pygments to >=2.20.0 (f58a91e)
- Fix lint and test issues (190001c)

### 📚 Documentation

- Add `datasets.md` with full API reference, usage examples, and schema selection guide for `load_test_cases()` (0ceb6d4)

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
