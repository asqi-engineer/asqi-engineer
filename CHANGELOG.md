# Changelog
## [0.4.9] - 2026-04-06

### 🚀 Features

- Enable support for rich schema definition for label map (dd7d2e7)
- Add docs for labelmaps (dd7d2e7)

### 🐛 Bug Fixes

- Ruff linting (dd7d2e7)
- Update labelmap schema to be more specific (dd7d2e7)
- Update the schema definitions (dd7d2e7)
- Security issue by upgrading pygments to >=2.20.0 (dd7d2e7)
## [0.4.8] - 2026-03-18

### 🚀 Features

- Introduce embedding_api to asqi engineer (d9103a9)
- Add ui_config field for optional UI configuration hints as arbitrary key–value pairs (d9103a9)

### 🐛 Bug Fixes

- Pass parent_id to execute_single_test and execute_data_generation (d9103a9)
- Formatting (d9103a9)
- Typo in docs (d9103a9)
- Formatting (d9103a9)
## [0.4.7] - 2026-03-11

### 🚀 Features

- Add thinking and reasoning params to llm api params (aa61001)
- Introduce embedding_api to asqi engineer (1d64689)

### 🐛 Bug Fixes

- Send workflow dispatch event instead of repository dispatch event (97aa2e7)
- Reference the correct workflow to trigger with workflow dispatch (34ee25b)
- Pass parent_id to execute_single_test and execute_data_generation (1d64689)
- Formatting (1d64689)
- Typo in docs (1d64689)
- Raise error on duplicate manifest keys instead of silent overwrite (#339) (da2795e)
- Raise error on duplicate manifest keys instead of silent overwrite (da2795e)

### 📚 Documentation

- Update docs for thinking and reasoning model examples (6fd2ffd)

### 🧪 Testing

- Workflow dispatch for syncing (d513869)
- Overwrite private content (2905fb1)
- Try to overwrite private content via sync (a6a8e25)
- Try to overwrite private content via sync (a6a8e25)
- Private content should not be overwritten (2b30ae3)
- Private content should not be overwritten (2b30ae3)
## [0.4.6] - 2026-02-09

### 🚀 Features

- Add HuggingFace vision demo (18e744d)
- Add Hub dataset support to load_hf_dataset and refactor hf_vision_tester to use streaming (18e744d)
- Add Hub dataset support to load_hf_dataset with streaming, refactor hf_vision_tester (18e744d)
- Implement asqi package preparation script and update CI workflow for container builds (18e744d)
- Add concurrency param with default 1 to harbor container (f5c8f3a)

### 🐛 Bug Fixes

- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Pass manifest to run_container_with_args (54d91db)
- Improve container ID detection in devcontainer host path (#336) (67639f4)
- Improve container ID detection in devcontainer host path (67639f4)
- Fix line length in test_container_manager.py (67639f4)
- Use /proc/self/mountinfo for container ID detection (cgroup v1+v2 compatible) (67639f4)
- Improve container ID detection in devcontainer host path (18e744d)
- Format code and suppress expected HF dataset warnings (18e744d)
- Support dedicated HF endpoints and containerize with repo-root build context (18e744d)
- Fix formatting (18e744d)
- Add missing newline at end of asqi_systems_config.schema.json (#340) (6ec0756)

### 🚜 Refactor

- Remove bind mount logic for _configure_job_dirs (54d91db)
- Use logging instead of print (54d91db)
- Merge hf_detr system config into image_systems.yaml (18e744d)
- Revert back to original devcontainer_host_path (18e744d)
- Revert back to original devcontainer_host_path (18e744d)
- Revert back to original devcontainer_host_path (18e744d)
- Revert back to original devcontainer_host_path (18e744d)

### PLT-980

- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
- Harbor integration (#334) (54d91db)
## [0.4.5] - 2026-01-28

### 🚀 Features

- Add tracking function to asqi library (6f91e05)
- Use custom openai client in llmperf (#330) (5df8761)
- Use custom openai client in llmperf (5df8761)
- Update inspect to return usage metadata (57b8580)
- Add metadata tracing to garak (57b8580)

### 🐛 Bug Fixes

- Resolve overwriting job_id issue. (6f91e05)
- Fix : resolve feedbacks (6f91e05)
- Fix : modify chatbot simulator Dockerfile into 2 stage build (93ed30d)
- Pass parent_id to execute_single_test and execute_data_generation (#333) (8bb5215)

### 🚜 Refactor

- Create common util function with test case (6f91e05)
- Use pydantic schema (6f91e05)
## [0.4.4] - 2026-01-26

### 🚀 Features

- Support huggingface feature types with required field (fe515ac)
- Add video feature type support (fe515ac)
- Improve dataset field validation (fe515ac)
- Add required field to DatasetFeature (fe515ac)
- Add composite types for input params (fa7995f)
- Add validation of input param types (fa7995f)
- Improve trustllm manifest typings (fa7995f)

### 🐛 Bug Fixes

- Parse workflow error output (#317) (159789d)
- Parse workflow error output (159789d)
- Handle case where there is docker error with parsing failure (159789d)

### 🚜 Refactor

- Simplify validation (fa7995f)

### 📚 Documentation

- Add huggingface feature types to docs (fe515ac)
- Document composite types (fa7995f)
- Add validation abd recommended use of load_hf_dataset (27051c3)
- Add imports to docs (27051c3)
## [0.4.3] - 2026-01-23

### 🐛 Bug Fixes

- Fix : resolve the issue to create container so now it only creates test_id + uuid (#315) (3179b20)
## [0.4.2] - 2026-01-16

### 🚀 Features

- Support test scenarios as input datasets for chatbot simulator (#311) (d75cca1)
- Support test scenarios as input datasets for chatbot simulator (ce96b81)

### 🐛 Bug Fixes

- Fix shared-dictionary bug in test plan building for multi-system… (#314) (ed89a9e)
- Fix shared-dictionary bug in test plan building for multi-system evaluations (ed89a9e)
- Fix shared-dictionary bug in test plan building for multi-system… (#314) (ed89a9e)
## [0.4.1] - 2026-01-13
## [0.4.0] - 2026-01-12

### 🚀 Features

- Update schema to support multiple systems (a6c7af0)
- Support multiple system validation (a6c7af0)
- Allow indicator to filter by test type (a6c7af0)
- Generalize mock tester to support multiple llm types (a6c7af0)
- Update asqi engineer to accept generation specific parameters from the user (cc71857)
- Add standalone dataset abstraction (cc71857)
- Export DatasetsConfig as json schema (cc71857)
- Add support for generated datasets (caf23d8)
- Add output datasets schema and validation (caf23d8)
- Add claude workflow bot (#303) (e2ec8bb)
- Add response schema (e0a2cf8)
- Use response schema (e0a2cf8)
- Export data generation config schema (6635adc)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)
- Add multi-format dataset support for input_datasets (#308) (b5aadee)

### 🐛 Bug Fixes

- Fix typo and input output volume (cc71857)
- Revert validation change that allows empty system_name (cc71857)
- Remove non existent oltp log endpoint (cc71857)
- Adjust order which dataset config is resolved (cc71857)
- Resolving relative path on devcontainer (cc71857)
- Make system optional for data gen (cc71857)
- Improve type hint for dataset_config input (cc71857)
- Update docstrings from test to data gen (cc71857)
- Update validate_container_output logic (e0a2cf8)
- Pylance errors (9ea8f1c)
- Replace assert with proper type checks (9ea8f1c)
- Add placeholder system input (6635adc)
- Make systems-params optional for manifest (6635adc)
- Circular references (6635adc)
- Add missing pyproject.toml (#307) (e874b17)
- Validate schema if huggingface is specified (b5aadee)

### 🚜 Refactor

- Consolidate dataset schema and improve typing (cc71857)
- Specify dataset types as discriminated union (cc71857)
- Use common validation functions across job type (e0a2cf8)
- Extract helper functions across both workflows (9ea8f1c)
- Create internal function to execute container job (9ea8f1c)
- Simplify sut handling (9ea8f1c)
- Update project structure (6635adc)
- Simplify sdg example (6635adc)
- Hoist imports to the top (6635adc)
- Use asqi util functions from main branch (6635adc)
- Consolidate normalization functions (b5aadee)

### 📚 Documentation

- Add docs on multiple system type in score card and manifest (a6c7af0)
- Update readme (6635adc)
- Update dataset docs (32257e3)
- Use existing docs without change (32257e3)
- Update configuration (32257e3)
- Fix generate-dataset cmd (32257e3)
- Update readme with supported system types (32257e3)
- Document multiple dataset type support (b5aadee)

### 🧪 Testing

- Add test suite for multi-type manifest (a6c7af0)
- Add new datasets_config_path to mocks (cc71857)
- Add coverage for data generation validation (cc71857)
- Add coverage for data generation workflow and other helper functions (cc71857)
- Add regression test for relative path handling (caf23d8)
- Add tests for dataset definition (e0a2cf8)

### Hotfix

- Sut_name no longer required for container naming (cc71857)
## [0.3.5] - 2025-12-30

### 🚀 Features

- Add report fields to manifest and score card schemas (132e5de)
- Update schemas (132e5de)
- Add technical report validations and logic (132e5de)
- Add `quick_summary` report and config to mock_tester (132e5de)
- Add requierements to mock_tester (132e5de)
- Refactor based on review feedback (132e5de)
- Add `ExecutionMode` Enum instead of string literals (132e5de)
- Refactor based on review feedback (132e5de)

### 📚 Documentation

- Update documentation for technical reports (132e5de)
- Update broken link and code blocks (#286) (2fa3dc2)

### 🧪 Testing

- Add tests for technical reports (132e5de)
## [0.3.4] - 2025-12-23

### 🚀 Features

- Added fixes and tests (a20043d)
- Add complex formula support (28b5772)
- Add tests (28b5772)
- Convert bool result to int (28b5772)
- Add new system type classes (864e317)
- Create image response schema validation (864e317)
- Create mock image generation tester (864e317)
- Create mock image editing tester (864e317)
- Add mock vlm tester (864e317)
- Add sample configurations and tests for image generation, image editing, and VLM APIs (864e317)
- Extend image response validation with new test cases and mock configurations for image generation, editing, and VLM APIs (864e317)
- Add example system configs for the new system types (864e317)
- Add demo image generation and evaluation test suite configuration (864e317)
- Add OpenAI image generation and VLM configurations (1955e48)
- Create a simple test container to test system configs and litellm proxy with calls to a image generation API and VLM API (1955e48)
- Introduce image generation, image editing and VLM API system types in the docs (276fd47)
- Force supports_vision parameter to true for vlm api system types (35ce266)
- Add supports_vision parameter to asqi_systems_config schema (35ce266)

### 🐛 Bug Fixes

- Default supports_vision to true for VLMs instead of validating (864e317)
- Still validate supports_vision in VLM (864e317)
- Remove unnecessary image generation test case from demo suite (864e317)
- Remove obsolete image response validation from test containers (864e317)
- Improve score extraction logic in ImageVLMTester to handle direct float conversion and fallback to regex (1955e48)
- Organize the systems under the correct header (1955e48)
- Add missing systems header to image_systems.yaml (1955e48)
- Add missing import (1955e48)
- Make model names consistent with descriptions and make use of the openai/* entry in the litellm config file (1955e48)
- Correct typo (1955e48)
- Update evaluator system from gpt5_mini_vlm to gpt4_1_mini_vlm in image_vlm_test.yaml and corresponding system definition (1955e48)
- Update systems_under_test in demo_image_test.yaml from gpt4o_vision to gpt4_1_mini_vlm (1955e48)
- Remove evaluator.py from Dockerfile (1955e48)
- Fix (7bee7c5)
- Configure hatch-vcs for proper PyPI version detection (#283) (7180679)
- Use no-guess-dev version scheme for clean tag versions (#284) (91d1774)

### 🚜 Refactor

- Remove UsageInfo class and related usage statistics from image response schema as it is optional (864e317)
- Remove supports_vision field from VLMAPIParams (redundant validation) (864e317)
- Rename vlm_system to evaluator_system and update test parameters in image_vlm_test.yaml (1955e48)
- Rename VLMApiConfig to VLMAPIConfig for consistency across schemas and tests (35ce266)
- Remove supports_vision parameter from VLM evaluator entrypoint and manifest (we no longer need to check this in containers, as it is enforced) (35ce266)
- Remove unused supports_vision parameter from image systems configuration (35ce266)

### 📚 Documentation

- Update documentation with examples (28b5772)
- Reverted unintended change in doc update (28b5772)
- Expand documentation on configuration of image generation, image editing, and VLM API systems (276fd47)
- Add information on the minimal proof of concept test container (Image VLM Tester), detailing system requirements, input parameters, output metrics, and build instructions (276fd47)
- Remove unused response field (276fd47)
- Update configuration for image generation and VLM systems to use consistent (OpenAI model) naming conventions (276fd47)
- Make naming consistent (276fd47)
- The model in the request should match the model name in the litellm proxy config (276fd47)
- Correct model names (276fd47)
- Update test container configuration to use concrete examples for systems under test for image generation and evaluation (276fd47)

### 🧪 Testing

- Add validation for supports_vision parameter in VLMAPIConfig (35ce266)

### PLT-484

- Add support for more operators for scorecard expression calculation (#260) (28b5772)

### Security

- Bump urllib3 from 0.2.5 to 0.2.6 (#261) (982fdde)

### Signed-off-by

- Dependabot[bot] <support@github.com> (4b30d99)
- Dependabot[bot] <support@github.com> (4b30d99)
- Dependabot[bot] <support@github.com> (e843be8)

### Bugfix

- Fix Docker Hub login step and `cache-to` condition (#277) (04bc06b)
- Fix Docker Hub login step and `cache-to` condition (04bc06b)
## [0.3.3] - 2025-12-04

### 🚀 Features

- Add RAG API system type and mock rag test container (c0c12c5)
- Moved user_group to test level input parameter rather than system, improved rag response validation (c0c12c5)

### 🐛 Bug Fixes

- Ran linting and addressed comments (c0c12c5)
- Removed extra code and deleted symlink, copied file exactly (c0c12c5)
- Remove mock_rag_tester from bandit checks (c0c12c5)
- Change Dockerfile to multi-stage build like other test containers (c0c12c5)
- Removed RAG_API_KEY from configuration.md (c0c12c5)

### 📚 Documentation

- Edits to documentation (c0c12c5)
- Remove mention of first release from README (c0c12c5)
- Minor changes to documentation for user_group param (c0c12c5)
- Added score range to docs (c0c12c5)

### 🧪 Testing

- Add test for new RAG API system type (c0c12c5)
- Added tests for rag response schema validation (c0c12c5)
## [0.3.2] - 2025-12-02

### 🚀 Features

- Add version display option to CLI (#243) (b7bc8cd)
- Add version display option to CLI (b7bc8cd)
- Add version display option to CLI (#243) (b7bc8cd)

### 🐛 Bug Fixes

- Update version flag from -v to -V in CLI and tests (b7bc8cd)

### Signed-off-by

- Dependabot[bot] <support@github.com> (3879677)
- Dependabot[bot] <support@github.com> (3879677)
- Dependabot[bot] <support@github.com> (8a957e9)

### Bugfix

- Add score card schema (#250) (257236d)
## [0.3.1] - 2025-11-19

### 🚀 Features

- Add `workflow_id` container label (eb83988)
- Add unit tests and address review comment (eb83988)
## [0.3.0] - 2025-11-13

### 🚀 Features

- Add unique id field to indicators (#226) (986f849)
- Add unique ID field for indicators and update validation (986f849)
- Add tests for indicator ids and id validation (986f849)

### 🐛 Bug Fixes

- Fix JSON parsing logic in parse_container_json_output for improved robustness and clarity (#231) (93f15cc)
- Fix JSON parsing logic in parse_container_json_output for improved robustness and clarity (93f15cc)
- Add test cases for json output parsing (93f15cc)
- Move results from logs to normal ouput and trimmed garak output (d2f3e39)
- Update `chatbot_asqi_assessment` file (d2f3e39)
- Fix logdir name display (1b0977f)
- Refactor code based on review feedback (986f849)

### 📚 Documentation

- Add indicator id field to score card configs and docs (986f849)
- Update `chatbot_asqi_assessment` to include new indicator id field (986f849)

### Signed-off-by

- Dependabot[bot] <support@github.com> (d452742)
- Dependabot[bot] <support@github.com> (f984e3c)
- Dependabot[bot] <support@github.com> (f984e3c)
## [0.2.1] - 2025-10-23

### 🚀 Features

- PLT-183 add log output for container results (3e4ecd1)
- PLT-183 add description and providers fields for systems, description for suites (733e32d)

### 🐛 Bug Fixes

- Fix/182 - Add DOCKER_HOST to .env file in setup.sh (#183) (6b7db3e)
- Add DOCKER_HOST to .env file. (6b7db3e)
- Make Docker socket path configurable for Docker-in-docker tests (6b7db3e)
- Fix/204 - Add validation for LLM API params (#206) (b92de80)
- Add validation for LLM API params (b92de80)
- Fix/193 outdated output (#205) (5facd92)
- Output unused fields and add logs (5facd92)
- Include test container results in score card evaluation and update chatbot_asqi_assessment (5facd92)

### 📚 Documentation

- PLT-183 add env file configuration (3e4ecd1)
- PLT-183 update configuration yaml with new fields (733e32d)
- Add library usage guide (#199) (bb42145)
- Add library usage guide (bb42145)

### Signed-off-by

- Dependabot[bot] <support@github.com> (f3bc048)
- Dependabot[bot] <support@github.com> (392138c)
- Dependabot[bot] <support@github.com> (392138c)
## [0.2.0] - 2025-10-01

### 🚀 Features

- Add optional container name parameter to run_container_with_args (#158) (ba02dc7)
- Add optional container name parameter to run_container_with_args (ba02dc7)
- Enhance evaluation configurations for cybersecurity and reasoning tests (055c2dd)
- Add Inspect Scheming Evaluation (055c2dd)
- Update Inspect Evals manifest with detailed task descriptions and parameters (055c2dd)
- Restricted docker-out-of-docker capabilities (055c2dd)
- Update documentation and README.md to include inspect_evals (055c2dd)

### 🐛 Bug Fixes

- Add test name validation to check when no scorecard indicators match any test results' test names (#167) (d785c17)
- Scorecard test names validation to prevent error-filled results with grades (d785c17)
- Handle HTTP timeouts gracefully during test execution (#174) (8e64723)
- Handle HTTP timeouts gracefully during test execution (8e64723)
- Refactor timeout exception handling to use a constant tuple (8e64723)
- Fix more tests (055c2dd)
- Update evaluation configurations (055c2dd)
- Handle manifest extraction failure gracefully in test execution (055c2dd)
- Fix deepeval version (a714e3b)

### 🚜 Refactor

- Use set comprehension directly to avoid casts (d785c17)
- Update Dockerfile and entrypoint.py for improved dataset handling and environment variable configuration (055c2dd)
- Restructure Dockerfile for improved build stages and environment setup (055c2dd)

### 📚 Documentation

- Docker host (055c2dd)

### 🧪 Testing

- Modify tests for evaluate_scorecard, checking both partial and no matches (d785c17)
- Enhance error handling in workflow and container execution tests (8e64723)

### Signed-off-by

- Dependabot[bot] <support@github.com> (701827d)
- Dependabot[bot] <support@github.com> (701827d)
## [0.1.2] - 2025-09-10

### 📚 Documentation

- Update README.md to enhance clarity and structure of project overview (#145) (1189730)
- Update README.md to enhance clarity and structure of project overview (#145) (1189730)
- Add contribution guide with detailed interaction steps and guidelines (1189730)
- Simplify introduction in contribution guide for clarity (1189730)
- Update contribution guide to streamline pre-commit instructions and enhance clarity (1189730)
- Remove redundant help and contact section from contribution guide (1189730)
- Standardize section titles in contribution guide for consistency (1189730)
- Restructure contribution and development documentation for clarity and accessibility (1189730)
- Fix heading formatting in contribution guide (1189730)

### Signed-off-by

- Dependabot[bot] <support@github.com> (7bcedfd)
## [0.1.1] - 2025-09-03
## [0.1.0] - 2025-09-03

### 🐛 Bug Fixes

- Fix tarfile extraction (e7d8084)
- Don't pass exceptions (e7d8084)
- Fixed issue from github action (8890881)
- Fix deepteam version and opt out of data collection (#52) (85b16fa)
- Fix garak to v0.12.0 (165faf3)
- Fix test issue (26f4eb0)

### 🚜 Refactor

- Refactor assert (ceb1bac)

### Signed-off-by

- Dependabot[bot] <support@github.com> (87b2623)
