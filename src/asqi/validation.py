import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel
from rich.console import Console

from asqi.config import ExecutionMode, load_config_file
from asqi.errors import DuplicateIDError, MissingIDFieldError
from asqi.schemas import (
    HF_DATASET_TYPE,
    AuditScoreCardIndicator,
    EnvironmentVariable,
    FileDatasetConfig,
    HFDatasetConfig,
    Manifest,
    ScoreCard,
    SuiteConfig,
    SystemsConfig,
    TestDefinition,
)

logger = logging.getLogger()

# Initialize Rich console and execution queue
console = Console()


def validate_ids(*config_paths: str) -> None:
    """
    Validate that all IDs within a config file are unique.
    Supports multiple config files and reports duplicates found across all of them.

    Args:
        *config_paths: Variable number of configuration file paths to validate

    Notes:
        Only supports:
            - test suite configs
            - score card configs

    Raises:
        DuplicateIDError: If duplicate IDs are found
        MissingIDFieldError: If required ID fields are missing
    """
    all_ids: Dict[str, Any] = {}
    for config_path in config_paths:
        if config_path:
            try:
                config_dict = load_config_file(config_path)
                extract_ids(all_ids, config_dict, config_path)

            except yaml.YAMLError as e:
                console.print(
                    f"[yellow]ID Validation. Failed to parse config file {config_path}:[/yellow] {e}"
                )
            except MissingIDFieldError as e:
                raise e
            except Exception as e:
                console.print(
                    f"[yellow]ID Validation. Failed to process config file {config_path}:[/yellow] {e}"
                )
    duplicate_dict = get_duplicate_ids(all_ids)
    if duplicate_dict:
        raise DuplicateIDError(duplicate_dict)


def extract_ids(
    all_ids: Dict[str, Any],
    config_dict: Dict[str, Any],
    config_path: str,
) -> None:
    """
    Extract IDs from a config file and register them in the ID state.

    Args:
        all_ids: Dictionary tracking all collected IDs
        config_dict: Parsed configuration dictionary
        config_path: Path to the config file

    Notes:
        Only processes IDs from:
            - score cards
            - test suites
    """

    score_card_name = config_dict.get("score_card_name", "")
    test_suite_name = config_dict.get("suite_name", "")

    if score_card_name:
        indicators = config_dict.get("indicators", [])
        for indicator in indicators:
            indicator_id = indicator.get("id", "")
            if not indicator_id:
                raise MissingIDFieldError(
                    f"Missing required id field in indicator of score card '{score_card_name}'"
                )

            indicator_name = indicator.get("name", "")
            indicator_key = f"s_{indicator_id}"

            if indicator_key not in all_ids:
                all_ids[indicator_key] = {
                    "config_type": "score_card",
                    "id": f"{indicator_id}",
                    "occurrences": [],
                }

            all_ids[indicator_key]["occurrences"].append(
                {
                    "location": config_path,
                    "score_card_name": score_card_name,
                    "indicator_name": indicator_name,
                }
            )

    elif test_suite_name:
        test_suite = config_dict.get("test_suite", [])
        for test in test_suite:
            test_id = test.get("id", "")
            if not test_id:
                raise MissingIDFieldError(
                    f"Missing required id field in test of test suite '{test_suite_name}'"
                )

            test_name = test.get("name", "")
            test_name_key = f"t_{test_id}"

            if test_name_key not in all_ids:
                all_ids[test_name_key] = {
                    "config_type": "test_suite",
                    "id": f"{test_id}",
                    "occurrences": [],
                }

            all_ids[test_name_key]["occurrences"].append(
                {
                    "location": config_path,
                    "test_suite_name": test_suite_name,
                    "test_name": test_name,
                }
            )


def get_duplicate_ids(all_ids: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find only duplicate ID entries from the collected IDs.

    Args:
        all_ids: Current state of the project IDs

    Returns:
        Dictionary containing duplicate groups based on their key (ID and file type).
        Returns an empty dictionary if no duplicates found.
    """
    duplicate_ids = {}
    for k, duplicate in all_ids.items():
        if len(duplicate["occurrences"]) > 1:
            duplicate_ids[k] = duplicate
    return duplicate_ids


def validate_test_volumes(
    suite: SuiteConfig,
    allowed_keys: tuple[str, ...] = ("input", "output"),
    require_at_least_one: bool = True,
) -> None:
    """
    Validate per-test volumes and raise ValueError on the first problem.

    Rules:
    - volumes may be omitted entirely (skip)
    - if present, must be a dict
    - require_at_least_one=True => at least one of allowed_keys must be present
    - only validate keys that are present
    - each provided path must be a non-empty string, exist, and be a directory
    """
    allowed = set(allowed_keys)

    for test in suite.test_suite:
        vols = getattr(test, "volumes", None)
        if not vols:
            continue

        if not isinstance(vols, dict):
            raise ValueError(
                f"'volumes' for test '{test.name}' must be a dict, got {type(vols).__name__}"
            )

        present = allowed & vols.keys()
        if require_at_least_one and not present:
            raise ValueError(
                f"Test '{test.name}' must specify at least one of: {', '.join(sorted(allowed))}"
            )

        for key in present:
            raw_path = vols[key]
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(
                    f"Invalid '{key}' volume path in test '{test.name}': must be a non-empty string."
                )

            path = Path(raw_path).expanduser().resolve()

            if not path.exists():
                raise ValueError(
                    f"Configured '{key}' volume does not exist for test '{test.name}': {path}"
                )

            if not path.is_dir():
                raise ValueError(
                    f"Configured '{key}' volume is not a directory for test '{test.name}': {path}"
                )


def validate_test_parameters(test, manifest: Manifest) -> List[str]:
    """
    Validate test parameters against manifest schema.

    Args:
        test: Test definition from suite config
        manifest: Manifest for the test container

    Returns:
        List of validation error messages
    """
    errors = []
    schema_params = {p.name: p for p in manifest.input_schema}

    # Check for required but missing params
    test_params = test.params or {}
    for schema_param in manifest.input_schema:
        if schema_param.required and schema_param.name not in test_params:
            errors.append(
                f"Test '{test.name}': Missing required parameter '{schema_param.name}' (type: {schema_param.type}, description: {schema_param.description or 'none'})"
            )

    # Check for unknown params
    for provided_param in test_params:
        if provided_param not in schema_params:
            valid_params = ", ".join(schema_params.keys()) if schema_params else "none"
            errors.append(
                f"Test '{test.name}': Unknown parameter '{provided_param}'. Valid parameters: {valid_params}"
            )

    return errors


def validate_dataset_configs(test: TestDefinition, manifest: Manifest) -> List[str]:
    """
    Validate test.datasets against the manifest's input_datasets schema.

    - Ensures required datasets in input_datasets are provided.
    - Ensures no unknown dataset names are provided in test.datasets.
    - Ensures feature mappings for each of test.datasets do not include unknown features.
    """
    errors = []
    if test.datasets is None:
        test_datasets = {}
    else:
        test_datasets = test.datasets

    # Check for required but missing datasets
    for schema_dataset in manifest.input_datasets:
        if schema_dataset.required and schema_dataset.name not in test_datasets:
            errors.append(
                f"Test '{test.name}': Missing required dataset '{schema_dataset.name}' (description: {schema_dataset.description or 'none'})"
            )

        elif schema_dataset.name in test_datasets:
            if schema_dataset.type == HF_DATASET_TYPE:
                if not isinstance(test_datasets[schema_dataset.name], HFDatasetConfig):
                    errors.append(
                        f"Test '{test.name}':  Dataset '{schema_dataset.name}' of type {HF_DATASET_TYPE} must have config of type HFDatasetConfig."
                    )
                else:
                    # Check for unknown mapped fields.
                    expected_features = {f.name for f in schema_dataset.features}
                    unknown_mappings: list[str] = []
                    for mapped_feature in test_datasets[
                        schema_dataset.name
                    ].mapping.keys():
                        if mapped_feature not in expected_features:
                            unknown_mappings.append(mapped_feature)
                    if unknown_mappings:
                        errors.append(
                            f"Test '{test.name}': Unknown feature mappings '{', '.join(unknown_mappings)}' in dataset '{schema_dataset.name}'. Valid features: {', '.join(expected_features) if expected_features else 'none'}"
                        )

            elif not isinstance(test_datasets[schema_dataset.name], FileDatasetConfig):
                errors.append(
                    f"Test '{test.name}': Dataset '{schema_dataset.name}' of type {schema_dataset.type} must have config of type FileDatasetConfig."
                )

    # Check for unknown dataset names
    schema_datasets = {d.name: d for d in manifest.input_datasets}
    for provided_dataset_name in test_datasets:
        if provided_dataset_name not in schema_datasets:
            errors.append(
                f"Test '{test.name}': Unknown dataset '{provided_dataset_name}'. Valid datasets: {', '.join(schema_datasets.keys()) if schema_datasets else 'none'}"
            )

    return errors


def validate_system_compatibility(
    test, system_definitions: Dict, manifest: Manifest
) -> List[str]:
    """
    Validate system compatibility with test container.

    Args:
        test: Test definition from suite config
        system_definitions: Dictionary of system definitions
        manifest: Manifest for the test container

    Returns:
        List of validation error messages
    """
    errors = []

    # Create a map of required vs optional systems from manifest
    required_systems = {}
    optional_systems = {}

    for system_input in manifest.input_systems:
        if system_input.required:
            required_systems[system_input.name] = system_input.type
        else:
            optional_systems[system_input.name] = system_input.type

    # Validate systems_under_test (maps to system_under_test in manifest)
    # Get all supported types for system_under_test from manifest
    supported_types_for_system = [
        s.type for s in manifest.input_systems if s.name == "system_under_test"
    ]

    # Allow systems_under_test to be None or empty if not required e.g. datagen containers
    target_systems = test.systems_under_test or []

    for system_name in target_systems:
        if system_name not in system_definitions:
            available_systems = (
                ", ".join(system_definitions.keys()) if system_definitions else "none"
            )
            errors.append(
                f"Test '{test.name}': Unknown system '{system_name}'. Available systems: {available_systems}"
            )
            continue

        system_def = system_definitions[system_name]
        if system_def.type not in supported_types_for_system:
            supported_types = (
                ", ".join(supported_types_for_system)
                if supported_types_for_system
                else "none"
            )
            errors.append(
                f"Test '{test.name}' on system '{system_name}': "
                f"Image '{test.image}' does not support system type '{system_def.type}'. Supported types: {supported_types}"
            )

    # Validate additional systems from test.systems field
    if hasattr(test, "systems") and test.systems:
        # Combine required and optional systems for validation
        all_systems = {**required_systems, **optional_systems}

        for system_role, system_name in test.systems.items():
            # Check if this system role is declared in manifest
            expected_type = all_systems.get(system_role)
            if not expected_type:
                valid_roles = ", ".join(all_systems.keys()) if all_systems else "none"
                errors.append(
                    f"Test '{test.name}': Unknown system role '{system_role}'. Valid roles: {valid_roles}"
                )
                continue

            # Check if referenced system exists
            if system_name not in system_definitions:
                available_systems = (
                    ", ".join(system_definitions.keys())
                    if system_definitions
                    else "none"
                )
                errors.append(
                    f"Test '{test.name}': Unknown {system_role} '{system_name}'. Available systems: {available_systems}"
                )
                continue

            # Check if system type matches expected type
            system_def = system_definitions[system_name]
            if system_def.type != expected_type:
                errors.append(
                    f"Test '{test.name}' {system_role} '{system_name}': Expected type '{expected_type}', got '{system_def.type}'"
                )

    return errors


def find_manifest_for_image(
    image_name: str, manifests: Dict[str, Manifest]
) -> Optional[Manifest]:
    """
    Find manifest for a given image name.

    For runtime (workflow), manifests are keyed by full image names.
    For local validation, manifests are keyed by container directory names.

    Args:
        image_name: Full image name (e.g., "my-registry/mock_tester:latest")
        manifests: Dictionary of available manifests

    Returns:
        Manifest if found, None otherwise
    """
    # Try exact match first (for runtime/workflow usage)
    if image_name in manifests:
        return manifests[image_name]

    # For local validation, try to match by container name
    # Extract container name from image (e.g., "my-registry/mock_tester:latest" -> "mock_tester")
    if "/" in image_name:
        container_name = image_name.split("/")[-1].split(":")[0]
        if container_name in manifests:
            return manifests[container_name]

    # Also try the base name without registry/tag
    base_name = image_name.split(":")[0].split("/")[-1]
    if base_name in manifests:
        return manifests[base_name]

    return None


def validate_manifests_against_tests(
    suite: SuiteConfig, systems: SystemsConfig, manifests: Dict[str, Manifest]
) -> List[str]:
    """
    Validate that all tests can be executed with available manifests.

    Args:
        suite: Test suite configuration
        systems: Systems configuration
        manifests: Dictionary of available manifests

    Returns:
        List of validation error messages
    """
    errors = []
    # Get the system definitions
    system_definitions = systems.systems

    for test in suite.test_suite:
        # Check if manifest exists for test image
        manifest = find_manifest_for_image(test.image, manifests)
        if manifest is None:
            available_images = ", ".join(manifests.keys()) if manifests else "none"
            errors.append(
                f"Test '{test.name}': No manifest available for image '{test.image}'. Images with manifests: {available_images}"
            )
            continue

        # Validate test parameters
        param_errors = validate_test_parameters(test, manifest)
        errors.extend(param_errors)

        # Validate provided dataset configs
        dataset_errors = validate_dataset_configs(test, manifest)
        errors.extend(dataset_errors)

        # Validate system compatibility
        system_errors = validate_system_compatibility(
            test, system_definitions, manifest
        )
        errors.extend(system_errors)

    return errors


def create_test_execution_plan(
    suite: SuiteConfig, systems: SystemsConfig, image_availability: Dict[str, bool]
) -> List[Dict[str, Any]]:
    """
    Create execution plan for all valid test combinations.

    Args:
        suite: Test suite configuration
        systems: Systems configuration
        image_availability: Dictionary of image availability status

    Returns:
        List of test execution plans
    """

    def get_system_params_dict(params):
        """
        Returns a dict of a system's params
        Handles GenericSystemConfig and Pydantic models

        Args:
            params: System params

        Returns:
             Dict with the system params
        """
        if isinstance(params, BaseModel):
            return params.model_dump()
        return params

    if not suite or not suite.test_suite:
        return []

    # Get the system definitions
    system_definitions = systems.systems
    if not systems or not system_definitions:
        return []

    plan: List[Dict[str, Any]] = []
    available_images = {img for img, ok in image_availability.items() if ok}

    for test in suite.test_suite:
        if not (image := getattr(test, "image", None)):
            logger.warning(f"Skipping test with missing image: {test}")
            continue

        if image not in available_images:
            continue

        vols = getattr(test, "volumes", None)
        datasets = getattr(test, "datasets", None)
        base_params = getattr(test, "params", None)

        if vols:
            _params = dict(base_params or {})
            _params["__volumes"] = vols  # reserved key
            _params["volumes"] = vols  # Also pass volumes directly for container access
            if datasets:
                _params["datasets"] = {
                    name: (
                        config.model_dump()
                        if hasattr(config, "model_dump")
                        else config.dict()
                        if hasattr(config, "dict")
                        else config
                    )
                    for name, config in datasets.items()
                }
            test_params = _params
        else:
            if datasets:
                raise ValueError(
                    f"Test config for {test.name} provided with datasets {datasets} but without volumes; datasets must be passed in via mounted volumes."
                )
            test_params = base_params or {}

        systems_params = {}
        # Add additional systems if specified
        if hasattr(test, "systems") and test.systems:
            for system_role, referenced_system_name in test.systems.items():
                referenced_system_def = system_definitions.get(referenced_system_name)
                if referenced_system_def:
                    systems_params[system_role] = {
                        "type": referenced_system_def.type,
                        "description": referenced_system_def.description,
                        "provider": referenced_system_def.provider,
                        **get_system_params_dict(referenced_system_def.params),
                    }

        # Get the target systems
        target_systems = test.systems_under_test or []

        # Process valid combinations
        for system_name in target_systems:
            system_def = system_definitions.get(system_name)
            if not system_def or not getattr(system_def, "type", None):
                continue

            # Build unified systems_params with system_under_test and additional systems
            systems_params["system_under_test"] = {
                k: v
                for k, v in {
                    "type": system_def.type,
                    "description": system_def.description,
                    "provider": system_def.provider,
                    **get_system_params_dict(system_def.params),
                }.items()
                if v is not None
            }

            plan.append(
                {
                    "test_name": test.name,
                    "test_id": test.id,
                    "image": image,
                    "sut_name": system_name,
                    "systems_params": systems_params,
                    "test_params": test_params,
                    "env_file": test.env_file,
                    "environment": test.environment,
                }
            )

        if not target_systems:
            # No target systems, just add the test with its params
            plan.append(
                {
                    "test_name": test.name,
                    "test_id": test.id,
                    "image": image,
                    "sut_name": None,
                    "systems_params": systems_params,
                    "test_params": test_params,
                    "env_file": test.env_file,
                    "environment": test.environment,
                }
            )

    return plan


def validate_test_plan(
    suite: SuiteConfig, systems: SystemsConfig, manifests: Dict[str, Manifest]
) -> List[str]:
    """
    Validates the entire test plan by cross-referencing the suite, systems, and manifests.

    Args:
        suite: The parsed SuiteConfig object.
        systems: The parsed systems configuration object.
        manifests: A dictionary mapping image names to their parsed Manifest objects.

    Returns:
        A list of error strings. An empty list indicates successful validation.
    """
    errors = []
    # Get the system definitions
    system_definitions = systems.systems

    for test in suite.test_suite:
        # 1. Check if the test's image has a corresponding manifest
        manifest = find_manifest_for_image(test.image, manifests)
        if manifest is None:
            errors.append(
                f"Test '{test.name}': Image '{test.image}' does not have a loaded manifest."
            )
            continue  # Cannot perform further validation for this test
        supported_system_types = [
            s.type for s in manifest.input_systems if s.name == "system_under_test"
        ]

        # 2. Check parameters against the manifest's input_schema
        schema_params = {p.name: p for p in manifest.input_schema}

        # Check for required but missing params
        test_params = test.params or {}
        for schema_param in manifest.input_schema:
            if schema_param.required and schema_param.name not in test_params:
                errors.append(
                    f"Test '{test.name}': Required parameter '{schema_param.name}' is missing."
                )

        # Check for unknown params
        for provided_param in test_params:
            if provided_param not in schema_params:
                errors.append(
                    f"Test '{test.name}': Unknown parameter '{provided_param}' is not defined in manifest for '{test.image}'."
                )

        # 3. For each target system, perform validation
        # Get the target systems
        target_systems = test.systems_under_test or []

        for system_name in target_systems:
            # 3a. Check if the system exists in the systems config
            if system_name not in system_definitions:
                errors.append(
                    f"Test '{test.name}': Target system '{system_name}' is not defined in the systems file."
                )
                continue  # Cannot perform further validation for this system

            system_def = system_definitions[system_name]

            # 3b. Check if the container supports the system's type
            if system_def.type not in supported_system_types:
                errors.append(
                    f"Test '{test.name}' on system '{system_name}': Image '{test.image}' "
                    f"(supports: {supported_system_types}) is not compatible with system type '{system_def.type}'."
                )

        # 4. Validate optional simulator_system and evaluator_system fields
        for optional_system_field in ["simulator_system", "evaluator_system"]:
            optional_system_name = getattr(test, optional_system_field, None)
            if optional_system_name:
                if optional_system_name not in system_definitions:
                    errors.append(
                        f"Test '{test.name}': {optional_system_field} '{optional_system_name}' is not defined in the systems file."
                    )
                else:
                    optional_system_def = system_definitions[optional_system_name]
                    if optional_system_def.type not in supported_system_types:
                        errors.append(
                            f"Test '{test.name}' {optional_system_field} '{optional_system_name}': Image '{test.image}' "
                            f"(supports: {supported_system_types}) is not compatible with system type '{optional_system_def.type}'."
                        )

    return errors


def validate_execution_inputs(
    suite_path: str,
    systems_path: str,
    execution_mode: ExecutionMode,
    audit_responses_data: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Validate inputs for test execution workflows.

    Args:
        suite_path: Path to test suite YAML file
        systems_path: Path to systems YAML file
        execution_mode: Execution mode
        audit_responses_data: Optional dictionary of audit responses data
        output_path: Optional output file path

    Raises:
        ValueError: If any input is invalid
    """
    if not suite_path or not isinstance(suite_path, str):
        raise ValueError("Invalid suite_path: must be non-empty string")

    if not systems_path or not isinstance(systems_path, str):
        raise ValueError("Invalid systems_path: must be non-empty string")

    if execution_mode not in [ExecutionMode.TESTS_ONLY, ExecutionMode.END_TO_END]:
        raise ValueError(
            f"Invalid execution_mode '{execution_mode}': must be 'ExecutionMode.TESTS_ONLY' or 'ExecutionMode.END_TO_END'"
        )
    if audit_responses_data is not None and not isinstance(audit_responses_data, dict):
        raise ValueError("Invalid audit_responses_data: must be a dictionary or None")

    if output_path is not None and not isinstance(output_path, str):
        raise ValueError("Invalid output_path: must be string or None")


def validate_score_card_inputs(
    input_path: str,
    score_card_configs: List[Dict[str, Any]],
    audit_responses_data: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Validate inputs for score card evaluation workflows.

    Args:
        input_path: Path to input JSON file
        score_card_configs: List of score card configurations
        audit_responses_data: Optional dictionary of audit responses data
        output_path: Optional output file path

    Raises:
        ValueError: If any input is invalid
    """
    if not input_path or not isinstance(input_path, str):
        raise ValueError("Invalid input_path: must be non-empty string")

    if not score_card_configs or not isinstance(score_card_configs, list):
        raise ValueError("Invalid score_card_configs: must be non-empty list")

    if audit_responses_data is not None and not isinstance(audit_responses_data, dict):
        raise ValueError("Invalid audit_responses_data: must be a dictionary or None")

    if output_path is not None and not isinstance(output_path, str):
        raise ValueError("Invalid output_path: must be string or None")


def validate_test_execution_inputs(
    test_id: str,
    image: str,
    system_name: str,
    system_params: Dict[str, Any],
    test_params: Dict[str, Any],
) -> None:
    """
    Validate inputs for individual test execution.

    Args:
        test_id: ID of the test
        image: Docker image name
        system_name: Name of the system
        system_params: System parameters dictionary (flattened configuration)
        test_params: Test parameters dictionary

    Raises:
        ValueError: If any input is invalid
    """

    if not test_id or not isinstance(test_id, str):
        raise ValueError("Invalid test id: must be non-empty string")

    if not image or not isinstance(image, str):
        raise ValueError("Invalid image: must be non-empty string")

    if system_name and not isinstance(system_name, str):
        raise ValueError("Invalid system name: must be non-empty string")

    if not isinstance(system_params, dict):
        raise ValueError("Invalid system parameters: must be dictionary")

    if not isinstance(test_params, dict):
        raise ValueError("Invalid test parameters: must be dictionary")


def build_env_var_error_message(
    missing_vars: List["EnvironmentVariable"],
    test_name: str,
    image: str,
) -> str:
    """
    Build helpful error message for missing environment variables.

    Args:
        missing_vars: List of EnvironmentVariable objects that are missing
        test_name: Name of the test being executed
        image: Docker image being used

    Returns:
        Formatted error message with suggestions for fixing the issue
    """
    var_list = "\n".join(
        [
            f"  - {var.name}: {var.description or 'No description'}"
            for var in missing_vars
        ]
    )

    var_names = ", ".join(v.name for v in missing_vars)

    env_dict_lines = "\n".join(
        f'    {var.name}: "${{{var.name}}}"' for var in missing_vars
    )

    return f"""Test '{test_name}' (image: {image}) requires environment variables that are not available:

{var_list}

To fix this, add to your test configuration:

Option 1 - Use env_file:
  env_file: ".env"  # File containing {var_names}

Option 2 - Specify environment variables directly:
  environment:
{env_dict_lines}

Then ensure these variables are set in your shell or .env file.
"""


def validate_workflow_configurations(
    suite: SuiteConfig,
    systems: SystemsConfig,
    manifests: Optional[Dict[str, Manifest]] = None,
) -> List[str]:
    """
    Comprehensive validation of workflow configurations.

    Combines all configuration validation checks in one place.

    Args:
        suite: Test suite configuration
        systems: Systems configuration
        manifests: Optional manifests dictionary

    Returns:
        List of validation error messages

    Raises:
        ValueError: If configuration objects are invalid
    """
    errors = []

    # Basic structure validation
    if not isinstance(suite, SuiteConfig):
        raise ValueError("Invalid suite: must be SuiteConfig instance")

    if not isinstance(systems, SystemsConfig):
        raise ValueError("Invalid systems: must be SystemsConfig instance")

    if manifests is not None and not isinstance(manifests, dict):
        raise ValueError("Invalid manifests: must be dictionary")

    # Content validation
    if not suite.test_suite:
        errors.append("Test suite is empty: no tests to validate")

    # Get the system definitions
    system_definitions = systems.systems
    if not system_definitions:
        errors.append("Systems configuration is empty: no systems defined")

    # Detailed validation if manifests are provided
    if manifests is not None and not errors:  # Only if basic validation passes
        errors.extend(validate_manifests_against_tests(suite, systems, manifests))

    return errors


def validate_indicator_display_reports(
    manifests: Dict[str, Manifest],
    score_cards: List[ScoreCard],
    test_id_to_image: Dict[str, str],
) -> List[str]:
    """
    Validate that all requested `display_reports` exist in the corresponding test container manifest and that there are no duplicate report names.

    Args:
        manifests: Dictionary linking each image to its manifest
        score_cards: List of score card configurations to validate
        test_id_to_image: Dictionary linking each test id to the image used

    Returns:
        List of validation error messages or empty list if none found
    """
    errors = []
    if not score_cards:
        return errors

    for score_card in score_cards:
        for indicator in score_card.indicators:
            if isinstance(indicator, AuditScoreCardIndicator):
                continue
            test_id = indicator.apply_to.test_id
            if test_id not in test_id_to_image:
                errors.append(
                    f"Indicator id '{indicator.id}': Referenced test id '{test_id}' does not exist in the test"
                )
                continue
            image = test_id_to_image[test_id]
            manifest = manifests.get(image)
            if not manifest:
                errors.append(
                    f"Indicator id '{indicator.id}': No manifest found for image '{image}' (test '{test_id}')"
                )
                continue

            available_reports = {
                report.name for report in (manifest.output_reports or [])
            }
            # Check for duplicate report names in display_reports
            reports_set = set()
            for report_name in indicator.display_reports:
                if report_name in reports_set:
                    errors.append(
                        f"Indicator id '{indicator.id}': Duplicate report name '{report_name}' in display_reports"
                    )
                reports_set.add(report_name)

            # Validate that the reports are declared in the test container manifest
            for requested_report in indicator.display_reports:
                if requested_report not in available_reports:
                    errors.append(
                        f"Indicator id '{indicator.id}': Requested display_report '{requested_report}' from test '{test_id}' not found with an exact case sensitive match. Manifest for image '{image}' only defines: {sorted(list(available_reports)) if available_reports else 'none'}"
                    )

    return errors


def verify_score_card_reports(all_evaluations: List[Dict[str, Any]]) -> None:
    """
    Verifies that all generated reports referenced in the score card evaluations
    exist on the local filesystem and logs the results to the console.

    Args:
        all_evaluations: List of score card evaluation results
    """
    if not all_evaluations:
        return

    console.print("\n[bold blue]Verifying generated reports...[/bold blue]")
    reports_count = 0
    for evaluation in all_evaluations:
        indicator_id = evaluation.get("indicator_id", "")
        report_paths = evaluation.get("report_paths") or []
        for report_path_str in report_paths:
            try:
                path = Path(report_path_str)
                if path.exists() and path.is_file():
                    reports_count += 1
                    console.print(
                        f"Indicator id [bold]'{indicator_id}'[/bold]: Report saved to [bold]{report_path_str}[/bold]"
                    )
                else:
                    console.print(
                        f"Indicator id [bold]'{indicator_id}'[/bold]: Report [red]{path.name}[/red] is missing. Current path: {report_path_str}"
                    )
            except (TypeError, ValueError, OSError) as e:
                console.print(
                    f"Indicator id [bold]'{indicator_id}'[/bold]: Invalid report path [red]{report_path_str}[/red] ({str(e)})"
                )
    if reports_count == 0:
        console.print("No reports were generated")
