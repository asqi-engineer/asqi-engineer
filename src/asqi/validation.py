import logging
from typing import Any, Dict, List, Optional

from asqi.schemas import Manifest, SuiteConfig, SystemsConfig

logger = logging.getLogger()


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
    for schema_param in manifest.input_schema:
        if schema_param.required and schema_param.name not in test.params:
            errors.append(
                f"Test '{test.name}': Missing required parameter '{schema_param.name}' (type: {schema_param.type}, description: {schema_param.description or 'none'})"
            )

    # Check for unknown params
    for provided_param in test.params:
        if provided_param not in schema_params:
            valid_params = ", ".join(schema_params.keys()) if schema_params else "none"
            errors.append(
                f"Test '{test.name}': Unknown parameter '{provided_param}'. Valid parameters: {valid_params}"
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

    target_systems = test.systems_under_test

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
        for system_role, system_name in test.systems.items():
            # Check if this system role is declared in manifest
            expected_type = optional_systems.get(system_role)
            if not expected_type:
                valid_roles = (
                    ", ".join(optional_systems.keys()) if optional_systems else "none"
                )
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

        # Get the target systems
        target_systems = test.systems_under_test

        if not target_systems:
            logger.warning(f"Skipping test '{test.name}' with no target systems")
            continue

        # Process valid combinations
        for system_name in target_systems:
            system_def = system_definitions.get(system_name)
            if not system_def or not getattr(system_def, "type", None):
                continue

            vols = getattr(test, "volumes", None)
            base_params = getattr(test, "params", None)

            if vols:
                _params = dict(base_params or {})
                _params["__volumes"] = vols  # reserved key
                test_params = _params
            else:
                test_params = base_params or {}

            # Build unified systems_params with system_under_test and additional systems
            systems_params = {
                "system_under_test": {"type": system_def.type, **system_def.params}
            }

            # Add additional systems if specified
            if hasattr(test, "systems") and test.systems:
                for system_role, referenced_system_name in test.systems.items():
                    referenced_system_def = system_definitions.get(
                        referenced_system_name
                    )
                    if referenced_system_def:
                        systems_params[system_role] = {
                            "type": referenced_system_def.type,
                            **referenced_system_def.params,
                        }

            plan.append(
                {
                    "test_name": test.name,
                    "image": image,
                    "sut_name": system_name,
                    "systems_params": systems_params,
                    "test_params": test_params,
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
        for schema_param in manifest.input_schema:
            if schema_param.required and schema_param.name not in test.params:
                errors.append(
                    f"Test '{test.name}': Required parameter '{schema_param.name}' is missing."
                )

        # Check for unknown params
        for provided_param in test.params:
            if provided_param not in schema_params:
                errors.append(
                    f"Test '{test.name}': Unknown parameter '{provided_param}' is not defined in manifest for '{test.image}'."
                )

        # 3. For each target system, perform validation
        # Get the target systems
        target_systems = test.systems_under_test

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
    execution_mode: str,
    output_path: Optional[str] = None,
) -> None:
    """
    Validate inputs for test execution workflows.

    Args:
        suite_path: Path to test suite YAML file
        systems_path: Path to systems YAML file
        execution_mode: Execution mode string
        output_path: Optional output file path

    Raises:
        ValueError: If any input is invalid
    """
    if not suite_path or not isinstance(suite_path, str):
        raise ValueError("Invalid suite_path: must be non-empty string")

    if not systems_path or not isinstance(systems_path, str):
        raise ValueError("Invalid systems_path: must be non-empty string")

    if execution_mode not in ["tests_only", "end_to_end"]:
        raise ValueError(
            f"Invalid execution_mode '{execution_mode}': must be 'tests_only' or 'end_to_end'"
        )

    if output_path is not None and not isinstance(output_path, str):
        raise ValueError("Invalid output_path: must be string or None")


def validate_score_card_inputs(
    input_path: str,
    score_card_configs: List[Dict[str, Any]],
    output_path: Optional[str] = None,
) -> None:
    """
    Validate inputs for score card evaluation workflows.

    Args:
        input_path: Path to input JSON file
        score_card_configs: List of score card configurations
        output_path: Optional output file path

    Raises:
        ValueError: If any input is invalid
    """
    if not input_path or not isinstance(input_path, str):
        raise ValueError("Invalid input_path: must be non-empty string")

    if not score_card_configs or not isinstance(score_card_configs, list):
        raise ValueError("Invalid score_card_configs: must be non-empty list")

    if output_path is not None and not isinstance(output_path, str):
        raise ValueError("Invalid output_path: must be string or None")


def validate_test_execution_inputs(
    test_name: str,
    image: str,
    system_name: str,
    system_params: Dict[str, Any],
    test_params: Dict[str, Any],
) -> None:
    """
    Validate inputs for individual test execution.

    Args:
        test_name: Name of the test
        image: Docker image name
        system_name: Name of the system
        system_params: System parameters dictionary (flattened configuration)
        test_params: Test parameters dictionary

    Raises:
        ValueError: If any input is invalid
    """
    if not test_name or not isinstance(test_name, str):
        raise ValueError("Invalid test name: must be non-empty string")

    if not image or not isinstance(image, str):
        raise ValueError("Invalid image: must be non-empty string")

    if not system_name or not isinstance(system_name, str):
        raise ValueError("Invalid system name: must be non-empty string")

    if not isinstance(system_params, dict):
        raise ValueError("Invalid system parameters: must be dictionary")

    if not isinstance(test_params, dict):
        raise ValueError("Invalid test parameters: must be dictionary")


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
