import logging
from typing import Any, Dict, List, Optional

from asqi.schemas import Manifest, SuiteConfig, SUTsConfig

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


def validate_sut_compatibility(
    test, sut_definitions: Dict, manifest: Manifest
) -> List[str]:
    """
    Validate SUT compatibility with test container.

    Args:
        test: Test definition from suite config
        sut_definitions: Dictionary of SUT definitions
        manifest: Manifest for the test container

    Returns:
        List of validation error messages
    """
    errors = []
    supported_sut_types = [s.type for s in manifest.supported_suts]

    for sut_name in test.target_suts:
        if sut_name not in sut_definitions:
            available_suts = (
                ", ".join(sut_definitions.keys()) if sut_definitions else "none"
            )
            errors.append(
                f"Test '{test.name}': Unknown SUT '{sut_name}'. Available SUTs: {available_suts}"
            )
            continue

        sut_def = sut_definitions[sut_name]
        if sut_def.type not in supported_sut_types:
            supported_types = (
                ", ".join(supported_sut_types) if supported_sut_types else "none"
            )
            errors.append(
                f"Test '{test.name}' on SUT '{sut_name}': "
                f"Image '{test.image}' does not support SUT type '{sut_def.type}'. Supported types: {supported_types}"
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
    suite: SuiteConfig, suts: SUTsConfig, manifests: Dict[str, Manifest]
) -> List[str]:
    """
    Validate that all tests can be executed with available manifests.

    Args:
        suite: Test suite configuration
        suts: SUTs configuration
        manifests: Dictionary of available manifests

    Returns:
        List of validation error messages
    """
    errors = []
    sut_definitions = suts.systems_under_test

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

        # Validate SUT compatibility
        sut_errors = validate_sut_compatibility(test, sut_definitions, manifest)
        errors.extend(sut_errors)

    return errors


def create_test_execution_plan(
    suite: SuiteConfig, suts: SUTsConfig, image_availability: Dict[str, bool]
) -> List[Dict[str, Any]]:
    """
    Create execution plan for all valid test combinations.

    Args:
        suite: Test suite configuration
        suts: SUTs configuration
        image_availability: Dictionary of image availability status

    Returns:
        List of test execution plans
    """
    if not suite or not suite.test_suite:
        return []
    if not suts or not suts.systems_under_test:
        return []

    plan: List[Dict[str, Any]] = []
    available_images = {img for img, ok in image_availability.items() if ok}

    for test in suite.test_suite:
        if not (image := getattr(test, "image", None)):
            logger.warning(f"Skipping test with missing image: {test}")
            continue

        if image not in available_images:
            continue

        if not (target_suts := getattr(test, "target_suts", None)):
            logger.warning(f"Skipping test '{test.name}' with no target SUTs")
            continue

        # Process valid combinations
        for sut_name in target_suts:
            sut_def = suts.systems_under_test.get(sut_name)
            if not sut_def or not getattr(sut_def, "type", None):
                continue

            plan.append(
                {
                    "test_name": test.name,
                    "image": image,
                    "sut_name": sut_name,
                    "sut_params": {"type": sut_def.type, **sut_def.params},
                    "test_params": getattr(test, "params", {}),
                }
            )

    return plan


def validate_test_plan(
    suite: SuiteConfig, suts: SUTsConfig, manifests: Dict[str, Manifest]
) -> List[str]:
    """
    Validates the entire test plan by cross-referencing the suite, SUTs, and manifests.

    Args:
        suite: The parsed SuiteConfig object.
        suts: The parsed SUTsConfig object.
        manifests: A dictionary mapping image names to their parsed Manifest objects.

    Returns:
        A list of error strings. An empty list indicates successful validation.
    """
    errors = []
    sut_definitions = suts.systems_under_test

    for test in suite.test_suite:
        # 1. Check if the test's image has a corresponding manifest
        manifest = find_manifest_for_image(test.image, manifests)
        if manifest is None:
            errors.append(
                f"Test '{test.name}': Image '{test.image}' does not have a loaded manifest."
            )
            continue  # Cannot perform further validation for this test
        supported_sut_types = [s.type for s in manifest.supported_suts]

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

        # 3. For each target SUT, perform validation
        for sut_name in test.target_suts:
            # 3a. Check if the SUT exists in the SUTs config
            if sut_name not in sut_definitions:
                errors.append(
                    f"Test '{test.name}': Target SUT '{sut_name}' is not defined in the SUTs file."
                )
                continue  # Cannot perform further validation for this SUT

            sut_def = sut_definitions[sut_name]

            # 3b. Check if the container supports the SUT's type
            if sut_def.type not in supported_sut_types:
                errors.append(
                    f"Test '{test.name}' on SUT '{sut_name}': Image '{test.image}' "
                    f"(supports: {supported_sut_types}) is not compatible with SUT type '{sut_def.type}'."
                )

    return errors


def validate_execution_inputs(
    suite_path: str,
    suts_path: str,
    execution_mode: str,
    output_path: Optional[str] = None,
) -> None:
    """
    Validate inputs for test execution workflows.

    Args:
        suite_path: Path to test suite YAML file
        suts_path: Path to SUTs YAML file
        execution_mode: Execution mode string
        output_path: Optional output file path

    Raises:
        ValueError: If any input is invalid
    """
    if not suite_path or not isinstance(suite_path, str):
        raise ValueError("Invalid suite_path: must be non-empty string")

    if not suts_path or not isinstance(suts_path, str):
        raise ValueError("Invalid suts_path: must be non-empty string")

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
    sut_name: str,
    sut_params: Dict[str, Any],
    test_params: Dict[str, Any],
) -> None:
    """
    Validate inputs for individual test execution.

    Args:
        test_name: Name of the test
        image: Docker image name
        sut_name: Name of the SUT
        sut_params: SUT parameters dictionary (flattened configuration)
        test_params: Test parameters dictionary

    Raises:
        ValueError: If any input is invalid
    """
    if not test_name or not isinstance(test_name, str):
        raise ValueError("Invalid test name: must be non-empty string")

    if not image or not isinstance(image, str):
        raise ValueError("Invalid image: must be non-empty string")

    if not sut_name or not isinstance(sut_name, str):
        raise ValueError("Invalid SUT name: must be non-empty string")

    if not isinstance(sut_params, dict):
        raise ValueError("Invalid SUT parameters: must be dictionary")

    if not isinstance(test_params, dict):
        raise ValueError("Invalid test parameters: must be dictionary")


def validate_workflow_configurations(
    suite: SuiteConfig,
    suts: SUTsConfig,
    manifests: Optional[Dict[str, Manifest]] = None,
) -> List[str]:
    """
    Comprehensive validation of workflow configurations.

    Combines all configuration validation checks in one place.

    Args:
        suite: Test suite configuration
        suts: SUTs configuration
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

    if not isinstance(suts, SUTsConfig):
        raise ValueError("Invalid suts: must be SUTsConfig instance")

    if manifests is not None and not isinstance(manifests, dict):
        raise ValueError("Invalid manifests: must be dictionary")

    # Content validation
    if not suite.test_suite:
        errors.append("Test suite is empty: no tests to validate")

    if not suts.systems_under_test:
        errors.append("SUTs configuration is empty: no systems under test defined")

    # Detailed validation if manifests are provided
    if manifests is not None and not errors:  # Only if basic validation passes
        errors.extend(validate_manifests_against_tests(suite, suts, manifests))

    return errors
