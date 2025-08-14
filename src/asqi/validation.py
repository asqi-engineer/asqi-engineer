from typing import Dict, List

from asqi.schemas import Manifest, SuiteConfig, SUTsConfig


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
                f"Test '{test.name}': Missing required parameter '{schema_param.name}'"
            )

    # Check for unknown params
    for provided_param in test.params:
        if provided_param not in schema_params:
            errors.append(f"Test '{test.name}': Unknown parameter '{provided_param}'")

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
            errors.append(f"Test '{test.name}': Unknown SUT '{sut_name}'")
            continue

        sut_def = sut_definitions[sut_name]
        if sut_def.type not in supported_sut_types:
            errors.append(
                f"Test '{test.name}' on SUT '{sut_name}': "
                f"Image '{test.image}' does not support SUT type '{sut_def.type}'"
            )

    return errors


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
        if test.image not in manifests:
            errors.append(
                f"Test '{test.name}': No manifest available for image '{test.image}'"
            )
            continue

        manifest = manifests[test.image]

        # Validate test parameters
        param_errors = validate_test_parameters(test, manifest)
        errors.extend(param_errors)

        # Validate SUT compatibility
        sut_errors = validate_sut_compatibility(test, sut_definitions, manifest)
        errors.extend(sut_errors)

    return errors


def create_test_execution_plan(
    suite: SuiteConfig, suts: SUTsConfig, image_availability: Dict[str, bool]
) -> List[Dict]:
    """
    Create execution plan for all valid test combinations.

    Args:
        suite: Test suite configuration
        suts: SUTs configuration
        image_availability: Dictionary of image availability status

    Returns:
        List of test execution plans
    """
    execution_plan = []

    for test_def in suite.test_suite:
        # Skip if image not available
        if not image_availability.get(test_def.image, False):
            continue

        for sut_name in test_def.target_suts:
            sut_definition = suts.systems_under_test[sut_name]
            sut_config = {"type": sut_definition.type, **sut_definition.config}

            execution_plan.append(
                {
                    "test_name": test_def.name,
                    "image": test_def.image,
                    "sut_name": sut_name,
                    "sut_config": sut_config,
                    "test_params": test_def.params,
                }
            )

    return execution_plan


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
        if test.image not in manifests:
            errors.append(
                f"Test '{test.name}': Image '{test.image}' does not have a loaded manifest."
            )
            continue  # Cannot perform further validation for this test

        manifest = manifests[test.image]
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
