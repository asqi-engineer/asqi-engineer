import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from asqi.schemas import Manifest, SuiteConfig, SystemsConfig

logger = logging.getLogger()


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


@dataclass
class ManifestMatchResult:
    """Result of attempting to resolve an image to a manifest."""

    manifest: Optional[Manifest]
    matched_by: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)


@dataclass
class _Candidate:
    manifest: Manifest
    display: str
    priority: int


def _normalize_token(value: str) -> str:
    return value.strip().lower() if value else ""


def _strip_digest(reference: str) -> str:
    return reference.split("@", 1)[0].strip() if reference else ""


def _split_repo_and_tag(reference: str) -> tuple[str, Optional[str]]:
    cleaned = _strip_digest(reference)
    if not cleaned:
        return "", None

    repo = cleaned
    tag: Optional[str] = None

    if "/" in cleaned:
        head, tail = cleaned.rsplit("/", 1)
        if ":" in tail:
            name, tag = tail.rsplit(":", 1)
            repo = f"{head}/{name}" if head else name
    else:
        if ":" in cleaned:
            repo, tag = cleaned.rsplit(":", 1)

    return repo, tag


def _strip_registry(repo: str) -> str:
    if not repo:
        return ""
    if "/" not in repo:
        return repo
    first, remainder = repo.split("/", 1)
    if "." in first or ":" in first or first == "localhost":
        return remainder
    return repo


def _basename(repo: str) -> str:
    if not repo:
        return ""
    if "/" not in repo:
        return repo
    return repo.rsplit("/", 1)[-1]


def _deduplicate_preserve(items: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _exact_variations(reference: str) -> List[str]:
    base = reference.strip() if reference else ""
    no_digest = _strip_digest(base)
    return _deduplicate_preserve([base, no_digest])


def _without_tag_variations(reference: str) -> List[str]:
    base = _strip_digest(reference)
    repo, tag = _split_repo_and_tag(base)
    if not repo or tag is None:
        return []
    return _deduplicate_preserve([repo])


def _without_registry_variations(reference: str) -> List[str]:
    base = _strip_digest(reference)
    repo, tag = _split_repo_and_tag(base)
    stripped = _strip_registry(repo)
    if not stripped or stripped == repo:
        return []

    variations = []
    if tag:
        variations.append(f"{stripped}:{tag}")
    variations.append(stripped)
    return _deduplicate_preserve(variations)


def _basename_variations(reference: str) -> List[str]:
    base = _strip_digest(reference)
    repo, tag = _split_repo_and_tag(base)
    name = _basename(repo)
    if not name:
        return []

    variations = []
    if tag:
        variations.append(f"{name}:{tag}")
    variations.append(name)
    return _deduplicate_preserve(variations)


def _register_values(
    index: Dict[str, _Candidate],
    display_lookup: Dict[str, str],
    manifest: Manifest,
    values: List[str],
    priority: int,
) -> None:
    for value in values:
        normalized = _normalize_token(value)
        if not normalized:
            continue

        candidate = index.get(normalized)
        if candidate is None or priority < candidate.priority:
            index[normalized] = _Candidate(
                manifest=manifest, display=value, priority=priority
            )

        if normalized not in display_lookup:
            display_lookup[normalized] = value


def find_manifest_for_image(
    image_name: str, manifests: Dict[str, Manifest]
) -> ManifestMatchResult:
    """
    Find the best manifest match for a given image name.

    Supports matching across multi-level registries, manifest aliases,
    and provides suggestions when no exact match exists.

    Args:
        image_name: Docker image reference (e.g., "registry/org/image:tag")
        manifests: Dictionary of available manifests

    Returns:
        ManifestMatchResult describing the match or suggestions.
    """
    if not manifests:
        return ManifestMatchResult(manifest=None)

    candidate_index: Dict[str, _Candidate] = {}
    display_lookup: Dict[str, str] = {}

    for key, manifest in manifests.items():
        aliases = getattr(manifest, "aliases", []) or []

        _register_values(
            candidate_index, display_lookup, manifest, _exact_variations(key), 0
        )
        for alias in aliases:
            _register_values(
                candidate_index, display_lookup, manifest, _exact_variations(alias), 0
            )

        _register_values(
            candidate_index, display_lookup, manifest, _without_tag_variations(key), 1
        )
        for alias in aliases:
            _register_values(
                candidate_index,
                display_lookup,
                manifest,
                _without_tag_variations(alias),
                1,
            )

        _register_values(
            candidate_index,
            display_lookup,
            manifest,
            _without_registry_variations(key),
            2,
        )
        for alias in aliases:
            _register_values(
                candidate_index,
                display_lookup,
                manifest,
                _without_registry_variations(alias),
                2,
            )

        _register_values(
            candidate_index, display_lookup, manifest, _basename_variations(key), 3
        )
        for alias in aliases:
            _register_values(
                candidate_index,
                display_lookup,
                manifest,
                _basename_variations(alias),
                3,
            )

        if manifest.name:
            _register_values(
                candidate_index, display_lookup, manifest, [manifest.name], 4
            )

    search_values = _deduplicate_preserve(
        _exact_variations(image_name)
        + _without_tag_variations(image_name)
        + _without_registry_variations(image_name)
        + _basename_variations(image_name)
    )

    for value in search_values:
        normalized = _normalize_token(value)
        if not normalized:
            continue
        candidate = candidate_index.get(normalized)
        if candidate:
            logger.debug(
                "Resolved image '%s' to manifest '%s' via '%s'",
                image_name,
                candidate.manifest.name,
                candidate.display,
            )
            return ManifestMatchResult(
                manifest=candidate.manifest, matched_by=candidate.display
            )

    normalized_candidates = list(display_lookup.keys())
    search_key = _normalize_token(_strip_digest(image_name))
    suggestions = []
    if search_key and normalized_candidates:
        close_keys = difflib.get_close_matches(
            search_key, normalized_candidates, n=3, cutoff=0.6
        )
        suggestions = [display_lookup[key] for key in close_keys]

    logger.debug(
        "No manifest found for image '%s'. Suggestions: %s", image_name, suggestions
    )
    return ManifestMatchResult(manifest=None, suggestions=suggestions)


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
        match = find_manifest_for_image(test.image, manifests)
        manifest = match.manifest
        if manifest is None:
            available_images = ", ".join(manifests.keys()) if manifests else "none"
            message = (
                f"Test '{test.name}': No manifest available for image '{test.image}'. "
                f"Images with manifests: {available_images}"
            )
            if match.suggestions:
                message += f". Did you mean: {', '.join(match.suggestions)}?"
            errors.append(message)
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
                _params["volumes"] = (
                    vols  # Also pass volumes directly for container access
                )
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
        match = find_manifest_for_image(test.image, manifests)
        manifest = match.manifest
        if manifest is None:
            message = f"Test '{test.name}': Image '{test.image}' does not have a loaded manifest."
            if match.suggestions:
                message += f" Did you mean: {', '.join(match.suggestions)}?"
            errors.append(message)
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

    # Validate that each test has systems_under_test after defaults merging
    for test in suite.test_suite:
        if not test.systems_under_test:
            errors.append(
                f"Test '{test.name}': systems_under_test is required but not provided in test definition or test_suite_default"
            )

    # Get the system definitions
    system_definitions = systems.systems
    if not system_definitions:
        errors.append("Systems configuration is empty: no systems defined")

    # Detailed validation if manifests are provided
    if manifests is not None and not errors:  # Only if basic validation passes
        errors.extend(validate_manifests_against_tests(suite, systems, manifests))

    return errors
