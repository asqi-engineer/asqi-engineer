import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from asqi.errors import DuplicateIDError, MissingIDFieldError
from asqi.main import load_and_validate_plan
from asqi.rag_response_schema import RAGCitation, RAGContext, validate_rag_response
from asqi.schemas import (
    AssessmentRule,
    GenericSystemConfig,
    LLMAPIConfig,
    LLMAPIParams,
    Manifest,
    RAGAPIConfig,
    ScoreCard,
    ScoreCardFilter,
    ScoreCardIndicator,
    SuiteConfig,
    SystemsConfig,
)
from asqi.score_card_engine import ScoreCardEngine
from asqi.validation import (
    create_test_execution_plan,
    find_manifest_for_image,
    validate_execution_inputs,
    validate_ids,
    validate_manifests_against_tests,
    validate_score_card_inputs,
    validate_system_compatibility,
    validate_test_execution_inputs,
    validate_test_parameters,
    validate_test_plan,
    validate_test_volumes,
    validate_workflow_configurations,
)
from asqi.workflow import TestExecutionResult
from test_data import (
    MOCK_SCORE_CARD_CONFIG,
    MOCK_AUDIT_RESPONSES,
)

# Test data
DEMO_SUITE_YAML = """
suite_name: "Mock Tester Sanity Check"
description: "Suite description"
test_suite:
  - name: "run_mock_on_compatible_system"
    id: "run_mock_on_compatible_system"
    description: "Test description"
    image: "my-registry/mock_tester:latest"
    systems_under_test:
      - "my_llm_api" 
    params:
      delay_seconds: 1
"""

DEMO_RAG_SUITE_YAML = """   
suite_name: "Mock RAG Tester Sanity Check"
description: "Suite description"
test_suite:
    - name: "run_mock_on_compatible_system"
      id: "run_mock_on_compatible_system"
      description: "Test description"
      image: "my-registry/mock_rag_tester:latest"
      systems_under_test:
          - "my_rag_api"
      params:
          delay_seconds: 1
"""

DEMO_systems_YAML = """
systems:
  # This system is compatible with our mock_tester
  my_llm_service:
    type: "llm_api"
    provider: "some_provider"
    description: "Some Description"
    params:
      base_url: "http://URL"
      model: "model-x"
      env_file: "MY_ENV_FILE"
      api_key: "MY_LLM_API_KEY"
  # This system is compatible with our mock_tester, for testing multiple systems
  another_llm_service:
    type: "llm_api"
    provider: "some_provider"
    description: "Some Description"
    params:
      base_url: "http://URL"
      model: "model-y"
      env_file: "MY_ENV_FILE"
      api_key: "MY_LLM_API_KEY"
"""

MOCK_TESTER_MANIFEST = {
    "name": "mock_tester",
    "version": "1.0.0",
    "description": "A minimal mock container for testing the executor interface.",
    "input_systems": [
        {"name": "system_under_test", "type": "llm_api", "required": True},
    ],
    "input_schema": [
        {
            "name": "delay_seconds",
            "type": "integer",
            "required": False,
            "description": "Seconds to sleep to simulate work.",
        }
    ],
    "output_metrics": ["success", "score", "delay_used"],
}

MOCK_RAG_TESTER_MANIFEST = {
    "name": "mock_rag_tester",
    "version": "1.0.0",
    "description": "A mock container for testing RAG systems with context validation.",
    "input_systems": [
        {"name": "system_under_test", "type": "rag_api", "required": True},
    ],
    "input_schema": [
        {
            "name": "delay_seconds",
            "type": "integer",
            "required": False,
            "description": "Seconds to sleep to simulate work.",
        }
    ],
    "output_metrics": ["success", "score", "delay_used", "base_url", "model", "user_group"],
}

MOCK_GENERIC_MANIFEST = {
    "name": "generic",
    "version": "0.1.0",
    "description": "A minimal mock container for testing a generic systems.",
    "input_systems": [
        {"name": "system_under_test", "type": "new_system", "required": True},
    ],
}


MOCK_MULTIPLE_MANIFEST = {
    "name": "garak",
    "version": "0.2.0",
    "description": "A security and safety probing tool for Large Language Models.",
    "input_systems": [
        {"name": "system_under_test", "type": "llm_api", "required": True},
        {"name": "system_under_test", "type": "rest_api", "required": True},
    ],
    "input_schema": [
        {
            "name": "probes",
            "type": "list",
            "required": True,
            "description": "List of garak probe modules to run.",
        }
    ],
    "output_metrics": ["status", "probes_run", "total_failed"],
}


@pytest.fixture
def demo_suite():
    """Fixture providing parsed demo test suite."""
    data = yaml.safe_load(DEMO_SUITE_YAML)
    return SuiteConfig(**data)

@pytest.fixture
def demo_rag_suite():
    """Fixture providing parsed demo test suite."""
    data = yaml.safe_load(DEMO_RAG_SUITE_YAML)
    return SuiteConfig(**data)


@pytest.fixture
def demo_systems():
    """Fixture providing parsed demo systems."""
    data = yaml.safe_load(DEMO_systems_YAML)
    return SystemsConfig(**data)


@pytest.fixture
def manifests():
    """Fixture providing test manifests."""
    return {
        "my-registry/mock_tester:latest": Manifest(**MOCK_TESTER_MANIFEST),
        "my-registry/mock_rag_tester:latest": Manifest(**MOCK_RAG_TESTER_MANIFEST),
        "my-registry/generic:latest": Manifest(**MOCK_GENERIC_MANIFEST),
        "my-registry/garak:latest": Manifest(**MOCK_MULTIPLE_MANIFEST),
    }


class TestSchemaValidation:
    """Test that YAML files parse correctly into Pydantic schemas."""

    def test_suite_schema_validation(self, demo_suite):
        """Test that demo suite YAML parses correctly."""
        assert demo_suite.suite_name == "Mock Tester Sanity Check"
        assert len(demo_suite.test_suite) == 1

        # Check first test
        test1 = demo_suite.test_suite[0]
        assert test1.name == "run_mock_on_compatible_system"
        assert test1.image == "my-registry/mock_tester:latest"
        assert test1.systems_under_test == ["my_llm_api"]
        assert test1.params["delay_seconds"] == 1

    def test_systems_schema_llm_validation(self, demo_systems):
        """Test that the LLMs systems YAML parses correctly."""
        systems = demo_systems.systems
        assert len(systems) == 2

        # Check LLM service
        llm_system = systems["my_llm_service"]
        assert llm_system.type == "llm_api"
        assert llm_system.params.base_url == "http://URL"
        assert llm_system.params.model == "model-x"
        assert llm_system.params.env_file == "MY_ENV_FILE"
        assert llm_system.params.api_key == "MY_LLM_API_KEY"

        llm_another_system = systems["another_llm_service"]
        assert llm_another_system.type == "llm_api"
        assert llm_another_system.params.base_url == "http://URL"
        assert llm_another_system.params.model == "model-y"
        assert llm_another_system.params.env_file == "MY_ENV_FILE"
        assert llm_another_system.params.api_key == "MY_LLM_API_KEY"

    def test_generic_systems_schema(self, manifests):
        """Test a generic system used for system types that don't have their own config classes."""

        # This system type is not yet implemented. It’s just to check backward compatibility
        system = SystemsConfig(
            systems={
                "new_system": GenericSystemConfig(
                    type="new_system",
                    description="New System description",
                    provider="openai",
                    params={
                        "random_param": "aexcea",
                        "base_url": "http://URL",
                        "model": "y-model",
                    },
                )
            }
        )

        new_system = system.systems["new_system"]
        assert new_system.type == "new_system"
        assert new_system.description == "New System description"
        assert new_system.params["random_param"] == "aexcea"
        assert new_system.params["base_url"] == "http://URL"
        assert new_system.params["model"] == "y-model"

        suite_data = {
            "suite_name": "Compatible Test",
            "description": "Suite description",
            "test_suite": [
                {
                    "name": "test_llm_service",
                    "id": "test_llm_service",
                    "description": "Test description",
                    "image": "my-registry/generic:latest",
                    "systems_under_test": ["new_system"],
                }
            ],
        }
        errors = validate_test_plan(SuiteConfig(**suite_data), system, manifests)
        assert errors == [], f"Expected no errors, but got: {errors}"

    def test_missing_params_llm_api_systems_schema(self):
        """Test that validates required LLM API system parameters."""

        # This system base_url param is missing
        with pytest.raises(ValidationError, match="missing"):
            SystemsConfig(
                systems={
                    "test_system": LLMAPIConfig(
                        type="llm_api",
                        description="System description",
                        provider="openai",
                        params=LLMAPIParams(
                            env_file="ENV_FILE",
                            model="gpt-4o-mini",
                            api_key="sk-123",
                        ),  # type: ignore base_url missing
                    )
                }
            )
        # This system model param is missing
        with pytest.raises(ValidationError, match="missing"):
            SystemsConfig(
                systems={
                    "test_system": LLMAPIConfig(
                        type="llm_api",
                        description="System description",
                        provider="openai",
                        params=LLMAPIParams(
                            base_url="http://URL",
                            api_key="sk-123",
                        ),  # type: ignore model missing
                    )
                }
            )

    def test_optional_params_llm_api_systems_schema(self, demo_suite, manifests):
        """Test that validates optional LLM API system parameters."""

        system = SystemsConfig(
            systems={
                "my_llm_api": LLMAPIConfig(
                    type="llm_api",
                    description="System description",
                    provider="openai",
                    params=LLMAPIParams(
                        base_url="http://URL",
                        model="x-model",
                    ),  # type: ignore optional params
                )
            }
        )

        errors = validate_test_plan(demo_suite, system, manifests)
        assert errors == [], f"Expected no errors, but got: {errors}"

    def test_missing_params_rag_api_systems_schema(self):
        """Test that validates required RAG API system parameters."""

        # This system base_url param is missing
        with pytest.raises(ValidationError, match="missing"):
            SystemsConfig(
                systems={
                    "test_rag_system": RAGAPIConfig(
                        type="rag_api",
                        description="RAG System description",
                        provider="custom",
                        params=LLMAPIParams(
                            env_file="ENV_FILE",
                            model="rag-model",
                            api_key="sk-123",
                        ),  # type: ignore base_url missing
                    )
                }
            )
        # This system model param is missing
        with pytest.raises(ValidationError, match="missing"):
            SystemsConfig(
                systems={
                    "test_rag_system": RAGAPIConfig(
                        type="rag_api",
                        description="RAG System description",
                        provider="custom",
                        params=LLMAPIParams(
                            base_url="http://URL",
                            api_key="sk-123",
                        ),  # type: ignore model missing
                    )
                }
            )

    def test_optional_params_rag_api_systems_schema(self, demo_rag_suite, manifests):
        """Test that validates optional RAG API system parameters."""

        system = SystemsConfig(
            systems={
                "my_rag_api": RAGAPIConfig(
                    type="rag_api",
                    description="RAG System description",
                    provider="custom",
                    params=LLMAPIParams(
                        base_url="http://URL",
                        model="rag-model",
                        user_group="admin",  # optional param
                    ),  # type: ignore optional params
                )
            }
        )

        errors = validate_test_plan(demo_rag_suite, system, manifests)
        assert errors == [], f"Expected no errors, but got: {errors}"

    def test_manifest_schema_validation(self, manifests):
        """Test that manifests parse correctly."""
        mock_manifest = manifests["my-registry/mock_tester:latest"]
        assert mock_manifest.name == "mock_tester"
        assert len(mock_manifest.input_systems) == 1
        assert mock_manifest.input_systems[0].type == "llm_api"

        rag_manifest = manifests["my-registry/mock_rag_tester:latest"]
        assert rag_manifest.name == "mock_rag_tester"
        assert len(rag_manifest.input_systems) == 1
        assert rag_manifest.input_systems[0].type == "rag_api"


class TestCrossFileValidation:
    """Test validation logic that checks consistency across files."""

    def test_successful_validation(self, demo_systems, manifests):
        """Test validation passes for compatible systems."""
        # Create a suite with only compatible systems
        compatible_suite_data = {
            "suite_name": "Compatible Test",
            "description": "Suite description",
            "test_suite": [
                {
                    "name": "test_llm_service",
                    "id": "test_llm_service",
                    "description": "Test description",
                    "image": "my-registry/mock_tester:latest",
                    "systems_under_test": ["my_llm_service"],
                    "params": {"delay_seconds": 1},
                }
            ],
        }
        suite = SuiteConfig(**compatible_suite_data)

        errors = validate_test_plan(suite, demo_systems, manifests)
        assert errors == [], f"Expected no errors, but got: {errors}"

    def test_missing_image_manifest(self, demo_suite, demo_systems):
        """Test validation fails when manifest is missing for an image."""
        # Empty manifests dict
        empty_manifests = {}

        errors = validate_test_plan(demo_suite, demo_systems, empty_manifests)
        assert len(errors) > 0
        assert any("does not have a loaded manifest" in error for error in errors)

    def test_missing_system_definition(self, demo_systems, manifests):
        """Test validation fails when system is not defined."""
        suite_data = {
            "suite_name": "Test with Missing system",
            "description": "Suite Description",
            "test_suite": [
                {
                    "name": "test_missing_system",
                    "id": "test_missing_system",
                    "description": "Test Description",
                    "image": "my-registry/mock_tester:latest",
                    "systems_under_test": ["nonexistent_system"],
                    "params": {},
                }
            ],
        }
        suite = SuiteConfig(**suite_data)

        errors = validate_test_plan(suite, demo_systems, manifests)
        assert len(errors) > 0
        assert any("is not defined in the systems file" in error for error in errors)

    def test_missing_required_parameter(self, demo_systems, manifests):
        """Test validation fails when required parameters are missing."""
        # Use garak which has required parameters
        suite_data = {
            "suite_name": "Test Missing Required Param",
            "description": "Suite Description",
            "test_suite": [
                {
                    "name": "garak_without_probes",
                    "id": "garak_without_probes",
                    "description": "Test Description",
                    "image": "my-registry/garak:latest",
                    "systems_under_test": ["my_llm_service"],
                    "params": {},  # Missing required 'probes' parameter
                }
            ],
        }
        suite = SuiteConfig(**suite_data)

        errors = validate_test_plan(suite, demo_systems, manifests)
        assert len(errors) > 0
        assert any(
            "Required parameter 'probes' is missing" in error for error in errors
        )

    def test_unknown_parameter(self, demo_systems, manifests):
        """Test validation fails when unknown parameters are provided."""
        suite_data = {
            "suite_name": "Test Unknown Param",
            "description": "Suite Description",
            "test_suite": [
                {
                    "name": "test_unknown_param",
                    "id": "test_unknown_param",
                    "description": "Test Description",
                    "image": "my-registry/mock_tester:latest",
                    "systems_under_test": ["my_llm_service"],
                    "params": {"delay_seconds": 1, "unknown_param": "should_fail"},
                }
            ],
        }
        suite = SuiteConfig(**suite_data)

        errors = validate_test_plan(suite, demo_systems, manifests)
        assert len(errors) > 0
        assert any("Unknown parameter 'unknown_param'" in error for error in errors)

    def test_optional_suites_description_fields(self, demo_systems, manifests):
        """Test validation optional description fields."""
        suite_data = {
            "suite_name": "Advanced Chatbot Testing",
            "test_suite": [
                {
                    "name": "chatbot_simulation",
                    "id": "chatbot_simulation",
                    "image": "my-registry/mock_tester:latest",
                    "systems_under_test": ["my_llm_service"],
                    "params": {"delay_seconds": 1},
                }
            ],
        }

        suite = SuiteConfig(**suite_data)

        errors = validate_test_plan(suite, demo_systems, manifests)
        assert errors == [], f"Expected no errors, but got: {errors}"

    def test_invalid_suite_name_description(self):
        """Test validation fails when suite name description is not a string."""
        suite_data = {
            "suite_name": "Test Invalid Suite Name Description",
            "description": 33,
            "test_suite": [
                {
                    "name": "test_demo",
                    "id": "test_demo",
                    "description": "Test Description",
                    "image": "my-registry/mock_tester:latest",
                    "systems_under_test": ["my_llm_service"],
                    "params": {
                        "delay_seconds": 1,
                    },
                }
            ],
        }
        with pytest.raises(ValidationError):
            SuiteConfig(**suite_data)

    def test_invalid_test_description(self):
        """Test validation fails when test description is not a string."""
        suite_data = {
            "suite_name": "Test Invalid Test Description",
            "description": "Suite Description",
            "test_suite": [
                {
                    "name": "test_demo",
                    "id": "test_demo",
                    "description": 33,
                    "image": "my-registry/mock_tester:latest",
                    "systems_under_test": ["my_llm_service"],
                    "params": {"delay_seconds": 1},
                }
            ],
        }
        with pytest.raises(ValidationError):
            SuiteConfig(**suite_data)


class TestFileLoading:
    """Test the file loading functionality from main.py."""

    def test_load_yaml_files_from_disk(self):
        """Test loading actual YAML files from temporary disk files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write test files
            suite_file = temp_path / "demo_suite.yaml"
            systems_file = temp_path / "demo_systems.yaml"
            manifest_dir = temp_path / "manifests"
            manifest_dir.mkdir()

            # Write YAML files
            with open(suite_file, "w") as f:
                f.write(DEMO_SUITE_YAML)

            with open(systems_file, "w") as f:
                f.write(DEMO_systems_YAML)

            # Write manifest file
            manifest_file = manifest_dir / "manifest.yaml"
            with open(manifest_file, "w") as f:
                yaml.dump(MOCK_TESTER_MANIFEST, f)

            # Test that we can load and validate

            result = load_and_validate_plan(
                str(suite_file), str(systems_file), str(manifest_dir)
            )

            # Should have some validation errors due to incompatible system
            # (my_backend_api is not supported by mock_tester in this setup)
            assert result["status"] == "failure"
            assert len(result["errors"]) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_test_suite(self, demo_systems, manifests):
        """Test validation with empty test suite."""
        empty_suite_data = {"suite_name": "Empty Suite", "test_suite": []}
        suite = SuiteConfig(**empty_suite_data)

        errors = validate_test_plan(suite, demo_systems, manifests)
        assert errors == []  # Empty suite should be valid

    def test_multiple_systems_under_test(self, demo_systems, manifests):
        """Test validation with multiple target systems."""
        multi_system_data = {
            "suite_name": "Multi system Test",
            "description": "Suite Description",
            "test_suite": [
                {
                    "name": "test_multiple_systems",
                    "id": "test_multiple_systems",
                    "description": "Test Description",
                    "image": "my-registry/garak:latest",
                    "systems_under_test": ["my_llm_service", "another_llm_service"],
                    "params": {
                        "probes": ["probe1", "probe2"]
                    },  # Provide required param for garak
                }
            ],
        }
        suite = SuiteConfig(**multi_system_data)

        errors = validate_test_plan(suite, demo_systems, manifests)
        # Should pass since garak supports both llm_api and rest_api
        assert errors == []

    def test_no_parameters(self, demo_systems, manifests):
        """Test validation with no parameters (should be fine for mock_tester)."""
        no_params_data = {
            "suite_name": "No Params Test",
            "description": "No Param Description",
            "test_suite": [
                {
                    "name": "test_no_params",
                    "id": "test_no_params",
                    "description": "No Param Description",
                    "image": "my-registry/mock_tester:latest",
                    "systems_under_test": ["my_llm_service"],
                    "params": {},
                }
            ],
        }
        suite = SuiteConfig(**no_params_data)

        errors = validate_test_plan(suite, demo_systems, manifests)
        assert errors == []  # Should be valid since delay_seconds is optional


class TestValidationFunctions:
    def test_validate_test_parameters(self, manifests):
        manifest = manifests["my-registry/mock_tester:latest"]

        # Test with missing required param (none required)
        class DummyTest:
            name = "t1"
            params = {}

        test = DummyTest()
        errors = validate_test_parameters(test, manifest)
        assert errors == []

        # Test with unknown param
        test.params = {"foo": 1}
        errors = validate_test_parameters(test, manifest)
        assert any("Unknown parameter 'foo'" in e for e in errors)

        # Test with required param (garak)
        garak_manifest = manifests["my-registry/garak:latest"]
        test2 = DummyTest()
        test2.name = "t2"
        test2.params = {}
        errors = validate_test_parameters(test2, garak_manifest)
        assert any("Missing required parameter 'probes'" in e for e in errors)
        test2.params = {"probes": ["p1"]}
        errors = validate_test_parameters(test2, garak_manifest)
        assert errors == []

    def test_validate_system_compatibility(self, demo_systems, manifests):
        manifest = manifests["my-registry/mock_tester:latest"]

        class DummyTest:
            name = "t1"
            image = "my-registry/mock_tester:latest"
            systems_under_test = ["my_llm_service", "my_backend_api"]

        test = DummyTest()
        errors = validate_system_compatibility(test, demo_systems.systems, manifest)

        # Check that there's no error for the supported llm_api system type
        assert not any("does not support system type 'llm_api'" in e for e in errors)

        # Unknown system
        test.systems_under_test = ["not_a_system"]
        errors = validate_system_compatibility(test, demo_systems.systems, manifest)
        assert any("Unknown system 'not_a_system'" in e for e in errors)

    def test_validate_system_compatibility_with_additional_systems(self, demo_systems):
        """Test validation of additional systems from test.systems field."""
        # Create a manifest with optional systems
        manifest_data = {
            "name": "multi_system_tester",
            "version": "1.0.0",
            "description": "A test container with multiple system support",
            "input_systems": [
                {"name": "system_under_test", "type": "llm_api", "required": True},
                {"name": "simulator_system", "type": "llm_api", "required": False},
                {"name": "evaluator_system", "type": "llm_api", "required": False},
            ],
            "input_schema": [],
            "output_metrics": ["success"],
        }
        manifest = Manifest(**manifest_data)

        class DummyTest:
            name = "multi_test"
            image = "multi-system:latest"
            systems_under_test = ["my_llm_service"]
            systems = {
                "simulator_system": "my_llm_service",
                "evaluator_system": "my_llm_service",
            }

        test = DummyTest()
        errors = validate_system_compatibility(test, demo_systems.systems, manifest)
        assert errors == []  # Should pass since all systems are llm_api

        test.systems = {"unknown_role": "my_llm_service"}
        errors = validate_system_compatibility(test, demo_systems.systems, manifest)
        assert any("Unknown system role 'unknown_role'" in e for e in errors)

        test.systems = {"simulator_system": "unknown_system"}
        errors = validate_system_compatibility(test, demo_systems.systems, manifest)
        assert any("Unknown simulator_system 'unknown_system'" in e for e in errors)

        test.systems = {"simulator_system": "another_llm_service"}
        errors = validate_system_compatibility(test, demo_systems.systems, manifest)
        assert errors == []

    def test_validate_manifests_against_tests(self, demo_systems, manifests):
        suite_data = {
            "suite_name": "Valid",
            "description": "Suite Description",
            "test_suite": [
                {
                    "name": "t1",
                    "id": "t1",
                    "description": "Test Description",
                    "image": "my-registry/mock_tester:latest",
                    "systems_under_test": ["my_llm_service"],
                    "params": {},
                }
            ],
        }
        suite = SuiteConfig(**suite_data)
        errors = validate_manifests_against_tests(suite, demo_systems, manifests)
        assert errors == []

        # Missing manifest
        suite_data["test_suite"][0]["image"] = "notfound:latest"
        suite = SuiteConfig(**suite_data)
        errors = validate_manifests_against_tests(suite, demo_systems, manifests)
        assert any("No manifest available for image" in e for e in errors)

    def test_create_test_execution_plan(self, demo_systems, manifests):
        suite_data = {
            "suite_name": "ExecPlan",
            "description": "Suite Description",
            "test_suite": [
                {
                    "name": "t1",
                    "id": "t1",
                    "description": "T1 Description",
                    "image": "my-registry/mock_tester:latest",
                    "systems_under_test": ["my_llm_service"],
                    "params": {"delay_seconds": 1},
                },
                {
                    "name": "t2",
                    "id": "t2",
                    "description": "T2 Description",
                    "image": "my-registry/mock_tester:latest",
                    "systems_under_test": ["my_llm_service", "another_llm_service"],
                    "params": {},
                },
            ],
        }
        suite = SuiteConfig(**suite_data)
        image_availability = {"my-registry/mock_tester:latest": True}
        plan = create_test_execution_plan(suite, demo_systems, image_availability)
        # Should create 3 plans (1 + 2)
        assert len(plan) == 3
        names = [p["test_name"] for p in plan]
        assert names.count("t1") == 1
        assert names.count("t2") == 2

        # Check that different systems are included
        system_names = [p["sut_name"] for p in plan]
        assert "my_llm_service" in system_names
        assert "another_llm_service" in system_names

        # If image not available, plan is empty
        image_availability = {"my-registry/mock_tester:latest": False}
        plan = create_test_execution_plan(suite, demo_systems, image_availability)
        assert plan == []


class TestFindManifestForImage:
    """Test the find_manifest_for_image function with various image name patterns."""

    def test_exact_match(self):
        """Test exact image name match."""
        manifests = {"my-registry/mock_tester:latest": Manifest(**MOCK_TESTER_MANIFEST)}
        result = find_manifest_for_image("my-registry/mock_tester:latest", manifests)
        assert result is not None
        assert result.name == "mock_tester"

    def test_container_name_match(self):
        """Test matching by container name when full image not found."""
        manifests = {"mock_tester": Manifest(**MOCK_TESTER_MANIFEST)}
        result = find_manifest_for_image("my-registry/mock_tester:latest", manifests)
        assert result is not None
        assert result.name == "mock_tester"

    def test_base_name_match(self):
        """Test matching by base name without registry/tag."""
        manifests = {"mock_tester": Manifest(**MOCK_TESTER_MANIFEST)}
        result = find_manifest_for_image("mock_tester:v1.0", manifests)
        assert result is not None
        assert result.name == "mock_tester"

    def test_no_match(self):
        """Test when no manifest is found."""
        manifests = {"other_image": Manifest(**MOCK_TESTER_MANIFEST)}
        result = find_manifest_for_image("unknown_image:latest", manifests)
        assert result is None

    def test_image_without_slash(self):
        """Test image name without registry."""
        manifests = {"mock_tester": Manifest(**MOCK_TESTER_MANIFEST)}
        result = find_manifest_for_image("mock_tester:latest", manifests)
        assert result is not None
        assert result.name == "mock_tester"


class TestValidationInputFunctions:
    """Test the input validation functions."""

    def test_validate_execution_inputs_valid(self):
        """Test valid execution inputs."""
        validate_execution_inputs(
            suite_path="suite.yaml",
            systems_path="systems.yaml",
            execution_mode="tests_only",
            output_path="output.json",
        )
        validate_execution_inputs(
            suite_path="suite.yaml",
            systems_path="systems.yaml",
            execution_mode="end_to_end",
            output_path=None,
        )

    def test_validate_execution_inputs_invalid(self):
        """Test invalid execution inputs."""
        # Invalid suite_path - empty string
        with pytest.raises(ValueError, match="Invalid suite_path"):
            validate_execution_inputs("", "systems.yaml", "tests_only")

        # Invalid systems_path - empty string
        with pytest.raises(ValueError, match="Invalid systems_path"):
            validate_execution_inputs("suite.yaml", "", "tests_only")

        # Invalid execution_mode
        with pytest.raises(ValueError, match="Invalid execution_mode"):
            validate_execution_inputs("suite.yaml", "systems.yaml", "invalid_mode")

    def test_validate_score_card_inputs_valid(self):
        """Test valid score card inputs."""
        score_card_configs = [{"indicator": "test"}]
        validate_score_card_inputs(
            input_path="input.json",
            score_card_configs=score_card_configs,
            output_path="output.json",
        )
        validate_score_card_inputs(
            input_path="input.json",
            score_card_configs=score_card_configs,
            output_path=None,
        )

    def test_validate_score_card_inputs_invalid(self):
        """Test invalid score card inputs."""
        score_card_configs = [{"indicator": "test"}]
        # Invalid input_path - empty string
        with pytest.raises(ValueError, match="Invalid input_path"):
            validate_score_card_inputs("", score_card_configs)
        # Invalid score_card_configs - empty list
        with pytest.raises(ValueError, match="Invalid score_card_configs"):
            validate_score_card_inputs("input.json", [])

    def test_validate_test_execution_inputs_valid(self):
        """Test valid test execution inputs."""
        validate_test_execution_inputs(
            test_id="test1",
            image="image:latest",
            system_name="system1",
            system_params={"key": "value"},
            test_params={"param": "value"},
        )

    def test_validate_test_execution_inputs_invalid(self):
        """Test invalid test execution inputs."""
        # Invalid test_name - empty string
        with pytest.raises(ValueError, match="Invalid test id"):
            validate_test_execution_inputs(
                "", "image:latest", "system1", {"key": "value"}, {"param": "value"}
            )
        # Invalid image - empty string
        with pytest.raises(ValueError, match="Invalid image"):
            validate_test_execution_inputs(
                "test1", "", "system1", {"key": "value"}, {"param": "value"}
            )
        # Invalid system_name - empty string
        with pytest.raises(ValueError, match="Invalid system name"):
            validate_test_execution_inputs(
                "test1",
                "image:latest",
                "",
                {"key": "value"},
                {"param": "value"},
            )

    def test_validate_execution_inputs_with_audit_responses_valid(self):
        """Test validate_execution_inputs accepts a valid audit_responses_data dict."""
        validate_execution_inputs(
            suite_path="suite.yaml",
            systems_path="systems.yaml",
            execution_mode="tests_only",
            audit_responses_data=MOCK_AUDIT_RESPONSES,
            output_path="output.json",
        )

        # Also verify it works with end_to_end mode
        validate_execution_inputs(
            suite_path="suite.yaml",
            systems_path="systems.yaml",
            execution_mode="end_to_end",
            audit_responses_data=MOCK_AUDIT_RESPONSES,
            output_path=None,
        )

    def test_validate_execution_inputs_with_invalid_audit_responses_type(self):
        """Test validate_execution_inputs rejects non-dict audit_responses_data."""
        # List instead of dict
        with pytest.raises(ValueError, match="Invalid audit_responses_data"):
            validate_execution_inputs(
                suite_path="suite.yaml",
                systems_path="systems.yaml",
                execution_mode="tests_only",
                audit_responses_data=["not", "a", "dict"],  # type: ignore[arg-type]
            )

        # String instead of dict
        with pytest.raises(ValueError, match="Invalid audit_responses_data"):
            validate_execution_inputs(
                suite_path="suite.yaml",
                systems_path="systems.yaml",
                execution_mode="tests_only",
                audit_responses_data="not-a-dict",  # type: ignore[arg-type]
            )

    def test_validate_score_card_inputs_with_audit_responses_valid(self):
        """Test validate_score_card_inputs accepts valid score cards and audit_responses_data."""
        score_card_configs = [MOCK_SCORE_CARD_CONFIG]

        validate_score_card_inputs(
            input_path="results.json",
            score_card_configs=score_card_configs,
            audit_responses_data=MOCK_AUDIT_RESPONSES,
            output_path="scorecard_output.json",
        )

        # Also ensure it works when output_path is None
        validate_score_card_inputs(
            input_path="results.json",
            score_card_configs=score_card_configs,
            audit_responses_data=MOCK_AUDIT_RESPONSES,
            output_path=None,
        )

    def test_validate_score_card_inputs_with_invalid_audit_responses_type(self):
        """Test validate_score_card_inputs rejects non-dict audit_responses_data."""
        score_card_configs = [MOCK_SCORE_CARD_CONFIG]

        # List instead of dict
        with pytest.raises(ValueError, match="Invalid audit_responses_data"):
            validate_score_card_inputs(
                input_path="results.json",
                score_card_configs=score_card_configs,
                audit_responses_data=["not", "a", "dict"],  # type: ignore[arg-type]
                output_path="scorecard_output.json",
            )

        # Integer instead of dict
        with pytest.raises(ValueError, match="Invalid audit_responses_data"):
            validate_score_card_inputs(
                input_path="results.json",
                score_card_configs=score_card_configs,
                audit_responses_data=123,  # type: ignore[arg-type]
                output_path="scorecard_output.json",
            )


class TestWorkflowValidation:
    """Test the validate_workflow_configurations function."""

    def test_validate_workflow_configurations_valid(
        self, demo_suite, demo_systems, manifests
    ):
        """Test valid workflow configurations."""
        errors = validate_workflow_configurations(demo_suite, demo_systems, manifests)
        # Should have some errors due to incompatible systems in demo data
        assert isinstance(errors, list)

    def test_validate_workflow_configurations_empty_content(self):
        """Test validation with empty configurations."""
        suite_data = {"suite_name": "Empty", "test_suite": []}
        suite = SuiteConfig(**suite_data)

        # Empty systems
        systems_data = {"systems": {}}
        systems = SystemsConfig(**systems_data)

        errors = validate_workflow_configurations(suite, systems)
        assert any("Test suite is empty" in e for e in errors)
        assert any("Systems configuration is empty" in e for e in errors)

    def test_validate_workflow_configurations_with_manifests(
        self, demo_systems, manifests
    ):
        """Test validation with manifests provided."""
        suite_data = {
            "suite_name": "Compatible Test",
            "description": "Suite Description",
            "test_suite": [
                {
                    "name": "test_llm",
                    "id": "test_llm",
                    "description": "Test Description",
                    "image": "my-registry/mock_tester:latest",
                    "systems_under_test": ["my_llm_service"],
                    "params": {},
                }
            ],
        }
        suite = SuiteConfig(**suite_data)

        errors = validate_workflow_configurations(suite, demo_systems, manifests)
        assert errors == []


class TestCreateExecutionPlanEdgeCases:
    """Test edge cases in create_test_execution_plan function."""

    def test_empty_suite(self):
        """Test with empty suite."""
        suite_data = {"suite_name": "Empty", "test_suite": []}
        suite = SuiteConfig(**suite_data)
        systems_data = {
            "systems": {
                "sys1": {
                    "type": "llm_api",
                    "params": {"base_url": "http://x", "model": "x-model"},
                }
            }
        }
        systems = SystemsConfig(**systems_data)  # type: ignore

        plan = create_test_execution_plan(suite, systems, {})
        assert plan == []

    def test_empty_systems(self):
        """Test with empty systems."""
        suite_data = {
            "suite_name": "Test",
            "description": "Suite Description",
            "test_suite": [
                {
                    "name": "test1",
                    "id": "test1",
                    "description": "Test Description",
                    "image": "image:latest",
                    "systems_under_test": ["sys1"],
                    "params": {},
                }
            ],
        }
        suite = SuiteConfig(**suite_data)
        systems_data = {"systems": {}}
        systems = SystemsConfig(**systems_data)

        plan = create_test_execution_plan(suite, systems, {"image:latest": True})
        assert plan == []

    def test_test_with_volumes(self, demo_systems):
        """Test execution plan with volumes."""
        suite_data = {
            "suite_name": "Volume Test",
            "description": "Suite Description",
            "test_suite": [
                {
                    "name": "test_with_volumes",
                    "id": "test_with_volumes",
                    "description": "Test Description",
                    "image": "image:latest",
                    "systems_under_test": ["my_llm_service"],
                    "params": {"param1": "value1"},
                    "volumes": {"/host": "/container"},
                }
            ],
        }
        suite = SuiteConfig(**suite_data)

        plan = create_test_execution_plan(suite, demo_systems, {"image:latest": True})
        assert len(plan) == 1
        assert "__volumes" in plan[0]["test_params"]
        assert plan[0]["test_params"]["__volumes"] == {"/host": "/container"}
        assert plan[0]["test_params"]["param1"] == "value1"

    def test_test_without_systems_under_test(self, demo_systems):
        """Test with empty systems_under_test."""
        suite_data = {
            "suite_name": "No SUT Test",
            "description": "Suite Description",
            "test_suite": [
                {
                    "name": "test_no_sut",
                    "id": "test_no_sut",
                    "description": "Test Description",
                    "image": "image:latest",
                    "systems_under_test": [],
                    "params": {},
                }
            ],
        }
        suite = SuiteConfig(**suite_data)

        plan = create_test_execution_plan(suite, demo_systems, {"image:latest": True})
        assert plan == []

    def test_test_with_additional_systems(self, demo_systems):
        """Test execution plan includes additional systems."""
        suite_data = {
            "suite_name": "Multi System Test",
            "description": "Suite Description",
            "test_suite": [
                {
                    "name": "test_multi_system",
                    "id": "test_multi_system",
                    "description": "Test Description",
                    "image": "image:latest",
                    "systems_under_test": ["my_llm_service"],
                    "systems": {"simulator_system": "another_llm_service"},
                    "params": {"param1": "value1"},
                }
            ],
        }
        suite = SuiteConfig(**suite_data)

        plan = create_test_execution_plan(suite, demo_systems, {"image:latest": True})
        assert len(plan) == 1

        # Check systems_params structure
        systems_params = plan[0]["systems_params"]
        assert "system_under_test" in systems_params
        assert "simulator_system" in systems_params
        assert systems_params["system_under_test"]["type"] == "llm_api"
        assert systems_params["simulator_system"]["type"] == "llm_api"

    def test_system_with_missing_image(self, demo_systems):
        """Test case where test image is not in image_availability."""
        suite_data = {
            "suite_name": "Missing Image Test",
            "description": "Suite Description",
            "test_suite": [
                {
                    "name": "test_missing_image",
                    "id": "test_missing_image",
                    "description": "Test Description",
                    "image": "missing:latest",
                    "systems_under_test": ["my_llm_service"],
                    "params": {},
                }
            ],
        }
        suite = SuiteConfig(**suite_data)

        # Image not in availability dict
        plan = create_test_execution_plan(suite, demo_systems, {"other:latest": True})
        assert plan == []  # Should skip tests with unavailable images


class TestParameterValidationEdgeCases:
    """Test edge cases in parameter validation."""

    def test_validate_test_parameters_with_empty_schema(self):
        """Test parameter validation with empty input schema."""
        manifest_data = {
            "name": "no_params_test",
            "version": "1.0.0",
            "description": "Container with no parameters",
            "input_systems": [
                {"name": "system_under_test", "type": "llm_api", "required": True}
            ],
            "input_schema": [],
            "output_metrics": ["success"],
        }
        manifest = Manifest(**manifest_data)

        class DummyTest:
            name = "test"
            params = {"unexpected_param": "value"}

        test = DummyTest()
        errors = validate_test_parameters(test, manifest)
        assert any("Unknown parameter 'unexpected_param'" in e for e in errors)
        assert "Valid parameters: none" in errors[0]


class TestVolumeValidation:
    def _suite(self, vols):
        return SuiteConfig(
            **{
                "suite_name": "Test Volumes Test",
                "description": "Suite Description",
                "test_suite": [
                    {
                        "name": "t",
                        "id": "t",
                        "description": "Tests Description",
                        "image": "img:latest",
                        "systems_under_test": ["my_llm_service"],
                        "params": {},
                        **({"volumes": vols} if vols is not None else {}),
                    }
                ],
            }
        )

    @pytest.mark.parametrize("variant", ["input_only", "output_only", "both"])
    def test_ok(self, tmp_path, variant):
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        vols = (
            {"input": str(in_dir)}
            if variant == "input_only"
            else {"output": str(out_dir)}
            if variant == "output_only"
            else {"input": str(in_dir), "output": str(out_dir)}
        )
        validate_test_volumes(self._suite(vols))

    def test_no_volumes_is_ok(self):
        validate_test_volumes(self._suite(None))

    def test_missing_both_keys_raises(self, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        with pytest.raises(ValueError, match="at least one of"):
            validate_test_volumes(self._suite({"other": str(other)}))

    def test_volumes_not_dict_raises(self, tmp_path):
        # build a valid suite first
        in_dir = tmp_path / "in"
        in_dir.mkdir()
        suite = self._suite({"input": str(in_dir)})

        suite.test_suite[0].volumes = "this is not a dict test"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="must be a dict"):
            validate_test_volumes(suite)

    def test_non_string_or_empty_path_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_test_volumes(self._suite({"input": "   "}))
        with pytest.raises(ValueError, match="non-empty string"):
            validate_test_volumes(self._suite({"output": 123}))

    def test_nonexistent_or_not_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            validate_test_volumes(self._suite({"input": str(tmp_path / "missing")}))

        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(ValueError, match="is not a directory"):
            validate_test_volumes(self._suite({"output": str(f)}))


class TestValidateIDs:
    score_card_duplicate_data = {
        "score_card_name": "id duplicated score card",
        "indicators": [
            {
                "id": "id_bayau",
                "name": "this is the first dup name",
            },
            {
                "id": "id_bayau",
                "name": "this is the second dup name",
            },
        ],
    }

    test_suite_duplicate_data = {
        "suite_name": "id duplicated test suite",
        "test_suite": [
            {
                "id": "id_bayau",
                "name": "this is the first dup name",
                "image": "validation:latest",
                "systems_under_test": ["garak"],
            },
            {
                "id": "id_bayau",
                "name": "this is the second dup name",
                "image": "validation:latest",
                "systems_under_test": ["garak"],
            },
        ],
    }

    def test_id_test_suite_validation_success(self, tmp_path):
        """Test ID validation in test suite with no duplicate IDs."""
        config_folder = tmp_path / "config"
        suite_config_path = config_folder / "demo_test.yaml"
        config_folder.mkdir()

        test_suite_data = {
            "suite_name": "id validation test suite",
            "test_suite": [
                {
                    "id": "id_bayau",
                    "name": "this is the name",
                    "image": "validation:latest",
                    "systems_under_test": ["garak"],
                },
            ],
        }

        with open(suite_config_path, "w") as f:
            yaml.dump(test_suite_data, f)

        validate_ids(suite_config_path)

    def test_id_test_suite_missing_id_field_error(self, tmp_path):
        """Test ID validation in test suite with missing ID field."""
        config_folder = tmp_path / "config"
        suite_config_path = config_folder / "demo_test.yaml"
        config_folder.mkdir()

        test_suite_data = {
            "suite_name": "id validation test suite",
            "test_suite": [
                {
                    # "id": "missing_id_field",
                    "name": "this is the name",
                    "image": "validation:latest",
                    "systems_under_test": ["garak"],
                },
            ],
        }

        with open(suite_config_path, "w") as f:
            yaml.dump(test_suite_data, f)

        with pytest.raises(
            MissingIDFieldError, match="Missing required id field in test of test suite"
        ):
            validate_ids(suite_config_path)

    def test_id_score_card_missing_id_field_error(self, tmp_path):
        """Test ID validation in score card with missing ID field."""
        config_folder = tmp_path / "config"
        score_card_config_path = config_folder / "demo_score_card.yaml"
        config_folder.mkdir()

        score_card_data = {
            "score_card_name": "id validation score card",
            "indicators": [
                {
                    # "id": "missing_id_field",
                    "name": "this is the name",
                },
            ],
        }

        with open(score_card_config_path, "w") as f:
            yaml.dump(score_card_data, f)

        with pytest.raises(
            MissingIDFieldError,
            match="Missing required id field in indicator of score card",
        ):
            validate_ids(score_card_config_path)

    def test_id_score_card_validation_success(self, tmp_path):
        """Test ID validation in score card with no duplicate IDs."""
        config_folder = tmp_path / "config"
        score_card_config_path = config_folder / "demo_score_card.yaml"
        config_folder.mkdir()

        score_card_data = {
            "score_card_name": "id validation score card",
            "indicators": [
                {
                    "id": "id_bayau",
                    "name": "this is the name",
                },
            ],
        }

        with open(score_card_config_path, "w") as f:
            yaml.dump(score_card_data, f)

        validate_ids(score_card_config_path)

    def test_id_score_card_and_test_suite_validation_success(self, tmp_path):
        """Test ID validation in test suite and score card with no duplicate IDs."""
        config_folder = tmp_path / "config"
        suite_config_path = config_folder / "demo_test.yaml"
        score_card_config_path = config_folder / "demo_score_card.yaml"
        config_folder.mkdir()

        test_suite_data = {
            "suite_name": "id validation test suite",
            "test_suite": [
                {
                    "id": "id_bayau",
                    "name": "this is the name",
                    "image": "validation:latest",
                    "systems_under_test": ["garak"],
                },
            ],
        }

        score_card_data = {
            "score_card_name": "id validation score card",
            "indicators": [
                {
                    "id": "id_bayau",
                    "name": "this is the name",
                },
            ],
        }

        with open(suite_config_path, "w") as f:
            yaml.dump(test_suite_data, f)

        with open(score_card_config_path, "w") as f:
            yaml.dump(score_card_data, f)

        validate_ids(suite_config_path, score_card_config_path)

    def test_validation_test_suite_with_duplicates_error(self, tmp_path):
        """Test ID validation in test suite with duplicate IDs and DuplicateIDError exception."""
        config_folder = tmp_path / "config"
        suite_config_path = config_folder / "demo_test.yaml"
        config_folder.mkdir()

        with open(suite_config_path, "w") as f:
            yaml.dump(self.test_suite_duplicate_data, f)

        with pytest.raises(
            DuplicateIDError, match="Duplicate id -> id_bayau in test_suite"
        ) as exe_raised:
            validate_ids(suite_config_path)

        error = exe_raised.value
        assert len(error.duplicate_dict["t_id_bayau"]["occurrences"]) == 2
        assert error.duplicate_dict["t_id_bayau"]["config_type"] == "test_suite"
        assert error.duplicate_dict["t_id_bayau"]["id"] == "id_bayau"
        assert (
            error.duplicate_dict["t_id_bayau"]["occurrences"][0]["test_name"]
            == "this is the first dup name"
        )
        assert (
            error.duplicate_dict["t_id_bayau"]["occurrences"][1]["test_name"]
            == "this is the second dup name"
        )

    def test_validation_score_card_with_duplicates_error(self, tmp_path):
        """Test ID validation in score card with duplicate IDs and DuplicateIDError exception."""
        config_folder = tmp_path / "config"
        score_card_config_path = config_folder / "demo_score_card.yaml"
        config_folder.mkdir()

        with open(score_card_config_path, "w") as f:
            yaml.dump(self.score_card_duplicate_data, f)

        with pytest.raises(
            DuplicateIDError, match="Duplicate id -> id_bayau in score_card"
        ) as exe_raised:
            validate_ids(score_card_config_path)

        error = exe_raised.value
        assert len(error.duplicate_dict["s_id_bayau"]["occurrences"]) == 2
        assert error.duplicate_dict["s_id_bayau"]["config_type"] == "score_card"
        assert error.duplicate_dict["s_id_bayau"]["id"] == "id_bayau"
        assert (
            error.duplicate_dict["s_id_bayau"]["occurrences"][0]["indicator_name"]
            == "this is the first dup name"
        )
        assert (
            error.duplicate_dict["s_id_bayau"]["occurrences"][1]["indicator_name"]
            == "this is the second dup name"
        )

    def test_validation_score_card_and_test_suite_with_duplicates_error(self, tmp_path):
        """Test ID validation in score card and test suite with duplicate IDs and DuplicateIDError exception."""
        config_folder = tmp_path / "config"
        suite_config_path = config_folder / "demo_test.yaml"
        score_card_config_path = config_folder / "demo_score_card.yaml"
        config_folder.mkdir()

        with open(suite_config_path, "w") as f:
            yaml.dump(self.test_suite_duplicate_data, f)

        with open(score_card_config_path, "w") as f:
            yaml.dump(self.score_card_duplicate_data, f)

        with pytest.raises(DuplicateIDError) as exe_raised:
            validate_ids(score_card_config_path, suite_config_path)

        error = exe_raised.value
        assert len(error.duplicate_dict["t_id_bayau"]["occurrences"]) == 2
        assert len(error.duplicate_dict["s_id_bayau"]["occurrences"]) == 2

    def test_invalid_id_formats_error(self):
        """Test invalid id formats fail schema validation"""

        # Examples of invalid IDs not in 0-9, a-z, _ and max length 32
        invalid_ids = ["Invalid-ID", "UPPERCASE", "has-hyphen", "has.dot", "a" * 33]

        for bad_id in invalid_ids:
            test_suite_data = {
                "suite_name": "invalid id",
                "test_suite": [
                    {
                        "id": bad_id,
                        "name": "invalid id test",
                        "description": "Test with invalid id",
                        "image": "demo:latest",
                        "systems_under_test": ["my_llm_service"],
                    }
                ],
            }
            score_card_data = {
                "score_card_name": "id validation score card",
                "indicators": [
                    {
                        "id": bad_id,
                        "name": "this is the name",
                        "apply_to": {"test_id": "some_test"},
                        "metric": "success",
                        "assessment": [
                            {
                                "outcome": "PASS",
                                "condition": "equal_to",
                                "threshold": True,
                            }
                        ],
                    },
                ],
            }

            with pytest.raises(ValidationError):
                SuiteConfig(**test_suite_data)

            with pytest.raises(ValidationError):
                ScoreCard(**score_card_data)

    def test_invalid_yaml_format(self, tmp_path):
        """Test invalid YAML format does not affect ID validation."""
        suite_folder = tmp_path / "suites"
        suite_config_path = suite_folder / "demo_test.yaml"
        suite_folder.mkdir()

        invalid_yaml_content = """
        invalid yaml
        """

        with open(suite_config_path, "w") as f:
            f.write(invalid_yaml_content)

        validate_ids(suite_config_path)

    def test_missing_test_suite_id_field_error(self):
        """Test missing id field in test suite fails schema validation."""
        test_suite_data = {
            "suite_name": "missing id",
            "test_suite": [
                {
                    # "id": "no_id_test",
                    "name": "no id test",
                    "description": "missing id",
                    "image": "demo:latest",
                    "systems_under_test": ["my_llm_service"],
                }
            ],
        }

        with pytest.raises(ValidationError):
            SuiteConfig(**test_suite_data)

    def test_missing_score_card_id_field_error(self):
        """Test missing id field in score card fails schema validation."""
        score_card_data = {
            "score_card_name": "missing id",
            "indicators": [
                {
                    # "id": "no_id_test",
                    "name": "no id test",
                    "apply_to": {"test_id": "some_test"},
                    "metric": "success",
                    "assessment": [
                        {
                            "outcome": "PASS",
                            "condition": "equal_to",
                            "threshold": True,
                        }
                    ],
                }
            ],
        }

        with pytest.raises(ValidationError):
            ScoreCard(**score_card_data)

    def test_score_card_referencing_unknown_test_error(self):
        """Test scorecard fails when indicators reference unknown test IDs."""

        engine = ScoreCardEngine()

        result = TestExecutionResult(
            "test_name", "existing_test_id", "sut", "image:latest"
        )
        result.test_results = {"success": True}

        score_card = ScoreCard(
            score_card_name="demo score card",
            indicators=[
                ScoreCardIndicator(
                    id="demo_indicator",
                    name="demo indicator",
                    apply_to=ScoreCardFilter(test_id="nonexistent_test_id"),
                    metric="success",
                    assessment=[
                        AssessmentRule(
                            outcome="PASS",
                            condition="equal_to",
                            threshold=True,
                            description="demo pass rule",
                        )
                    ],
                )
            ],
        )

        with pytest.raises(
            ValueError,
            match="Score card indicators don't match any test ids in the test results",
        ):
            engine.evaluate_scorecard([result], score_card)


class TestRAGResponseSchema:
    """Test cases for RAG response validation schemas."""

    def test_rag_citation_valid(self):
        """Test valid RAG citation creation."""
        citation = RAGCitation(
            retrieved_context="This is some retrieved text from a document.",
            document_id="policy.pdf",
            score=0.95,
            source_id="company_policies"
        )
        
        assert citation.retrieved_context == "This is some retrieved text from a document."
        assert citation.document_id == "policy.pdf"
        assert citation.score == 0.95
        assert citation.source_id == "company_policies"

    def test_rag_citation_validation_errors(self):
        """Test RAG citation validation errors."""
        # Empty retrieved_context
        with pytest.raises(ValidationError, match="string_too_short"):
            RAGCitation(
                retrieved_context="",
                document_id="doc.txt"
            )
        
        # Empty document_id
        with pytest.raises(ValidationError, match="string_too_short"):
            RAGCitation(
                retrieved_context="Some text",
                document_id=""
            )
        
        # Invalid score range
        with pytest.raises(ValidationError, match="greater_than_equal"):
            RAGCitation(
                retrieved_context="Some text",
                document_id="doc.txt",
                score=-0.1
            )
        
        with pytest.raises(ValidationError, match="less_than_equal"):
            RAGCitation(
                retrieved_context="Some text",
                document_id="doc.txt",
                score=1.5
            )

    def test_rag_context_valid(self):
        """Test valid RAG context creation."""
        citations = [
            RAGCitation(
                retrieved_context="First citation text.",
                document_id="doc1.pdf",
                score=0.9
            ),
            RAGCitation(
                retrieved_context="Second citation text.",
                document_id="doc2.pdf",
                score=0.8
            )
        ]
        
        context = RAGContext(citations=citations)
        assert len(context.citations) == 2
        assert context.citations[0].document_id == "doc1.pdf"
        assert context.citations[1].document_id == "doc2.pdf"

    def test_rag_context_empty_citations(self):
        """Test RAG context with empty citations list."""
        context = RAGContext(citations=[])
        assert context.citations == []

    def test_validate_rag_response_valid(self):
        """Test validate_rag_response with valid response structure."""
        response_dict = {
            "choices": [
                {
                    "message": {
                        "context": {
                            "citations": [
                                {
                                    "retrieved_context": "This is citation text.",
                                    "document_id": "source.pdf",
                                    "score": 0.85,
                                    "source_id": "knowledge_base"
                                }
                            ]
                        }
                    }
                }
            ]
        }
        
        citations = validate_rag_response(response_dict)
        assert len(citations) == 1
        assert citations[0].retrieved_context == "This is citation text."
        assert citations[0].document_id == "source.pdf"
        assert citations[0].score == 0.85
        assert citations[0].source_id == "knowledge_base"

    def test_validate_rag_response_multiple_citations(self):
        """Test validate_rag_response with multiple citations."""
        response_dict = {
            "choices": [
                {
                    "message": {
                        "context": {
                            "citations": [
                                {
                                    "retrieved_context": "First citation.",
                                    "document_id": "doc1.pdf"
                                },
                                {
                                    "retrieved_context": "Second citation.",
                                    "document_id": "doc2.pdf",
                                    "score": 0.7
                                }
                            ]
                        }
                    }
                }
            ]
        }
        
        citations = validate_rag_response(response_dict)
        assert len(citations) == 2
        assert citations[0].document_id == "doc1.pdf"
        assert citations[1].document_id == "doc2.pdf"

    def test_validate_rag_response_missing_required_structure(self):
        """Test validate_rag_response with missing required response structure."""
        # Missing message
        with pytest.raises(KeyError, match="message"):
            validate_rag_response({"choices": [{"context": {"citations": []}}]})
        
        # Missing context
        with pytest.raises(KeyError, match="context"):
            validate_rag_response({"choices": [{"message": {"citations": []}}]})
