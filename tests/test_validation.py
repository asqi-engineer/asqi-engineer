import tempfile
from pathlib import Path

import pytest
import yaml

from asqi.main import load_and_validate_plan
from asqi.schemas import Manifest, SuiteConfig, SUTsConfig
from asqi.validation import (
    create_test_execution_plan,
    validate_manifests_against_tests,
    validate_sut_compatibility,
    validate_test_parameters,
    validate_test_plan,
)

# Test data
DEMO_SUITE_YAML = """
suite_name: "Mock Tester Sanity Check"
test_suite:
  - name: "run_mock_on_compatible_sut"
    image: "my-registry/mock_tester:latest"
    target_suts:
      - "my_backend_api"  # This should fail since mock_tester doesn't support this in your demo
    params:
      delay_seconds: 1
"""

DEMO_SUTS_YAML = """
systems_under_test:
  # This SUT is compatible with our mock_tester
  my_llm_service:
    type: "llm_api"
    config:
      provider: "some_provider"
      model: "model-x"
      api_key_env: "MY_LLM_API_KEY"

  # This SUT is *not* compatible, for demonstrating validation failure
  my_backend_api:
    type: "rest_api"
    config:
      uri: "https://api.example.com/v1/data"
"""

MOCK_TESTER_MANIFEST = {
    "name": "mock_tester",
    "version": "1.0.0",
    "description": "A minimal mock container for testing the executor interface.",
    "image_name": "my-registry/mock_tester:latest",
    "supported_suts": [{"type": "llm_api"}],
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

MOCK_MULTIPLE_MANIFEST = {
    "name": "garak",
    "version": "0.2.0",
    "description": "A security and safety probing tool for Large Language Models.",
    "image_name": "my-registry/garak:latest",
    "supported_suts": [{"type": "llm_api"}, {"type": "rest_api"}],
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
def demo_suts():
    """Fixture providing parsed demo SUTs."""
    data = yaml.safe_load(DEMO_SUTS_YAML)
    return SUTsConfig(**data)


@pytest.fixture
def manifests():
    """Fixture providing test manifests."""
    return {
        "my-registry/mock_tester:latest": Manifest(**MOCK_TESTER_MANIFEST),
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
        assert test1.name == "run_mock_on_compatible_sut"
        assert test1.image == "my-registry/mock_tester:latest"
        assert test1.target_suts == ["my_backend_api"]
        assert test1.params["delay_seconds"] == 1

    def test_suts_schema_validation(self, demo_suts):
        """Test that demo SUTs YAML parses correctly."""
        suts = demo_suts.systems_under_test
        assert len(suts) == 2

        # Check LLM service
        llm_sut = suts["my_llm_service"]
        assert llm_sut.type == "llm_api"
        assert llm_sut.config["provider"] == "some_provider"

        # Check backend API
        api_sut = suts["my_backend_api"]
        assert api_sut.type == "rest_api"
        assert api_sut.config["uri"] == "https://api.example.com/v1/data"

    def test_manifest_schema_validation(self, manifests):
        """Test that manifests parse correctly."""
        mock_manifest = manifests["my-registry/mock_tester:latest"]
        assert mock_manifest.name == "mock_tester"
        assert len(mock_manifest.supported_suts) == 1
        assert mock_manifest.supported_suts[0].type == "llm_api"


class TestCrossFileValidation:
    """Test validation logic that checks consistency across files."""

    def test_successful_validation(self, demo_suts, manifests):
        """Test validation passes for compatible SUTs."""
        # Create a suite with only compatible SUTs
        compatible_suite_data = {
            "suite_name": "Compatible Test",
            "test_suite": [
                {
                    "name": "test_llm_service",
                    "image": "my-registry/mock_tester:latest",
                    "target_suts": ["my_llm_service"],
                    "params": {"delay_seconds": 1},
                }
            ],
        }
        suite = SuiteConfig(**compatible_suite_data)

        errors = validate_test_plan(suite, demo_suts, manifests)
        assert errors == [], f"Expected no errors, but got: {errors}"

    def test_missing_image_manifest(self, demo_suite, demo_suts):
        """Test validation fails when manifest is missing for an image."""
        # Empty manifests dict
        empty_manifests = {}

        errors = validate_test_plan(demo_suite, demo_suts, empty_manifests)
        assert len(errors) > 0
        assert any("does not have a loaded manifest" in error for error in errors)

    def test_missing_sut_definition(self, demo_suts, manifests):
        """Test validation fails when SUT is not defined."""
        suite_data = {
            "suite_name": "Test with Missing SUT",
            "test_suite": [
                {
                    "name": "test_missing_sut",
                    "image": "my-registry/mock_tester:latest",
                    "target_suts": ["nonexistent_sut"],
                    "params": {},
                }
            ],
        }
        suite = SuiteConfig(**suite_data)

        errors = validate_test_plan(suite, demo_suts, manifests)
        assert len(errors) > 0
        assert any("is not defined in the SUTs file" in error for error in errors)

    def test_missing_required_parameter(self, demo_suts, manifests):
        """Test validation fails when required parameters are missing."""
        # Use garak which has required parameters
        suite_data = {
            "suite_name": "Test Missing Required Param",
            "test_suite": [
                {
                    "name": "garak_without_probes",
                    "image": "my-registry/garak:latest",
                    "target_suts": ["my_llm_service"],
                    "params": {},  # Missing required 'probes' parameter
                }
            ],
        }
        suite = SuiteConfig(**suite_data)

        errors = validate_test_plan(suite, demo_suts, manifests)
        assert len(errors) > 0
        assert any(
            "Required parameter 'probes' is missing" in error for error in errors
        )

    def test_unknown_parameter(self, demo_suts, manifests):
        """Test validation fails when unknown parameters are provided."""
        suite_data = {
            "suite_name": "Test Unknown Param",
            "test_suite": [
                {
                    "name": "test_unknown_param",
                    "image": "my-registry/mock_tester:latest",
                    "target_suts": ["my_llm_service"],
                    "params": {"delay_seconds": 1, "unknown_param": "should_fail"},
                }
            ],
        }
        suite = SuiteConfig(**suite_data)

        errors = validate_test_plan(suite, demo_suts, manifests)
        assert len(errors) > 0
        assert any("Unknown parameter 'unknown_param'" in error for error in errors)


class TestFileLoading:
    """Test the file loading functionality from main.py."""

    def test_load_yaml_files_from_disk(self):
        """Test loading actual YAML files from temporary disk files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write test files
            suite_file = temp_path / "demo_suite.yaml"
            suts_file = temp_path / "demo_suts.yaml"
            manifest_dir = temp_path / "manifests"
            manifest_dir.mkdir()

            # Write YAML files
            with open(suite_file, "w") as f:
                f.write(DEMO_SUITE_YAML)

            with open(suts_file, "w") as f:
                f.write(DEMO_SUTS_YAML)

            # Write manifest file
            manifest_file = manifest_dir / "manifest.yaml"
            with open(manifest_file, "w") as f:
                yaml.dump(MOCK_TESTER_MANIFEST, f)

            # Test that we can load and validate

            result = load_and_validate_plan(
                str(suite_file), str(suts_file), str(manifest_dir)
            )

            # Should have some validation errors due to incompatible SUT
            # (my_backend_api is not supported by mock_tester in this setup)
            assert result["status"] == "failure"
            assert len(result["errors"]) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_test_suite(self, demo_suts, manifests):
        """Test validation with empty test suite."""
        empty_suite_data = {"suite_name": "Empty Suite", "test_suite": []}
        suite = SuiteConfig(**empty_suite_data)

        errors = validate_test_plan(suite, demo_suts, manifests)
        assert errors == []  # Empty suite should be valid

    def test_multiple_target_suts(self, demo_suts, manifests):
        """Test validation with multiple target SUTs."""
        multi_sut_data = {
            "suite_name": "Multi SUT Test",
            "test_suite": [
                {
                    "name": "test_multiple_suts",
                    "image": "my-registry/garak:latest",
                    "target_suts": ["my_llm_service", "my_backend_api"],
                    "params": {
                        "probes": ["probe1", "probe2"]
                    },  # Provide required param for garak
                }
            ],
        }
        suite = SuiteConfig(**multi_sut_data)

        errors = validate_test_plan(suite, demo_suts, manifests)
        # Should pass since garak supports both llm_api and rest_api
        assert errors == []

    def test_no_parameters(self, demo_suts, manifests):
        """Test validation with no parameters (should be fine for mock_tester)."""
        no_params_data = {
            "suite_name": "No Params Test",
            "test_suite": [
                {
                    "name": "test_no_params",
                    "image": "my-registry/mock_tester:latest",
                    "target_suts": ["my_llm_service"],
                    "params": {},
                }
            ],
        }
        suite = SuiteConfig(**no_params_data)

        errors = validate_test_plan(suite, demo_suts, manifests)
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

    def test_validate_sut_compatibility(self, demo_suts, manifests):
        manifest = manifests["my-registry/mock_tester:latest"]

        class DummyTest:
            name = "t1"
            image = "my-registry/mock_tester:latest"
            target_suts = ["my_llm_service", "my_backend_api"]

        test = DummyTest()
        errors = validate_sut_compatibility(
            test, demo_suts.systems_under_test, manifest
        )
        # my_llm_service is supported, my_backend_api is not
        assert any("does not support SUT type 'rest_api'" in e for e in errors)
        assert not any("llm_api" in e for e in errors)

        # Unknown SUT
        test.target_suts = ["not_a_sut"]
        errors = validate_sut_compatibility(
            test, demo_suts.systems_under_test, manifest
        )
        assert any("Unknown SUT 'not_a_sut'" in e for e in errors)

    def test_validate_manifests_against_tests(self, demo_suts, manifests):
        # Valid test
        suite_data = {
            "suite_name": "Valid",
            "test_suite": [
                {
                    "name": "t1",
                    "image": "my-registry/mock_tester:latest",
                    "target_suts": ["my_llm_service"],
                    "params": {},
                }
            ],
        }
        suite = SuiteConfig(**suite_data)
        errors = validate_manifests_against_tests(suite, demo_suts, manifests)
        assert errors == []

        # Missing manifest
        suite_data["test_suite"][0]["image"] = "notfound:latest"
        suite = SuiteConfig(**suite_data)
        errors = validate_manifests_against_tests(suite, demo_suts, manifests)
        assert any("No manifest available for image" in e for e in errors)

    def test_create_test_execution_plan(self, demo_suts, manifests):
        suite_data = {
            "suite_name": "ExecPlan",
            "test_suite": [
                {
                    "name": "t1",
                    "image": "my-registry/mock_tester:latest",
                    "target_suts": ["my_llm_service"],
                    "params": {"delay_seconds": 1},
                },
                {
                    "name": "t2",
                    "image": "my-registry/mock_tester:latest",
                    "target_suts": ["my_llm_service", "my_backend_api"],
                    "params": {},
                },
            ],
        }
        suite = SuiteConfig(**suite_data)
        image_availability = {"my-registry/mock_tester:latest": True}
        plan = create_test_execution_plan(suite, demo_suts, image_availability)
        # Should create 3 plans (1 + 2)
        assert len(plan) == 3
        names = [p["test_name"] for p in plan]
        assert names.count("t1") == 1
        assert names.count("t2") == 2

        # Check that different SUTs are included
        sut_names = [p["sut_name"] for p in plan]
        assert "my_llm_service" in sut_names
        assert "my_backend_api" in sut_names

        # If image not available, plan is empty
        image_availability = {"my-registry/mock_tester:latest": False}
        plan = create_test_execution_plan(suite, demo_suts, image_availability)
        assert plan == []
