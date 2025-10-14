from unittest.mock import patch

import pytest

from asqi.config import ContainerConfig
from asqi.schemas import SuiteConfig, SystemDefinition, SystemsConfig, TestDefinition
from asqi.validation import create_test_execution_plan
from asqi.workflow import execute_single_test


class TestEnvironmentVariables:
    """Test suite for environment variable handling."""

    @pytest.fixture
    def sample_system_params(self):
        """Sample system parameters (flattened configuration)."""
        return {"type": "llm_api", "model": "gpt-4o-mini"}

    @pytest.fixture
    def sample_suite_config(self):
        """Sample test suite configuration."""

        return SuiteConfig(
            suite_name="Environment Test Suite",
            description="Suite description",
            test_suite=[
                TestDefinition(
                    name="test_with_api_key",
                    description="Test description",
                    image="my-registry/test:latest",
                    systems_under_test=["test_system"],
                    systems=None,
                    params={"generations": 1},
                    tags=None,
                    volumes={},
                )
            ],
            test_suite_default=None,
        )

    @pytest.fixture
    def sample_systems_config(self):
        """Sample systems configuration with API key."""

        return SystemsConfig(
            systems={
                "test_system": SystemDefinition(
                    type="llm_api",
                    description="System description",
                    provider="openai",
                    params={
                        "model": "gpt-4o-mini",
                        "api_key": "sk-123",
                    },
                )
            }
        )

    def test_create_test_execution_plan_flattens_system_params(
        self, sample_suite_config, sample_systems_config
    ):
        """Test that create_test_execution_plan correctly flattens system parameters."""
        image_availability = {"my-registry/test:latest": True}

        execution_plan = create_test_execution_plan(
            sample_suite_config, sample_systems_config, image_availability
        )

        assert len(execution_plan) == 1
        systems_params = execution_plan[0]["systems_params"]
        system_params = systems_params["system_under_test"]

        # Verify the system params are flattened correctly
        assert system_params["type"] == "llm_api"
        assert system_params["description"] == "System description"
        assert system_params["provider"] == "openai"
        assert system_params["model"] == "gpt-4o-mini"
        assert system_params["api_key"] == "sk-123"

        # Ensure config is not nested
        assert "config" not in system_params

    @patch("asqi.workflow.run_container_with_args")
    def test_execute_single_test_passes_environment_variable_from_dotenv(
        self, mock_run_container, tmp_path, monkeypatch
    ):
        """Test that execute_single_test loads TEST_API_KEY from explicit env_file."""
        dotenv_content = "TEST_API_KEY=test_secret_key_12345\n"
        dotenv_path = tmp_path / "custom.env"
        dotenv_path.write_text(dotenv_content)
        monkeypatch.chdir(tmp_path)

        # System params with explicit env_file
        system_params_with_env_file = {
            "type": "llm_api",
            "description": "System description",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "env_file": "custom.env",
        }

        # Mock the container result
        mock_run_container.return_value = {
            "success": True,
            "exit_code": 0,
            "output": '{"success": true, "score": 0.8}',
            "error": "",
            "container_id": "test_container_123",
        }

        container_config: ContainerConfig = ContainerConfig()

        # Execute the test
        _result = execute_single_test(
            test_name="test_env_vars",
            image="my-registry/test:latest",
            sut_name="test_system",
            systems_params={"system_under_test": system_params_with_env_file},
            test_params={"generations": 1},
            container_config=container_config,
        )

        # Verify run_container_with_args was called with environment variables from custom.env
        mock_run_container.assert_called_once()
        call_kwargs = mock_run_container.call_args[1]

        assert "environment" in call_kwargs
        assert call_kwargs["environment"]["TEST_API_KEY"] == "test_secret_key_12345"

    @patch("asqi.workflow.run_container_with_args")
    def test_execute_single_test_explicit_api_key_only(
        self, mock_run_container, tmp_path, monkeypatch
    ):
        """Test that only explicit api_key is passed to container."""
        # Create .env file that should NOT be automatically loaded
        dotenv_content = "API_KEY=test_secret_key_12345\n"
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text(dotenv_content)
        monkeypatch.chdir(tmp_path)

        system_params = {
            "type": "llm_api",
            "description": "System description",
            "provider": "openai",
            "model": "gpt-4",
            "api_key": "sk-123",
        }

        # Mock the container result
        mock_run_container.return_value = {
            "success": True,
            "exit_code": 0,
            "output": '{"success": true, "score": 0.9}',
            "error": "",
            "container_id": "test_container_456",
        }

        container_config: ContainerConfig = ContainerConfig()

        # Execute the test
        _result = execute_single_test(
            test_name="test_specific_env_var",
            image="my-registry/test:latest",
            sut_name="openai_system",
            systems_params={"system_under_test": system_params},
            test_params={"generations": 2},
            container_config=container_config,
        )

        # Verify only the explicit API key is passed (no automatic .env loading)
        mock_run_container.assert_called_once()
        call_kwargs = mock_run_container.call_args[1]

        assert "environment" in call_kwargs
        assert call_kwargs["environment"]["API_KEY"] == "sk-123"
        # Verify .env file variables are NOT automatically loaded
        assert "TEST_SECRET_KEY" not in call_kwargs["environment"]
