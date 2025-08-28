from unittest.mock import patch

import pytest

from asqi.schemas import SuiteConfig, SUTDefinition, SUTsConfig, TestDefinition
from asqi.validation import create_test_execution_plan
from asqi.workflow import execute_single_test


class TestEnvironmentVariables:
    """Test suite for environment variable handling."""

    @pytest.fixture
    def sample_sut_params(self):
        """Sample SUT parameters (flattened configuration)."""
        return {"type": "llm_api", "model": "gpt-4o-mini"}

    @pytest.fixture
    def sample_suite_config(self):
        """Sample test suite configuration."""

        return SuiteConfig(
            suite_name="Environment Test Suite",
            test_suite=[
                TestDefinition(
                    name="test_with_api_key",
                    image="my-registry/test:latest",
                    target_suts=["test_sut"],
                    params={"generations": 1},
                    tags=None,
                    volumes={},
                )
            ],
        )

    @pytest.fixture
    def sample_suts_config(self):
        """Sample SUTs configuration with API key."""

        return SUTsConfig(
            systems_under_test={
                "test_sut": SUTDefinition(
                    type="llm_api",
                    params={
                        "model": "gpt-4o-mini",
                        "api_key": "sk-123",
                    },
                )
            }
        )

    def test_create_test_execution_plan_flattens_sut_params(
        self, sample_suite_config, sample_suts_config
    ):
        """Test that create_test_execution_plan correctly flattens SUT parameters."""
        image_availability = {"my-registry/test:latest": True}

        execution_plan = create_test_execution_plan(
            sample_suite_config, sample_suts_config, image_availability
        )

        assert len(execution_plan) == 1
        sut_params = execution_plan[0]["sut_params"]

        # Verify the SUT params are flattened correctly
        assert sut_params["type"] == "llm_api"
        assert sut_params["model"] == "gpt-4o-mini"
        assert sut_params["api_key"] == "sk-123"

        # Ensure config is not nested
        assert "config" not in sut_params

    @patch("asqi.workflow.run_container_with_args")
    def test_execute_single_test_passes_environment_variable_from_dotenv(
        self, mock_run_container, sample_sut_params, tmp_path, monkeypatch
    ):
        """Test that execute_single_test loads TEST_API_KEY from .env file."""
        dotenv_content = "TEST_API_KEY=test_secret_key_12345\n"
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text(dotenv_content)
        monkeypatch.chdir(tmp_path)

        # Mock the container result
        mock_run_container.return_value = {
            "success": True,
            "exit_code": 0,
            "output": '{"success": true, "score": 0.8}',
            "error": "",
            "container_id": "test_container_123",
        }

        # Execute the test
        _result = execute_single_test(
            test_name="test_env_vars",
            image="my-registry/test:latest",
            sut_name="test_sut",
            sut_params=sample_sut_params,
            test_params={"generations": 1},
        )

        # Verify run_container_with_args was called with environment variables from .env
        mock_run_container.assert_called_once()
        call_kwargs = mock_run_container.call_args[1]

        assert "environment" in call_kwargs
        assert call_kwargs["environment"]["TEST_API_KEY"] == "test_secret_key_12345"
        assert "OPENAI_API_KEY" not in call_kwargs["environment"]

    @patch("asqi.workflow.run_container_with_args")
    def test_execute_single_test_multiple_environment_variables(
        self, mock_run_container, tmp_path, monkeypatch
    ):
        """Test that api_key gets priority over environment variables."""
        dotenv_content = "API_KEY=test_secret_key_12345\n"
        dotenv_path = tmp_path / ".env"
        dotenv_path.write_text(dotenv_content)
        monkeypatch.chdir(tmp_path)

        sut_params = {
            "type": "llm_api",
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

        # Execute the test
        _result = execute_single_test(
            test_name="test_specific_env_var",
            image="my-registry/test:latest",
            sut_name="openai_sut",
            sut_params=sut_params,
            test_params={"generations": 2},
        )

        # Verify only the specified environment variable is passed
        mock_run_container.assert_called_once()
        call_kwargs = mock_run_container.call_args[1]

        assert "environment" in call_kwargs
        assert call_kwargs["environment"]["API_KEY"] == "sk-123"
