from unittest.mock import MagicMock, patch

import pytest

from asqi.container_manager import run_container_with_args
from asqi.schemas import SuiteConfig, SUTDefinition, SUTsConfig, TestDefinition
from asqi.validation import create_test_execution_plan
from asqi.workflow import execute_single_test


class TestEnvironmentVariables:
    """Test suite for environment variable handling."""

    @pytest.fixture
    def sample_sut_config(self):
        """Sample SUT configuration with API key environment variable."""
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
                    config={
                        "model": "gpt-4o-mini",
                        "api_key": "sk-123",
                    },
                )
            }
        )

    def test_create_test_execution_plan_flattens_sut_config(
        self, sample_suite_config, sample_suts_config
    ):
        """Test that create_test_execution_plan correctly flattens SUT configuration."""
        image_availability = {"my-registry/test:latest": True}

        execution_plan = create_test_execution_plan(
            sample_suite_config, sample_suts_config, image_availability
        )

        assert len(execution_plan) == 1
        sut_config = execution_plan[0]["sut_config"]

        # Verify the SUT config is flattened correctly
        assert sut_config["type"] == "llm_api"
        assert sut_config["model"] == "gpt-4o-mini"
        assert sut_config["api_key"] == "sk-123"

        # Ensure config is not nested
        assert "config" not in sut_config

    @patch("asqi.workflow.run_container_with_args")
    def test_execute_single_test_passes_environment_variable_from_dotenv(
        self, mock_run_container, sample_sut_config, tmp_path, monkeypatch
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
            sut_config=sample_sut_config,
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

        sut_config = {
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
            sut_config=sut_config,
            test_params={"generations": 2},
        )

        # Verify only the specified environment variable is passed
        mock_run_container.assert_called_once()
        call_kwargs = mock_run_container.call_args[1]

        assert "environment" in call_kwargs
        assert call_kwargs["environment"]["API_KEY"] == "sk-123"

    @patch("asqi.container_manager.docker_client")
    def test_run_container_with_args_environment_parameter(self, mock_docker_client):
        """Test that run_container_with_args correctly passes environment variables to Docker."""
        # Mock Docker client and container
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "test_container_789"
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b'{"success": true}'

        mock_client.containers.run.return_value = mock_container
        mock_docker_client.return_value.__enter__.return_value = mock_client

        # Test environment variables
        test_env = {"API_KEY": "secret123", "MODEL_NAME": "gpt-4"}

        # Call run_container_with_args with environment
        _result = run_container_with_args(
            image="test:latest", args=["--test", "arg"], environment=test_env
        )

        # Verify Docker container was created with correct environment
        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]

        assert "environment" in call_kwargs
        assert call_kwargs["environment"] == test_env

    @patch("asqi.container_manager.docker_client")
    def test_run_container_with_args_no_environment_parameter(self, mock_docker_client):
        """Test that run_container_with_args handles missing environment parameter."""
        # Mock Docker client and container
        mock_client = MagicMock()
        mock_container = MagicMock()
        mock_container.id = "test_container_999"
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b'{"success": true}'

        mock_client.containers.run.return_value = mock_container
        mock_docker_client.return_value.__enter__.return_value = mock_client

        # Call run_container_with_args without environment parameter
        _result = run_container_with_args(image="test:latest", args=["--test", "arg"])

        # Verify Docker container was created with empty environment
        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]

        assert "environment" in call_kwargs
        assert call_kwargs["environment"] == {}
