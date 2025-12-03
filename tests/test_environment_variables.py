from unittest.mock import patch

import pytest

from asqi.config import ContainerConfig
from asqi.schemas import (
    LLMAPIConfig,
    LLMAPIParams,
    RAGAPIConfig,
    SuiteConfig,
    SystemsConfig,
    TestDefinition,
)
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
                    id="test_id",
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
    def sample_llm_api_systems_config(self):
        """Sample llm api systems configuration with API key."""

        return SystemsConfig(
            systems={
                "test_system": LLMAPIConfig(
                    type="llm_api",
                    description="System description",
                    provider="openai",
                    params=LLMAPIParams(
                        base_url="http://URL",
                        env_file="ENV_FILE",
                        model="gpt-4o-mini",
                        api_key="sk-123",
                    ),
                )
            }
        )

    @pytest.fixture
    def sample_rag_api_systems_config(self):
        """Sample rag api systems configuration with API key."""

        return SystemsConfig(
            systems={
                "test_system": RAGAPIConfig(
                    type="rag_api",
                    description="RAG System description",
                    provider="openai",
                    params=LLMAPIParams(
                        base_url="http://RAG-URL",
                        env_file="ENV_FILE",
                        model="my-rag-chatbot",
                        api_key="sk-rag-123",
                    ),
                )
            }
        )

    def test_create_test_execution_plan_flattens_llm_api_system_params(
        self, sample_suite_config, sample_llm_api_systems_config
    ):
        """Test that create_test_execution_plan correctly flattens system parameters."""
        image_availability = {"my-registry/test:latest": True}

        execution_plan = create_test_execution_plan(
            sample_suite_config, sample_llm_api_systems_config, image_availability
        )

        assert len(execution_plan) == 1
        systems_params = execution_plan[0]["systems_params"]
        system_params = systems_params["system_under_test"]

        # Verify the system params are flattened correctly
        assert system_params["type"] == "llm_api"
        assert system_params["description"] == "System description"
        assert system_params["provider"] == "openai"
        assert system_params["base_url"] == "http://URL"
        assert system_params["env_file"] == "ENV_FILE"
        assert system_params["model"] == "gpt-4o-mini"
        assert system_params["api_key"] == "sk-123"

        # Ensure config is not nested
        assert "config" not in system_params

    def test_create_test_execution_plan_flattens_rag_api_system_params(
        self, sample_suite_config, sample_rag_api_systems_config
    ):
        """Test that create_test_execution_plan correctly flattens RAG API system parameters."""
        image_availability = {"my-registry/test:latest": True}

        execution_plan = create_test_execution_plan(
            sample_suite_config, sample_rag_api_systems_config, image_availability
        )

        assert len(execution_plan) == 1
        systems_params = execution_plan[0]["systems_params"]
        system_params = systems_params["system_under_test"]

        # Verify the system params are flattened correctly
        assert system_params["type"] == "rag_api"
        assert system_params["description"] == "RAG System description"
        assert system_params["provider"] == "openai"
        assert system_params["base_url"] == "http://RAG-URL"
        assert system_params["env_file"] == "ENV_FILE"
        assert system_params["model"] == "my-rag-chatbot"
        assert system_params["api_key"] == "sk-rag-123"

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
            "base_url": "http://x",
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
            test_id="test_env_vars",
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
            "base_url": "http://x",
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
            test_id="test_id",
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

    @patch("asqi.workflow.run_container_with_args")
    def test_test_level_env_file(self, mock_run_container, tmp_path, monkeypatch):
        """Test that test-level env_file loads environment variables."""
        test_env_content = "TEST_VAR=test_value\nANOTHER_VAR=another_value\n"
        test_env_path = tmp_path / "test.env"
        test_env_path.write_text(test_env_content)
        monkeypatch.chdir(tmp_path)

        system_params = {
            "type": "llm_api",
            "base_url": "http://localhost:4000",
            "model": "gpt-4o-mini",
        }

        mock_run_container.return_value = {
            "success": True,
            "exit_code": 0,
            "output": '{"success": true}',
            "error": "",
            "container_id": "test_123",
        }

        container_config: ContainerConfig = ContainerConfig()

        _result = execute_single_test(
            test_name="test_with_env_file",
            test_id="test_id",
            image="my-registry/test:latest",
            sut_name="test_system",
            systems_params={"system_under_test": system_params},
            test_params={},
            container_config=container_config,
            env_file="test.env",
            environment=None,
        )

        call_kwargs = mock_run_container.call_args[1]
        assert call_kwargs["environment"]["TEST_VAR"] == "test_value"
        assert call_kwargs["environment"]["ANOTHER_VAR"] == "another_value"

    @patch("asqi.workflow.run_container_with_args")
    def test_test_level_environment_dict(
        self, mock_run_container, tmp_path, monkeypatch
    ):
        """Test that test-level environment dict sets env vars."""
        monkeypatch.chdir(tmp_path)

        system_params = {
            "type": "llm_api",
            "base_url": "http://localhost:4000",
            "model": "gpt-4o-mini",
        }

        mock_run_container.return_value = {
            "success": True,
            "exit_code": 0,
            "output": '{"success": true}',
            "error": "",
            "container_id": "test_123",
        }

        container_config: ContainerConfig = ContainerConfig()

        _result = execute_single_test(
            test_name="test_with_env_dict",
            test_id="test_id",
            image="my-registry/test:latest",
            sut_name="test_system",
            systems_params={"system_under_test": system_params},
            test_params={},
            container_config=container_config,
            env_file=None,
            environment={"CUSTOM_VAR": "custom_value", "ANOTHER": "value2"},
        )

        call_kwargs = mock_run_container.call_args[1]
        assert call_kwargs["environment"]["CUSTOM_VAR"] == "custom_value"
        assert call_kwargs["environment"]["ANOTHER"] == "value2"

    @patch("asqi.workflow.run_container_with_args")
    def test_environment_interpolation(self, mock_run_container, tmp_path, monkeypatch):
        """Test that environment dict supports ${VAR} interpolation."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MY_SECRET_KEY", "interpolated_secret")

        system_params = {
            "type": "llm_api",
            "base_url": "http://localhost:4000",
            "model": "gpt-4o-mini",
        }

        mock_run_container.return_value = {
            "success": True,
            "exit_code": 0,
            "output": '{"success": true}',
            "error": "",
            "container_id": "test_123",
        }

        container_config: ContainerConfig = ContainerConfig()

        _result = execute_single_test(
            test_name="test_interpolation",
            test_id="test_id",
            image="my-registry/test:latest",
            sut_name="test_system",
            systems_params={"system_under_test": system_params},
            test_params={},
            container_config=container_config,
            env_file=None,
            environment={"OPENAI_API_KEY": "${MY_SECRET_KEY}"},
        )

        call_kwargs = mock_run_container.call_args[1]
        assert call_kwargs["environment"]["OPENAI_API_KEY"] == "interpolated_secret"

    @patch("asqi.workflow.run_container_with_args")
    def test_merge_priority_test_over_system(
        self, mock_run_container, tmp_path, monkeypatch
    ):
        """Test that test-level env vars override system-level ones."""
        # Create system-level .env
        system_env_content = "SHARED_VAR=system_value\nSYSTEM_ONLY=sys_value\n"
        system_env_path = tmp_path / "system.env"
        system_env_path.write_text(system_env_content)

        # Create test-level .env
        test_env_content = "SHARED_VAR=test_value\nTEST_ONLY=test_value\n"
        test_env_path = tmp_path / "test.env"
        test_env_path.write_text(test_env_content)

        monkeypatch.chdir(tmp_path)

        system_params = {
            "type": "llm_api",
            "base_url": "http://localhost:4000",
            "model": "gpt-4o-mini",
            "env_file": "system.env",
        }

        mock_run_container.return_value = {
            "success": True,
            "exit_code": 0,
            "output": '{"success": true}',
            "error": "",
            "container_id": "test_123",
        }

        container_config: ContainerConfig = ContainerConfig()

        _result = execute_single_test(
            test_name="test_merge",
            test_id="test_id",
            image="my-registry/test:latest",
            sut_name="test_system",
            systems_params={"system_under_test": system_params},
            test_params={},
            container_config=container_config,
            env_file="test.env",
            environment=None,
        )

        # Verify merge priority: test-level overrides system-level
        call_kwargs = mock_run_container.call_args[1]
        assert call_kwargs["environment"]["SHARED_VAR"] == "test_value"  # Overridden
        assert call_kwargs["environment"]["SYSTEM_ONLY"] == "sys_value"  # From system
        assert call_kwargs["environment"]["TEST_ONLY"] == "test_value"  # From test
