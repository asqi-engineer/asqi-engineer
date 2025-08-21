from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docker import errors as docker_errors

from asqi.container_manager import (
    ManifestExtractionError,
    MountExtractionError,
    _resolve_abs,
    check_images_availability,
    docker_client,
    run_container_with_args,
)


class TestResolveAbs:
    """Test suite for _resolve_abs function."""

    def test_resolve_abs_absolute_path(self, tmp_path):
        """Test that absolute paths are returned as-is."""
        test_path = str(tmp_path / "test_file.txt")
        result = _resolve_abs(test_path)
        assert result == test_path

    def test_resolve_abs_relative_path(self, tmp_path, monkeypatch):
        """Test that relative paths are converted to absolute."""
        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        relative_path = "test_file.txt"
        result = _resolve_abs(relative_path)
        expected = str(tmp_path / "test_file.txt")
        assert result == expected

    def test_resolve_abs_home_directory(self):
        """Test that home directory expansion works."""
        test_path = "~/test_file.txt"
        result = _resolve_abs(test_path)

        assert not result.startswith("~")
        assert result.endswith("/test_file.txt")
        assert Path(result).is_absolute()

    def test_resolve_abs_dot_notation(self, tmp_path, monkeypatch):
        """Test that dot notation paths are resolved correctly."""
        monkeypatch.chdir(tmp_path)

        subdir = tmp_path / "subdir"
        subdir.mkdir()

        test_path = "./subdir/../test_file.txt"
        result = _resolve_abs(test_path)
        expected = str(tmp_path / "test_file.txt")
        assert result == expected


class TestExceptionClasses:
    """Test suite for custom exception classes."""

    def test_manifest_extraction_error_basic(self):
        """Test basic ManifestExtractionError construction."""
        error = ManifestExtractionError("Test message", "TEST_ERROR")
        assert str(error) == "Test message"
        assert error.error_type == "TEST_ERROR"
        assert error.original_error is None

    def test_manifest_extraction_error_with_original(self):
        """Test ManifestExtractionError with original exception."""
        original_error = ValueError("Original error")
        error = ManifestExtractionError("Test message", "TEST_ERROR", original_error)

        assert str(error) == "Test message"
        assert error.error_type == "TEST_ERROR"
        assert error.original_error is original_error

    def test_mount_extraction_error(self):
        """Test MountExtractionError construction."""
        error = MountExtractionError("Mount error message")
        assert str(error) == "Mount error message"


class TestDockerClient:
    """Test suite for docker_client context manager."""

    @patch("asqi.container_manager.docker.from_env")
    def test_docker_client_context_manager_success(self, mock_docker_from_env):
        """Test that docker_client context manager works correctly."""
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        with docker_client() as client:
            assert client is mock_client

        # Verify client was closed
        mock_client.close.assert_called_once()

    @patch("asqi.container_manager.docker.from_env")
    def test_docker_client_context_manager_exception(self, mock_docker_from_env):
        """Test that docker_client closes client even when exception occurs."""
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        with pytest.raises(ValueError):
            with docker_client() as client:
                assert client is mock_client
                raise ValueError("Test exception")

        # Verify client was still closed
        mock_client.close.assert_called_once()


class TestCheckImagesAvailability:
    """Test suite for check_images_availability function."""

    @patch("asqi.container_manager.docker_client")
    def test_check_images_availability_all_available(self, mock_docker_client):
        """Test when all images are available."""
        mock_client = MagicMock()
        mock_docker_client.return_value.__enter__.return_value = mock_client

        # Mock successful image retrieval
        mock_client.images.get.return_value = MagicMock()

        images = ["image1:latest", "image2:latest"]
        result = check_images_availability(images)

        expected = {"image1:latest": True, "image2:latest": True}
        assert result == expected
        assert mock_client.images.get.call_count == 2

    @patch("asqi.container_manager.docker_client")
    def test_check_images_availability_mixed(self, mock_docker_client):
        """Test when some images are available and some are not."""
        mock_client = MagicMock()
        mock_docker_client.return_value.__enter__.return_value = mock_client

        def mock_get_image(image_name):
            if image_name == "available:latest":
                return MagicMock()
            else:
                raise docker_errors.ImageNotFound("Image not found")

        mock_client.images.get.side_effect = mock_get_image

        images = ["available:latest", "missing:latest"]
        result = check_images_availability(images)

        expected = {"available:latest": True, "missing:latest": False}
        assert result == expected

    @patch("asqi.container_manager.docker_client")
    def test_check_images_availability_api_error(self, mock_docker_client):
        """Test handling of Docker API errors."""
        mock_client = MagicMock()
        mock_docker_client.return_value.__enter__.return_value = mock_client

        mock_client.images.get.side_effect = docker_errors.APIError("API Error")

        images = ["problem:latest"]
        result = check_images_availability(images)

        expected = {"problem:latest": False}
        assert result == expected

    @patch("asqi.container_manager.docker_client")
    def test_check_images_availability_connection_error(self, mock_docker_client):
        """Test that ConnectionError is propagated."""
        mock_client = MagicMock()
        mock_docker_client.return_value.__enter__.return_value = mock_client

        mock_client.images.get.side_effect = ConnectionError("Connection failed")

        images = ["test:latest"]

        with pytest.raises(ConnectionError, match="Failed to connect to Docker daemon"):
            check_images_availability(images)

    @patch("asqi.container_manager.docker_client")
    def test_check_images_availability_empty_list(self, mock_docker_client):
        """Test with empty image list."""
        result = check_images_availability([])
        assert result == {}


class TestRunContainerWithArgs:
    """Test suite for run_container_with_args function (moved from test_environment_variables.py)."""

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
        result = run_container_with_args(
            image="test:latest", args=["--test", "arg"], environment=test_env
        )

        # Verify Docker container was created with correct environment
        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]

        assert "environment" in call_kwargs
        assert call_kwargs["environment"] == test_env

        # Verify result
        assert result["success"] is True
        assert result["exit_code"] == 0

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
        result = run_container_with_args(image="test:latest", args=["--test", "arg"])

        # Verify Docker container was created with empty environment
        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]

        assert "environment" in call_kwargs
        assert call_kwargs["environment"] == {}

        # Verify result
        assert result["success"] is True
        assert result["exit_code"] == 0

    @patch("asqi.container_manager.docker_client")
    def test_run_container_with_args_image_not_found(self, mock_docker_client):
        """Test handling of ImageNotFound error."""
        mock_client = MagicMock()
        mock_docker_client.return_value.__enter__.return_value = mock_client

        mock_client.containers.run.side_effect = docker_errors.ImageNotFound(
            "Image not found"
        )

        result = run_container_with_args(image="missing:latest", args=["--test"])

        assert result["success"] is False
        assert "not found" in result["error"]

    @patch("asqi.container_manager.docker_client")
    def test_run_container_with_args_api_error(self, mock_docker_client):
        """Test handling of Docker API error."""
        mock_client = MagicMock()
        mock_docker_client.return_value.__enter__.return_value = mock_client

        mock_client.containers.run.side_effect = docker_errors.APIError("API Error")

        result = run_container_with_args(image="test:latest", args=["--test"])

        assert result["success"] is False
        assert "Docker API error" in result["error"]

    @patch("asqi.container_manager.docker_client")
    def test_run_container_with_args_connection_error(self, mock_docker_client):
        """Test that ConnectionError is propagated."""
        mock_client = MagicMock()
        mock_docker_client.return_value.__enter__.return_value = mock_client

        mock_client.containers.run.side_effect = ConnectionError("Connection failed")

        with pytest.raises(ConnectionError, match="Failed to connect to Docker daemon"):
            run_container_with_args(image="test:latest", args=["--test"])
