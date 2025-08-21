import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from docker import errors as docker_errors

from asqi.container_manager import (
    ManifestExtractionError,
    MountExtractionError,
    _resolve_abs,
    check_images_availability,
    docker_client,
    extract_manifest_from_image,
    run_container_with_args,
)
from asqi.schemas import Manifest


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


class TestExtractManifestFromImage:
    """Test suite for extract_manifest_from_image function."""

    @pytest.fixture
    def sample_manifest_data(self):
        """Sample manifest YAML data for testing."""
        return {
            "name": "test_container",
            "version": "1.0",
            "description": "Test container",
            "supported_suts": [{"type": "llm_api", "required_config": ["model"]}],
            "input_schema": [
                {"name": "test_param", "type": "string", "required": False}
            ],
            "output_metrics": [
                {"name": "success", "type": "boolean", "description": "Test success"}
            ],
        }

    @pytest.fixture
    def mock_docker_setup(self):
        """Reusable Docker client and container mock setup."""
        with patch("asqi.container_manager.docker_client") as mock_docker_client:
            mock_client = MagicMock()
            mock_container = MagicMock()
            mock_client.containers.create.return_value = mock_container
            mock_docker_client.return_value.__enter__.return_value = mock_client
            yield mock_client, mock_container

    def create_tar_archive(self, manifest_content: str) -> bytes:
        """Create a mock tar archive containing manifest.yaml."""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            manifest_bytes = manifest_content.encode("utf-8")
            manifest_info = tarfile.TarInfo(name="manifest.yaml")
            manifest_info.size = len(manifest_bytes)
            tar.addfile(manifest_info, io.BytesIO(manifest_bytes))
        tar_buffer.seek(0)
        return tar_buffer.read()

    def test_extract_manifest_success(self, mock_docker_setup, sample_manifest_data):
        """Test successful manifest extraction and parsing."""
        mock_client, mock_container = mock_docker_setup
        manifest_yaml = yaml.dump(sample_manifest_data)
        tar_data = self.create_tar_archive(manifest_yaml)
        mock_container.get_archive.return_value = (iter([tar_data]), {})
        result = extract_manifest_from_image("test:latest")

        assert isinstance(result, Manifest)
        assert result.name == "test_container"
        mock_container.get_archive.assert_called_once_with("/app/manifest.yaml")
        mock_container.remove.assert_called_once()

    def test_extract_manifest_custom_path(
        self, mock_docker_setup, sample_manifest_data
    ):
        """Test manifest extraction with custom manifest path."""
        mock_client, mock_container = mock_docker_setup
        tar_data = self.create_tar_archive(yaml.dump(sample_manifest_data))
        mock_container.get_archive.return_value = (iter([tar_data]), {})
        result = extract_manifest_from_image("test:latest", "/custom/manifest.yaml")
        assert isinstance(result, Manifest)
        mock_container.get_archive.assert_called_once_with("/custom/manifest.yaml")

    @pytest.mark.parametrize(
        "exception,expected_error_type,expected_message",
        [
            (
                docker_errors.ImageNotFound("Image not found"),
                "IMAGE_NOT_FOUND",
                "not found",
            ),
            (
                docker_errors.APIError("API Error"),
                "DOCKER_API_ERROR",
                "API error while creating container",
            ),
            (RuntimeError("Unexpected"), "UNEXPECTED_ERROR", "Unexpected error"),
        ],
    )
    def test_extract_manifest_container_creation_errors(
        self, mock_docker_setup, exception, expected_error_type, expected_message
    ):
        """Test various container creation errors."""
        mock_client, mock_container = mock_docker_setup
        mock_client.containers.create.side_effect = exception

        with pytest.raises(ManifestExtractionError) as exc_info:
            extract_manifest_from_image("test:latest")

        assert exc_info.value.error_type == expected_error_type
        assert expected_message in str(exc_info.value)

    @pytest.mark.parametrize(
        "exception,expected_error_type,expected_message",
        [
            (
                docker_errors.NotFound("File not found"),
                "MANIFEST_FILE_NOT_FOUND",
                "not found in image",
            ),
            (
                docker_errors.APIError("Extract API Error"),
                "DOCKER_API_ERROR",
                "API error while extracting manifest",
            ),
        ],
    )
    def test_extract_manifest_archive_extraction_errors(
        self, mock_docker_setup, exception, expected_error_type, expected_message
    ):
        """Test various archive extraction errors."""
        mock_client, mock_container = mock_docker_setup
        mock_container.get_archive.side_effect = exception

        with pytest.raises(ManifestExtractionError) as exc_info:
            extract_manifest_from_image("test:latest")

        assert exc_info.value.error_type == expected_error_type
        assert expected_message in str(exc_info.value)
        mock_container.remove.assert_called_once()

    def test_extract_manifest_tar_processing_errors(self, mock_docker_setup):
        """Test tar processing errors."""
        mock_client, mock_container = mock_docker_setup

        # Test invalid tar data
        mock_container.get_archive.return_value = (iter([b"invalid_tar_data"]), {})
        with pytest.raises(ManifestExtractionError) as exc_info:
            extract_manifest_from_image("test:latest")
        assert exc_info.value.error_type == "TAR_EXTRACTION_ERROR"

        # Test I/O error during tar processing
        mock_container.get_archive.return_value = (iter([b"some_data"]), {})
        with patch("tarfile.open", side_effect=IOError("I/O Error")):
            with pytest.raises(ManifestExtractionError) as exc_info:
                extract_manifest_from_image("test:latest")
            assert exc_info.value.error_type == "TAR_IO_ERROR"

    def test_extract_manifest_file_processing_errors(
        self, mock_docker_setup, sample_manifest_data
    ):
        """Test file processing errors."""
        mock_client, mock_container = mock_docker_setup

        # Test empty tar
        empty_tar = io.BytesIO()
        with tarfile.open(fileobj=empty_tar, mode="w"):
            pass
        empty_tar.seek(0)
        mock_container.get_archive.return_value = (iter([empty_tar.read()]), {})

        with pytest.raises(ManifestExtractionError) as exc_info:
            extract_manifest_from_image("test:latest")
        assert exc_info.value.error_type == "MANIFEST_FILE_MISSING_AFTER_EXTRACTION"

        # Test YAML parsing error
        invalid_yaml = "invalid: yaml: content: [unclosed"
        mock_container.get_archive.return_value = (
            iter([self.create_tar_archive(invalid_yaml)]),
            {},
        )

        with pytest.raises(ManifestExtractionError) as exc_info:
            extract_manifest_from_image("test:latest")
        assert exc_info.value.error_type == "YAML_PARSING_ERROR"

        # Test empty file
        mock_container.get_archive.return_value = (
            iter([self.create_tar_archive("")]),
            {},
        )

        with pytest.raises(ManifestExtractionError) as exc_info:
            extract_manifest_from_image("test:latest")
        assert exc_info.value.error_type == "EMPTY_MANIFEST_FILE"

        # Test file read error
        tar_data = self.create_tar_archive(yaml.dump(sample_manifest_data))
        mock_container.get_archive.return_value = (iter([tar_data]), {})

        with patch("builtins.open", side_effect=IOError("File read error")):
            with pytest.raises(ManifestExtractionError) as exc_info:
                extract_manifest_from_image("test:latest")
            assert exc_info.value.error_type == "FILE_READ_ERROR"

    def test_extract_manifest_schema_validation_error(self, mock_docker_setup):
        """Test schema validation errors."""
        mock_client, mock_container = mock_docker_setup

        invalid_manifest = {"invalid_field": "missing required fields"}
        tar_data = self.create_tar_archive(yaml.dump(invalid_manifest))
        mock_container.get_archive.return_value = (iter([tar_data]), {})

        with pytest.raises(ManifestExtractionError) as exc_info:
            extract_manifest_from_image("test:latest")

        assert exc_info.value.error_type == "SCHEMA_VALIDATION_ERROR"

    def test_extract_manifest_cleanup_error(
        self, mock_docker_setup, sample_manifest_data
    ):
        """Test that cleanup errors are handled gracefully."""
        mock_client, mock_container = mock_docker_setup

        # Setup successful extraction but failing cleanup
        tar_data = self.create_tar_archive(yaml.dump(sample_manifest_data))
        mock_container.get_archive.return_value = (iter([tar_data]), {})
        mock_container.remove.side_effect = docker_errors.APIError("Cleanup failed")

        # Should still succeed despite cleanup error
        result = extract_manifest_from_image("test:latest")
        assert isinstance(result, Manifest)
        mock_container.remove.assert_called_once()


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
