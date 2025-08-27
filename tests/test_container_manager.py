import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml
from docker import errors as docker_errors

from asqi.container_manager import (
    ManifestExtractionError,
    MissingImageException,
    MountExtractionError,
    _resolve_abs,
    check_images_availabilty,
    docker_client,
    extract_manifest_from_image,
    run_container_with_args,
)
from asqi.container_manager import _decommission_container
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
    """Test suite for check_images_availabilty function."""

    @patch("asqi.container_manager.docker_client")
    def test_check_images_availability_all_available(self, mock_docker_client):
        """Test when all images are available."""
        mock_client = MagicMock()
        mock_docker_client.return_value.__enter__.return_value = mock_client

        # Mock successful image retrieval
        mock_client.images.get.return_value = MagicMock()

        images = ["image1:latest", "image2:latest"]
        result = check_images_availabilty(images)

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

        # Local images: one correct + used for suggestion
        mock_client.images.list.return_value = [
            MagicMock(tags=["availabel:latest"]),  # typo image in local list
            MagicMock(tags=["available:latest"]),  # correct local image
        ]

        images = [
            "available:latest",
            "availabel:latest",
            "available:second",
            "missing:latest",
        ]

        with pytest.raises(MissingImageException) as excinfo:
            check_images_availabilty(images)

        message = str(excinfo.value)

        # ✅ Correct image should not appear in error message
        assert "❌ Container not found: available:latest" not in message

        # ❌ Typo case: should suggest "available:latest"
        assert "❌ Container not found: availabel:latest" in message
        assert "Did you mean: available:latest" in message

        # ❌ Wrong tag case: should suggest "available:latest"
        assert "❌ Container not found: available:second" in message
        assert "Did you mean: available:latest" in message

        # ❌ Completely missing should say no similar images
        assert "❌ Container not found: missing:latest" in message
        assert "No similar images found." in message

    @patch("asqi.container_manager.docker_client")
    def test_check_images_availability_api_error(self, mock_docker_client):
        """Test handling of Docker API errors."""
        mock_client = MagicMock()
        mock_docker_client.return_value.__enter__.return_value = mock_client

        mock_client.images.get.side_effect = docker_errors.APIError("API Error")

        images = ["problem:latest"]
        result = check_images_availabilty(images)

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
            check_images_availabilty(images)

    @patch("asqi.container_manager.docker_client")
    def test_check_images_availability_empty_list(self, mock_docker_client):
        """Test with empty image list."""
        result = check_images_availabilty([])
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
    """Test suite for run_container_with_args function."""

    @pytest.fixture
    def mock_container_setup(self):
        """Reusable Docker client and container mock setup."""
        with (
            patch("asqi.container_manager.docker_client") as mock_docker_client,
            patch(
                "asqi.container_manager._extract_mounts_from_args"
            ) as mock_extract_mounts,
            patch("asqi.container_manager.create_container_logger"),
        ):
            mock_client = MagicMock()
            mock_container = MagicMock()
            mock_container.id = "test_container_123"
            mock_container.wait.return_value = {"StatusCode": 0}
            mock_container.logs.return_value = b'{"success": true}'
            mock_client.containers.run.return_value = mock_container
            mock_docker_client.return_value.__enter__.return_value = mock_client
            mock_extract_mounts.return_value = (["--test"], None)
            yield mock_client, mock_container, mock_extract_mounts

    def test_run_container_success(self, mock_container_setup):
        """Test successful container execution."""
        mock_client, mock_container, _ = mock_container_setup

        # Basic success
        result = run_container_with_args(image="test:latest", args=["--test"])
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert result["container_id"] == "test_container_123"
        mock_container.remove.assert_called_with(force=True)

        # Env
        test_env = {"API_KEY": "secret", "MODEL": "gpt-4"}
        run_container_with_args(
            image="test:latest", args=["--test"], environment=test_env
        )
        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["environment"] == test_env

        # Resource/network
        run_container_with_args(
            image="test:latest",
            args=["--test"],
            memory_limit="1g",
            cpu_quota=100000,
            network="bridge",
        )
        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["mem_limit"] == "1g"
        assert call_kwargs["cpu_quota"] == 100000
        assert call_kwargs["network_mode"] == "bridge"

    def test_run_container_streaming_logs(self, mock_container_setup):
        """Test log streaming functionality and error handling."""
        mock_client, mock_container, mock_extract_mounts = mock_container_setup
        log_lines = [b"Line 1\n", b"Line 2\n"]
        mock_container.logs.return_value = iter(log_lines)
        result = run_container_with_args(
            image="test:latest", args=["--test"], stream_logs=True
        )
        mock_container.logs.assert_called_with(stream=True, follow=True)
        assert "Line 1" in result["output"] and "Line 2" in result["output"]

    def test_run_container_volume_mounting(self, mock_container_setup):
        """Test that volume mounts are correctly extracted and applied."""
        mock_client, _, mock_extract_mounts = mock_container_setup
        from docker.types import Mount

        test_mounts = [
            Mount(target="/input", source="/host/input", type="bind", read_only=True)
        ]
        mock_extract_mounts.return_value = (["--test"], test_mounts)
        run_container_with_args(
            image="test:latest",
            args=["--test", "--test-params", '{"__volumes": {"input": "/host/input"}}'],
        )
        mock_extract_mounts.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["mounts"] == test_mounts

    @pytest.mark.parametrize(
        "exception,expected_error_message",
        [
            (docker_errors.ImageNotFound("Image not found"), "not found"),
            (
                docker_errors.ContainerError(MagicMock(), 1, "cmd", "image", "error"),
                "Container execution failed",
            ),
            (docker_errors.APIError("API Error"), "Docker API error"),
            (TimeoutError("Timeout"), "timed out"),
        ],
    )
    def test_run_container_exception_handling(
        self, mock_container_setup, exception, expected_error_message
    ):
        """Test container run exception handling."""
        mock_client, _, _ = mock_container_setup
        mock_client.containers.run.side_effect = exception
        result = run_container_with_args(image="test:latest", args=["--test"])
        assert result["success"] is False
        assert expected_error_message in result["error"]

    def test_run_container_connection_error_propagated(self, mock_container_setup):
        """Test that ConnectionError is propagated."""
        mock_client, _, _ = mock_container_setup
        mock_client.containers.run.side_effect = ConnectionError("Connection failed")
        with pytest.raises(ConnectionError, match="Failed to connect to Docker daemon"):
            run_container_with_args(image="test:latest", args=["--test"])

    def test_run_container_execution_edge_cases(self, mock_container_setup):
        """Test edge cases in container execution."""
        mock_client, mock_container, _ = mock_container_setup

        # Non-zero exit code
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.logs.return_value = b"error output"
        result = run_container_with_args(image="test:latest", args=["--test"])
        assert result["success"] is False
        assert result["exit_code"] == 1

        # Wait API error with successful kill
        mock_container.wait.side_effect = docker_errors.APIError("Wait error")
        result = run_container_with_args(image="test:latest", args=["--test"])
        mock_container.kill.assert_called_once()
        assert result["success"] is False
        assert "Container execution failed with API error" in result["error"]

        # Wait API error with kill failure
        mock_container.kill.side_effect = docker_errors.APIError("Kill failed")
        result = run_container_with_args(image="test:latest", args=["--test"])
        assert result["success"] is False

        # Log retrieval error
        mock_container.wait.side_effect = None
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = docker_errors.APIError("Log error")
        result = run_container_with_args(image="test:latest", args=["--test"])
        assert result["output"] == ""

        # Cleanup failure
        mock_container.logs.side_effect = None
        mock_container.logs.return_value = b"output"
        mock_container.remove.side_effect = docker_errors.APIError("Remove failed")
        result = run_container_with_args(image="test:latest", args=["--test"])
        assert result["success"] is True  # Should succeed despite cleanup failure

    @patch("asqi.container_manager.logger")
    @patch("asqi.container_manager.docker_client")
    @patch("asqi.container_manager._shutdown_in_progress", new=True)
    def test_no_run_when_shutdown_in_progress(self, mock_docker_client, mock_logger):
        """Ensure no containers are executed when shutdown has started."""
        # Act
        result = run_container_with_args(
            image="some/image:tag",
            args=["--foo", "bar"],
            timeout_seconds=1,
        )

        # Assert: docker_client context manager must not be invoked at all
        mock_docker_client.assert_not_called()

        # Assert: the function returns the default 'skipped' result
        assert result == {
            "success": False,
            "exit_code": -1,
            "output": "",
            "error": "",
            "container_id": "",
        }

        # And a warning was logged about skipping due to shutdown
        mock_logger.warning.assert_called()


class TestDecommissionContainer:
    """Test decommissioning of containers."""

    def test_decommission_container_no_container(self):
        """Test decommissioning when container is None."""
        _decommission_container(None)  # No exception should be raised

    @patch("asqi.container_manager._active_lock")
    @patch("asqi.container_manager._active_containers", new_callable=set)
    def test_decommission_container_success(
        self, mock_active_containers, mock_active_lock
    ):
        """Test successful decommissioning of a container."""
        mock_container = Mock()
        mock_container.id = "test_container"

        mock_active_containers.add(mock_container.id)

        _decommission_container(mock_container)

        mock_container.stop.assert_called_once_with(timeout=1)
        mock_container.remove.assert_called_once_with(force=True)
        assert mock_container.id not in mock_active_containers

    @patch("asqi.container_manager.logger")
    @patch("asqi.container_manager._active_lock")
    @patch("asqi.container_manager._active_containers", new_callable=set)
    def test_decommission_container_api_error(
        self, mock_active_containers, mock_active_lock, mock_logger
    ):
        """Test decommissioning when container removal raises APIError."""
        mock_container = Mock()
        mock_container.id = "test_container"

        mock_active_containers.add(mock_container.id)
        mock_container.remove.side_effect = docker_errors.APIError("API Error")

        _decommission_container(mock_container)

        mock_container.stop.assert_called_once_with(timeout=1)
        mock_container.remove.assert_called_once_with(force=True)
        mock_logger.warning.assert_called_once_with(
            "Failed to remove container during cleanup: API Error"
        )
        assert mock_container.id not in mock_active_containers

    @patch("asqi.container_manager.logger")
    @patch("asqi.container_manager._active_lock")
    @patch("asqi.container_manager._active_containers", new_callable=set)
    def test_decommission_container_not_found_error(
        self, mock_active_containers, mock_active_lock, mock_logger
    ):
        """Test decommissioning when container removal raises NotFound error."""
        mock_container = Mock()
        mock_container.id = "test_container"

        mock_active_containers.add(mock_container.id)
        mock_container.remove.side_effect = docker_errors.NotFound("Not Found")

        _decommission_container(mock_container)

        mock_container.stop.assert_called_once_with(timeout=1)
        mock_container.remove.assert_called_once_with(force=True)
        mock_logger.warning.assert_called_once_with(
            "Failed to remove container during cleanup: Not Found"
        )
        assert mock_container.id not in mock_active_containers

    @patch("asqi.container_manager.logger")
    @patch("asqi.container_manager._active_lock")
    @patch("asqi.container_manager._active_containers", new_callable=set)
    def test_decommission_container_stop_exception(
        self, mock_active_containers, mock_active_lock, mock_logger
    ):
        """Test decommissioning when container stop raises a general exception."""
        mock_container = Mock()
        mock_container.id = "test_container"

        mock_active_containers.add(mock_container.id)
        mock_container.stop.side_effect = docker_errors.APIError("Stop failed")

        _decommission_container(mock_container)

        mock_container.stop.assert_called_once_with(timeout=1)
        mock_container.remove.assert_called_once_with(force=True)
        mock_logger.debug.assert_called_once_with(
            "Failed to gracefully stop container: Stop failed"
        )
        mock_logger.warning.assert_not_called()  # Ensure no log generated for stop failure
        assert mock_container.id not in mock_active_containers
