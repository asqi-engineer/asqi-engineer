import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import yaml
from requests import exceptions as requests_exceptions

from asqi.config import ContainerConfig
from asqi.container_manager import (
    _decommission_container,
    _devcontainer_host_path,
    _resolve_abs,
    _shutdown_event,
    check_images_availability,
    docker_client,
    extract_manifest_from_image,
    pull_images,
    run_container_with_args,
)
from asqi.errors import (
    ManifestExtractionError,
    MissingImageError,
    MountExtractionError,
)
from asqi.schemas import Manifest
from docker import errors as docker_errors


@pytest.fixture(autouse=True)
def reset_shutdown_event():
    """Reset the shutdown event before each test to avoid cross-test contamination."""
    _shutdown_event.clear()
    yield
    _shutdown_event.clear()


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

        images = [
            "available:latest",
            "missing:latest",
        ]

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
            "input_systems": [
                {"name": "system_under_test", "type": "llm_api", "required": True}
            ],
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
        container_config = ContainerConfig.from_run_params(
            mem_limit="1g", cpu_quota=100000, network_mode="bridge"
        )

        # Basic success
        result = run_container_with_args(
            image="test:latest", args=["--test"], container_config=container_config
        )
        assert result["success"] is True
        assert result["exit_code"] == 0
        assert result["container_id"] == "test_container_123"
        mock_container.remove.assert_called_with(force=True)

        # Env
        test_env = {"API_KEY": "secret", "MODEL": "gpt-4"}

        run_container_with_args(
            image="test:latest",
            args=["--test"],
            environment=test_env,
            container_config=container_config,
        )
        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["environment"] == test_env

        # Resource/network
        run_container_with_args(
            image="test:latest",
            args=["--test"],
            container_config=container_config,
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
        container_config = ContainerConfig.with_streaming(True)
        result = run_container_with_args(
            image="test:latest", args=["--test"], container_config=container_config
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
        container_config: ContainerConfig = ContainerConfig()
        run_container_with_args(
            image="test:latest",
            args=["--test", "--test-params", '{"__volumes": {"input": "/host/input"}}'],
            container_config=container_config,
        )
        mock_extract_mounts.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["mounts"] == test_mounts

    def test_run_container_labels(self, mock_container_setup):
        """Test that container labels are correctly applied."""
        mock_client, _, mock_extract_mounts = mock_container_setup

        container_config: ContainerConfig = ContainerConfig()
        run_container_with_args(
            image="test:latest",
            args=["--test"],
            container_config=container_config,
            workflow_id="test_workflow",
        )
        mock_extract_mounts.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["labels"] == {
            "workflow_id": "test_workflow",
            "service": "asqi_engineer",
        }

    def test_run_container_labels_empty_workflow(self, mock_container_setup):
        """Test that container labels when workflow_id is empty"""
        mock_client, _, mock_extract_mounts = mock_container_setup

        container_config: ContainerConfig = ContainerConfig()
        run_container_with_args(
            image="test:latest", args=["--test"], container_config=container_config
        )
        mock_extract_mounts.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["labels"] == {"workflow_id": "", "service": "asqi_engineer"}

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
        container_config: ContainerConfig = ContainerConfig()
        result = run_container_with_args(
            image="test:latest", args=["--test"], container_config=container_config
        )
        assert result["success"] is False
        assert expected_error_message in result["error"]

    @pytest.mark.parametrize(
        "exception",
        [
            requests_exceptions.Timeout("Request timeout"),
            requests_exceptions.ReadTimeout("Read timeout"),
            requests_exceptions.ConnectionError("Connection failed"),
        ],
    )
    def test_run_container_timeout_exceptions(self, mock_container_setup, exception):
        """Test handling of requests timeout and connection errors."""
        mock_client, _, _ = mock_container_setup
        mock_client.containers.run.side_effect = exception
        container_config: ContainerConfig = ContainerConfig()
        result = run_container_with_args(
            image="test:latest", args=["--test"], container_config=container_config
        )
        assert result["success"] is False
        assert result["exit_code"] == 137
        assert "timed out" in result["error"]
        assert "test:latest" in result["error"]

    def test_run_container_connection_error_propagated(self, mock_container_setup):
        """Test that ConnectionError is propagated."""
        mock_client, _, _ = mock_container_setup
        mock_client.containers.run.side_effect = ConnectionError("Connection failed")
        container_config: ContainerConfig = ContainerConfig()
        with pytest.raises(ConnectionError, match="Failed to connect to Docker daemon"):
            run_container_with_args(
                image="test:latest", args=["--test"], container_config=container_config
            )

    def test_run_container_wait_timeout_exceptions(self, mock_container_setup):
        """Test handling of timeout exceptions during container.wait() until max_execution_time."""
        mock_client, mock_container, _ = mock_container_setup
        # Simulate container that keeps timing out (never finishes)
        # With our new loop implementation, this will keep retrying until max_execution_time
        mock_container.wait.side_effect = requests_exceptions.Timeout("Wait timeout")

        # Use very short timeout to avoid test hanging (default is 300s)
        container_config = ContainerConfig(timeout_seconds=3)  # 3 second max

        result = run_container_with_args(
            image="test:latest", args=["--test"], container_config=container_config
        )
        assert result["success"] is False
        assert result["exit_code"] == 137
        assert "timed out" in result["error"]
        # Container should be killed after max_execution_time exceeded
        mock_container.kill.assert_called()

    def test_run_container_log_retrieval_timeout_exceptions(self, mock_container_setup):
        """Test handling of timeout exceptions during log retrieval."""
        mock_client, mock_container, _ = mock_container_setup
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = requests_exceptions.Timeout("Log timeout")
        container_config: ContainerConfig = ContainerConfig()
        result = run_container_with_args(
            image="test:latest", args=["--test"], container_config=container_config
        )
        assert result["success"] is True  # Success despite log retrieval failure
        assert result["exit_code"] == 0
        assert result["output"] == ""  # Fallback to empty output

    def test_run_container_execution_edge_cases(self, mock_container_setup):
        """Test edge cases in container execution."""
        mock_client, mock_container, _ = mock_container_setup

        # Non-zero exit code
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.logs.return_value = b"error output"
        container_config: ContainerConfig = ContainerConfig()
        result = run_container_with_args(
            image="test:latest", args=["--test"], container_config=container_config
        )
        assert result["success"] is False
        assert result["exit_code"] == 1

        # Wait API error with successful kill
        mock_container.wait.side_effect = docker_errors.APIError("Wait error")
        container_config: ContainerConfig = ContainerConfig()
        result = run_container_with_args(
            image="test:latest", args=["--test"], container_config=container_config
        )
        mock_container.kill.assert_called_once()
        assert result["success"] is False
        assert "Container execution failed with API error" in result["error"]

        # Wait API error with kill failure
        mock_container.kill.side_effect = docker_errors.APIError("Kill failed")
        container_config: ContainerConfig = ContainerConfig()
        result = run_container_with_args(
            image="test:latest", args=["--test"], container_config=container_config
        )
        assert result["success"] is False

        # Log retrieval error
        mock_container.wait.side_effect = None
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = docker_errors.APIError("Log error")
        container_config: ContainerConfig = ContainerConfig()
        result = run_container_with_args(
            image="test:latest", args=["--test"], container_config=container_config
        )
        assert result["output"] == ""

        # Cleanup failure
        mock_container.logs.side_effect = None
        mock_container.logs.return_value = b"output"
        mock_container.remove.side_effect = docker_errors.APIError("Remove failed")
        container_config: ContainerConfig = ContainerConfig()
        result = run_container_with_args(
            image="test:latest", args=["--test"], container_config=container_config
        )
        assert result["success"] is True  # Should succeed despite cleanup failure

    @patch("asqi.container_manager.logger")
    @patch("asqi.container_manager.docker_client")
    @patch("asqi.container_manager._shutdown_in_progress", new=True)
    def test_no_run_when_shutdown_in_progress(self, mock_docker_client, mock_logger):
        """Ensure no containers are executed when shutdown has started."""
        # Act
        container_config: ContainerConfig = ContainerConfig()
        result = run_container_with_args(
            image="some/image:tag",
            args=["--foo", "bar"],
            container_config=container_config,
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

    def test_run_container_with_name(self, mock_container_setup):
        """Test that container name is passed to Docker when provided."""
        mock_client, mock_container, _ = mock_container_setup
        container_config = ContainerConfig()

        test_name = "test_container_name"
        result = run_container_with_args(
            image="test:latest",
            args=["--test"],
            container_config=container_config,
            name=test_name,
        )

        # Verify the name was passed to containers.run
        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["name"] == test_name
        assert result["success"] is True


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


class TestPullImages:
    @patch("asqi.container_manager.docker_client")
    def test_no_pull_when_all_present(self, mock_docker_client):
        # Setup client that returns images.get without error
        mock_client = MagicMock()
        mock_docker_client.return_value.__enter__.return_value = mock_client
        mock_client.images.get.return_value = MagicMock()

        pull_images(["repo1/img:1", "repo2/img:2"])  # should not raise

        # First check pass called twice
        assert mock_client.images.get.call_count == 2
        # No pull attempted because images_to_pull is empty
        mock_client.images.pull.assert_not_called()

    @patch("asqi.container_manager.docker_client")
    def test_pull_missing_then_verify_success(self, mock_docker_client):
        mock_client_first = MagicMock()
        mock_client_second = MagicMock()

        # First context manager: determine images_to_pull
        # imageA present, imageB missing
        def first_get(name):
            if name == "imageA:latest":
                return MagicMock()
            raise docker_errors.ImageNotFound("not here")

        mock_client_first.images.get.side_effect = first_get

        # Second context manager: perform pull and verify
        # pull ok; get after pull returns object
        mock_client_second.images.pull.return_value = MagicMock()
        mock_client_second.images.get.return_value = MagicMock()

        # Configure context manager to yield different clients sequentially
        mock_cm = mock_docker_client.return_value
        mock_cm.__enter__.side_effect = [mock_client_first, mock_client_second]

        pull_images(["imageA:latest", "imageB:1.2"])

        # Ensure only missing image was pulled
        mock_client_second.images.pull.assert_called_once_with("imageB:1.2")
        # Verification get called
        mock_client_second.images.get.assert_called_with("imageB:1.2")

    @patch("asqi.container_manager.logger")
    @patch("asqi.container_manager.docker_client")
    def test_pull_missing_failure_raises_with_suggestion(
        self, mock_docker_client, mock_logger
    ):
        mock_client_first = MagicMock()
        mock_client_second = MagicMock()

        # First pass: both missing
        mock_client_first.images.get.side_effect = docker_errors.ImageNotFound(
            "missing"
        )

        # Second pass: pull fails via APIError
        mock_client_second.images.pull.side_effect = docker_errors.APIError("denied")
        # Local image list contains a similar tag to suggest
        mock_image = MagicMock()
        mock_image.tags = ["repo/tool:latest", "repo/other:1.0"]
        mock_client_second.images.list.return_value = [mock_image]

        mock_cm = mock_docker_client.return_value
        mock_cm.__enter__.side_effect = [mock_client_first, mock_client_second]

        with pytest.raises(MissingImageError) as exc:
            pull_images(
                ["repo/tool:1.0"]
            )  # will fail to pull, suggestion should pick "repo/tool:latest"

        msg = str(exc.value)
        assert "Container not found: repo/tool:1.0" in msg
        assert "Did you mean:" in msg
        assert "repo/tool:latest" in msg or "repo/other:1.0" in msg
        # Ensure we attempted to list local images for suggestions
        mock_client_second.images.list.assert_called_once()
        # And error was logged
        mock_logger.error.assert_called()


class TestDevcontainerHostPath:
    """Test suite for _devcontainer_host_path function."""

    def test_non_workspaces_path_returned_as_is(self):
        """Test that paths not starting with /workspaces/ are returned as-is."""
        mock_client = MagicMock()

        # Host paths should be returned unchanged
        result = _devcontainer_host_path(mock_client, "/Users/dev/project")
        assert result == "/Users/dev/project"

        result = _devcontainer_host_path(mock_client, "/home/user/code")
        assert result == "/home/user/code"

        # Docker client should not be called for non-workspaces paths
        mock_client.api.inspect_container.assert_not_called()

    @patch("asqi.container_manager.Path")
    def test_not_in_container_returns_path_as_is(self, mock_path_class):
        """Test that paths are returned as-is when not running inside a container."""
        mock_client = MagicMock()

        # Mock Path to simulate /.dockerenv does not exist
        mock_dockerenv = MagicMock()
        mock_dockerenv.exists.return_value = False

        def path_side_effect(path_str):
            if path_str == "/.dockerenv":
                return mock_dockerenv
            # Return a real Path for other calls
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        result = _devcontainer_host_path(mock_client, "/workspaces/myproject/src")

        # Should return the path without attempting container inspection
        assert "/workspaces/myproject/src" in result
        mock_client.api.inspect_container.assert_not_called()

    @patch("asqi.container_manager.Path")
    def test_container_id_from_cgroup(self, mock_path_class):
        """Test that container ID is correctly extracted from /proc/self/cgroup."""
        mock_client = MagicMock()
        container_id = "abc123def456"

        # Mock /.dockerenv exists
        mock_dockerenv = MagicMock()
        mock_dockerenv.exists.return_value = True

        # Mock /proc/self/cgroup with docker container ID
        mock_cgroup = MagicMock()
        mock_cgroup.read_text.return_value = (
            "12:memory:/docker/" + container_id + "\n"
            "11:cpu:/docker/" + container_id + "\n"
        )

        def path_side_effect(path_str):
            if path_str == "/.dockerenv":
                return mock_dockerenv
            elif path_str == "/proc/self/cgroup":
                return mock_cgroup
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        # Mock container inspection
        mock_client.api.inspect_container.return_value = {
            "Mounts": [
                {
                    "Destination": "/workspaces/myproject",
                    "Source": "/host/path/to/project",
                }
            ]
        }

        result = _devcontainer_host_path(
            mock_client, "/workspaces/myproject/src/main.py"
        )

        mock_client.api.inspect_container.assert_called_once_with(container_id)
        assert result == "/host/path/to/project/src/main.py"

    @patch("asqi.container_manager.Path")
    def test_fallback_to_hostname_when_no_docker_in_cgroup(self, mock_path_class):
        """Test fallback to /etc/hostname when cgroup doesn't contain docker info."""
        mock_client = MagicMock()
        hostname_id = "container-hostname-123"

        # Mock /.dockerenv exists
        mock_dockerenv = MagicMock()
        mock_dockerenv.exists.return_value = True

        # Mock /proc/self/cgroup without docker info
        mock_cgroup = MagicMock()
        mock_cgroup.read_text.return_value = "12:memory:/\n11:cpu:/\n"

        # Mock /etc/hostname
        mock_hostname = MagicMock()
        mock_hostname.read_text.return_value = hostname_id + "\n"

        def path_side_effect(path_str):
            if path_str == "/.dockerenv":
                return mock_dockerenv
            elif path_str == "/proc/self/cgroup":
                return mock_cgroup
            elif path_str == "/etc/hostname":
                return mock_hostname
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        mock_client.api.inspect_container.return_value = {
            "Mounts": [
                {
                    "Destination": "/workspaces/project",
                    "Source": "/home/user/project",
                }
            ]
        }

        result = _devcontainer_host_path(mock_client, "/workspaces/project/file.txt")

        mock_client.api.inspect_container.assert_called_once_with(hostname_id)
        assert result == "/home/user/project/file.txt"

    @patch("asqi.container_manager.Path")
    def test_mount_with_target_key(self, mock_path_class):
        """Test that 'Target' key is also supported in mount info."""
        mock_client = MagicMock()
        container_id = "test123"

        mock_dockerenv = MagicMock()
        mock_dockerenv.exists.return_value = True

        mock_cgroup = MagicMock()
        mock_cgroup.read_text.return_value = "0::/docker/" + container_id

        def path_side_effect(path_str):
            if path_str == "/.dockerenv":
                return mock_dockerenv
            elif path_str == "/proc/self/cgroup":
                return mock_cgroup
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        # Use 'Target' instead of 'Destination'
        mock_client.api.inspect_container.return_value = {
            "Mounts": [
                {
                    "Target": "/workspaces/app",
                    "Source": "/var/lib/docker/volumes/app",
                }
            ]
        }

        result = _devcontainer_host_path(mock_client, "/workspaces/app/src/index.ts")

        assert result == "/var/lib/docker/volumes/app/src/index.ts"

    @patch("asqi.container_manager.logger")
    @patch("asqi.container_manager.Path")
    def test_exception_handling_returns_fallback(self, mock_path_class, mock_logger):
        """Test that exceptions are caught and fallback path is returned."""
        mock_client = MagicMock()

        mock_dockerenv = MagicMock()
        mock_dockerenv.exists.return_value = True

        mock_cgroup = MagicMock()
        mock_cgroup.read_text.side_effect = FileNotFoundError(
            "/proc/self/cgroup not found"
        )

        def path_side_effect(path_str):
            if path_str == "/.dockerenv":
                return mock_dockerenv
            elif path_str == "/proc/self/cgroup":
                return mock_cgroup
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        result = _devcontainer_host_path(mock_client, "/workspaces/project/file.py")

        # Should return normalized fallback path
        assert "workspaces/project/file.py" in result
        # Error should be logged
        mock_logger.error.assert_called_once()

    @patch("asqi.container_manager.Path")
    def test_no_matching_mount_returns_fallback(self, mock_path_class):
        """Test that path is returned as-is when no matching mount is found."""
        mock_client = MagicMock()
        container_id = "xyz789"

        mock_dockerenv = MagicMock()
        mock_dockerenv.exists.return_value = True

        mock_cgroup = MagicMock()
        mock_cgroup.read_text.return_value = "0::/docker/" + container_id

        def path_side_effect(path_str):
            if path_str == "/.dockerenv":
                return mock_dockerenv
            elif path_str == "/proc/self/cgroup":
                return mock_cgroup
            return Path(path_str)

        mock_path_class.side_effect = path_side_effect

        # Mounts don't include the path we're looking for
        mock_client.api.inspect_container.return_value = {
            "Mounts": [
                {
                    "Destination": "/workspaces/other-project",
                    "Source": "/host/other",
                }
            ]
        }

        result = _devcontainer_host_path(mock_client, "/workspaces/myproject/src")

        # Should return the original path since no mount matches
        assert "workspaces/myproject/src" in result
