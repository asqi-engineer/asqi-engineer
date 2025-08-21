from unittest.mock import Mock, patch

from docker import errors as docker_errors
from asqi.container_manager import _decommission_container, run_container_with_args


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


class TestRunContainerWithArgs:
    """Tests for running containers with args behavior."""

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
