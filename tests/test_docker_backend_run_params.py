"""Regression tests for Docker-backend run_kwargs assembly (AIP-3936).

Kept in a dedicated module so the assertion lives in a small, well-scoped file
rather than the large legacy ``test_docker_backend.py``.
"""

from unittest.mock import MagicMock, patch

from asqi.backends.docker_backend import run_container_with_args
from asqi.config import ContainerConfig


def test_k8s_only_run_params_stripped_before_docker_run():
    """k8s-only resource keys must not reach docker-py's containers.run().

    ``ContainerConfig.DEFAULT_RUN_PARAMS`` carries ``cpu_request`` (and callers may
    add ``mem_request``) for the Kubernetes backend. docker-py's ``containers.run()``
    rejects unknown kwargs with
    ``TypeError: run() got an unexpected keyword argument 'cpu_request'``, which
    previously crashed every Docker-backend test (0/N passed). The backend must strip
    these keys while leaving the docker-legal resource params intact.
    """
    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = "test_container_123"
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.return_value = b'{"success": true}'
    mock_client.containers.run.return_value = mock_container

    with (
        patch("asqi.backends.docker_backend.docker_client") as mock_docker_client,
        patch("asqi.backends.docker_backend._extract_mounts_from_args") as mock_extract_mounts,
        patch("asqi.backends.docker_backend.create_container_logger"),
    ):
        mock_docker_client.return_value.__enter__.return_value = mock_client
        mock_extract_mounts.return_value = (["--test"], None)

        # Real defaults include cpu_request; add mem_request to cover both keys.
        container_config = ContainerConfig(run_params={**ContainerConfig.DEFAULT_RUN_PARAMS, "mem_request": "256Mi"})
        assert "cpu_request" in container_config.run_params
        assert "mem_request" in container_config.run_params

        run_container_with_args(
            image="test:latest",
            args=["--test"],
            container_config=container_config,
        )

    mock_client.containers.run.assert_called_once()
    call_kwargs = mock_client.containers.run.call_args.kwargs
    # k8s-only keys must be gone...
    assert "cpu_request" not in call_kwargs
    assert "mem_request" not in call_kwargs
    # ...while docker-legal resource params survive unchanged.
    assert call_kwargs["mem_limit"] == "2g"
    assert call_kwargs["cpu_quota"] == 200000
    assert call_kwargs["cpu_period"] == 100000
