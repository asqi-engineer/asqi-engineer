import importlib.util
import logging
import os

from asqi.backends.base import ContainerBackend
from asqi.backends.docker_backend import DockerBackend
from asqi.backends.kubernetes_backend import KubernetesBackend

__all__ = ["ContainerBackend", "DockerBackend", "KubernetesBackend", "create_backend"]

logger = logging.getLogger(__name__)

_VALID_BACKENDS = ("docker", "k8s")


def create_backend(backend: str | None = None, namespace: str | None = None) -> ContainerBackend:
    """Create the appropriate ContainerBackend based on the ``RUN_BACKEND`` env var.

    Reads the provided backend value or ``RUN_BACKEND`` environment variable
    (case-insensitive, default: ``docker``) and constructs the matching backend:

    - ``docker`` → :class:`~asqi.backends.docker_backend.DockerBackend`
    - ``k8s``    → :class:`~asqi.backends.kubernetes_backend.KubernetesBackend`
      (namespace taken from the provided namespace or ``K8S_NAMESPACE``, default: ``"default"``;
      requires ``pip install 'asqi-engineer[k8s]'``)

    Raises:
        ValueError: If the resolved backend value is not one of ``docker`` or ``k8s``.
        ImportError: If ``k8s`` is selected but the ``kubernetes`` package is not installed.

    Returns:
        A :class:`ContainerBackend` instance.
    """
    raw = backend if backend is not None else os.environ.get("RUN_BACKEND", "docker")
    backend_key = raw.strip().lower()

    if backend_key == "k8s":
        if importlib.util.find_spec("kubernetes") is None:
            raise ImportError(
                "RUN_BACKEND='k8s' requires the Kubernetes Python client. "
                "Install asqi-engineer[k8s] or include the kubernetes package in the runner image."
            )

        selected_namespace = namespace if namespace is not None else os.environ.get("K8S_NAMESPACE", "default")
        logger.info("RUN_BACKEND=k8s — using KubernetesBackend (namespace=%s)", selected_namespace)
        return KubernetesBackend(namespace=selected_namespace)

    if backend_key != "docker":
        raise ValueError(f"Unknown RUN_BACKEND={raw!r} (valid: {', '.join(_VALID_BACKENDS)})")

    logger.debug("RUN_BACKEND=%s — using DockerBackend", backend_key)
    return DockerBackend()
