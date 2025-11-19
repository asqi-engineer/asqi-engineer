import os
import sys
import types
from types import SimpleNamespace


def _install_docker_stub():
    """Provide a lightweight docker module stub for tests."""
    if "docker" in sys.modules:
        return

    docker_stub = types.ModuleType("docker")
    errors_stub = types.ModuleType("docker.errors")
    types_stub = types.ModuleType("docker.types")

    class APIError(Exception):
        pass

    class ImageNotFound(Exception):
        pass

    class NotFound(ImageNotFound):
        pass

    class ContainerError(Exception):
        def __init__(self, container, exit_status, command, image, stderr):
            super().__init__(stderr)
            self.container = container
            self.exit_status = exit_status
            self.command = command
            self.image = image
            self.stderr = stderr

    errors_stub.APIError = APIError
    errors_stub.ImageNotFound = ImageNotFound
    errors_stub.NotFound = NotFound
    errors_stub.ContainerError = ContainerError

    class Mount:
        def __init__(self, *args, **kwargs):
            self.source = kwargs.get("source")
            self.target = kwargs.get("target")

    types_stub.Mount = Mount

    def from_env():
        client = SimpleNamespace(
            images=SimpleNamespace(
                get=lambda *args, **kwargs: SimpleNamespace(),
                pull=lambda *args, **kwargs: None,
                list=lambda: [],
            ),
            close=lambda: None,
        )
        return client

    docker_stub.from_env = from_env
    docker_stub.errors = errors_stub
    docker_stub.types = types_stub

    sys.modules["docker"] = docker_stub
    sys.modules["docker.errors"] = errors_stub
    sys.modules["docker.types"] = types_stub


_install_docker_stub()


def _install_dbos_stub():
    """Provide a lightweight dbos module stub for tests."""
    if "dbos" in sys.modules:
        return

    dbos_stub = types.ModuleType("dbos")

    class _DBOS:
        workflow_id = "test-workflow"
        logger = SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
            error=lambda *args, **kwargs: None,
            debug=lambda *args, **kwargs: None,
        )

        def __init__(self, config=None):
            self.config = config or {}

        @staticmethod
        def launch():
            return None

        @staticmethod
        def start_workflow(*args, **kwargs):
            return "workflow-stub"

        @staticmethod
        def step(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

        @staticmethod
        def workflow(*args, **kwargs):
            def decorator(func):
                return func

            return decorator

    class _DBOSConfig:
        pass

    class _Queue:
        def __init__(self, name, concurrency=1):
            self.name = name
            self.concurrency = concurrency

        def enqueue(self, *args, **kwargs):
            return SimpleNamespace(
                get_result=lambda: None,
                get_workflow_id=lambda: "workflow-stub",
            )

    dbos_stub.DBOS = _DBOS
    dbos_stub.DBOSConfig = _DBOSConfig
    dbos_stub.Queue = _Queue

    sys.modules["dbos"] = dbos_stub


_install_dbos_stub()

# Ensure the environment variable exists before importing modules under test
os.environ.setdefault(
    "DBOS_DATABASE_URL", "postgresql://testuser:testpass@example.com:5432/test_db"
)
