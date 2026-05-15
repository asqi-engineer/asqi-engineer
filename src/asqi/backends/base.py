from typing import Any, Protocol, runtime_checkable

from asqi.config import ContainerConfig
from asqi.schemas import Manifest


@runtime_checkable
class ContainerBackend(Protocol):
    def run(
        self,
        image: str,
        args: list[str],
        container_config: ContainerConfig,
        environment: dict[str, str] | None = None,
        name: str | None = None,
        workflow_id: str = "",
        manifest: Manifest | None = None,
    ) -> dict[str, Any]: ...

    def shutdown(self, workflow_ids: list[str] | None = None) -> None: ...

    def check_images(self, images: list[str]) -> dict[str, bool]: ...

    def pull_images(self, images: list[str]) -> None: ...

    def extract_manifest(
        self, image: str, manifest_path: str = ContainerConfig.MANIFEST_PATH
    ) -> Manifest | None: ...
