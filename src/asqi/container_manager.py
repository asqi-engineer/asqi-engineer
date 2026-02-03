import json
import logging
import tempfile
import threading
from contextlib import contextmanager
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from docker.types import Mount
from requests import exceptions as requests_exceptions

import docker
from asqi.config import ContainerConfig
from asqi.errors import (
    ManifestExtractionError,
    MissingImageError,
    MountExtractionError,
)
from asqi.logging_config import create_container_logger
from asqi.schemas import Manifest
from docker import errors as docker_errors

logger = logging.getLogger(__name__)

# === Constants ===
INPUT_MOUNT_PATH = Path("/input")
OUTPUT_MOUNT_PATH = Path("/output")
TIMEOUT_EXCEPTIONS = (
    requests_exceptions.Timeout,
    requests_exceptions.ReadTimeout,
    requests_exceptions.ConnectionError,
)

_active_lock = threading.Lock()
_active_containers: set[str] = set()
_shutdown_in_progress = False
_shutdown_event = threading.Event()


@contextmanager
def docker_client():
    """Context manager for Docker client with proper cleanup."""
    client = docker.from_env()
    try:
        yield client
    finally:
        client.close()


def check_images_availability(images: List[str]) -> Dict[str, bool]:
    """
    Check if Docker images are available locally.
    - Tries to fetch each image by exact name:tag.

    Returns:
        Dict mapping image -> True/False indicating availability

    """
    availability: Dict[str, bool] = {}

    # First pass: check availability
    with docker_client() as client:
        for image in images:
            try:
                client.images.get(image)
                availability[image] = True
            except docker_errors.ImageNotFound:
                availability[image] = False
            except docker_errors.APIError as e:
                logger.warning(f"Docker API error checking {image}: {e}")
                availability[image] = False
            except ConnectionError as e:
                raise ConnectionError(f"Failed to connect to Docker daemon: {e}")

    return availability


def pull_images(images: List[str]):
    """
    Pull Docker images from registry if not available locally.

    Args:
        images: List of image references (e.g., 'ubuntu:22.04', 'user/repo:tag')

    Returns:
        None on success

    Raises:
        MissingImageError: If images cannot be pulled (includes alternative suggestions)
        ConnectionError: If unable to connect to Docker daemon
    """
    images_to_pull = []
    # Quick local check to avoid unnecessary pulls
    with docker_client() as client:
        for image in images:
            try:
                client.images.get(image)
            except docker_errors.ImageNotFound:
                images_to_pull.append(image)
            except docker_errors.APIError as e:
                logger.warning(f"Docker API error checking local image {image}: {e}")

    # Attempt to pull only those not present
    if not images_to_pull:
        return

    missing = []
    with docker_client() as client:
        for image in images_to_pull:
            try:
                logger.info(f"Pulling image '{image}' from registry...")
                client.images.pull(image)
                # Verify
                try:
                    client.images.get(image)
                    logger.info(f"Successfully pulled '{image}'")
                except docker_errors.ImageNotFound:
                    logger.error(f"Image '{image}' not found after pull")
                    missing.append(image)
            except docker_errors.APIError as e:
                logger.error(f"Failed to pull image '{image}': {e}")
                missing.append(image)
            except Exception as e:
                logger.error(f"Unexpected error pulling '{image}': {e}")
                missing.append(image)

        if missing:
            try:
                local_images = client.images.list()
                local_tags = [tag for img in local_images for tag in img.tags]
            except docker_errors.APIError as e:
                logger.warning(f"Failed to list local Docker images: {e}")
                local_tags = []
    if not missing:
        return

    msgs = []
    for image in missing:
        repo = image.rsplit(":", 1)[0] if ":" in image else image
        repo_tags = [tag for tag in local_tags if tag.startswith(repo + ":")]

        suggestion = get_close_matches(image, local_tags, n=1)

        if repo_tags:  # different tags
            msg = f"❌ Container not found: {image}\nDid you mean: {repo_tags[0]}"
        elif suggestion:  # similar names
            msg = f"❌ Container not found: {image}\nDid you mean: {suggestion[0]}"
        else:
            msg = f"❌ Container not found: {image}\nNo similar images found."
        msgs.append(msg)
    raise MissingImageError("\n\n".join(msgs))


def extract_manifest_from_image(
    image: str, manifest_path: str = "/app/manifest.yaml"
) -> Optional[Manifest]:
    """
    Extract and parse manifest.yaml from a Docker image.

    Args:
        image: Docker image name
        manifest_path: Path to manifest file inside container

    Returns:
        Parsed Manifest object or None if extraction fails

    Raises:
        ManifestExtractionError: If extraction fails with detailed error information
    """
    with docker_client() as client:
        container = None
        try:
            # Create container without starting it
            try:
                container = client.containers.create(
                    image, command="echo 'manifest extraction'", detach=True
                )
            except docker_errors.ImageNotFound as e:
                raise ManifestExtractionError(
                    f"Docker image '{image}' not found", "IMAGE_NOT_FOUND", e
                )
            except docker_errors.APIError as e:
                raise ManifestExtractionError(
                    f"Docker API error while creating container for image '{image}': {e}",
                    "DOCKER_API_ERROR",
                    e,
                )

            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                local_manifest_path = temp_path / "manifest.yaml"

                try:
                    # Copy manifest from container
                    bits, _ = container.get_archive(manifest_path)
                except docker_errors.NotFound as e:
                    raise ManifestExtractionError(
                        f"Manifest file '{manifest_path}' not found in image '{image}'",
                        "MANIFEST_FILE_NOT_FOUND",
                        e,
                    )
                except docker_errors.APIError as e:
                    raise ManifestExtractionError(
                        f"Docker API error while extracting manifest from image '{image}': {e}",
                        "DOCKER_API_ERROR",
                        e,
                    )

                # Extract tar data
                import io
                import tarfile

                try:
                    tar_stream = io.BytesIO()
                    for chunk in bits:
                        tar_stream.write(chunk)
                    tar_stream.seek(0)

                    # Note: avoid tarfile.extractall used without any validation. Extract only the manifest file
                    with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                        for member in tar.getmembers():
                            if (
                                member.isfile()
                                and not member.name.startswith("/")
                                and ".." not in member.name
                            ):
                                tar.extract(member, temp_path)
                except tarfile.TarError as e:
                    raise ManifestExtractionError(
                        f"Invalid tar archive from image '{image}': {e}",
                        "TAR_EXTRACTION_ERROR",
                        e,
                    )
                except IOError as e:
                    raise ManifestExtractionError(
                        f"I/O error extracting tar archive from image '{image}': {e}",
                        "TAR_IO_ERROR",
                        e,
                    )

                # Read and parse manifest
                if not local_manifest_path.exists():
                    raise ManifestExtractionError(
                        f"Manifest file was not found in extracted archive from image '{image}'",
                        "MANIFEST_FILE_MISSING_AFTER_EXTRACTION",
                    )

                try:
                    with open(local_manifest_path, "r") as f:
                        manifest_data = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    raise ManifestExtractionError(
                        f"Failed to parse YAML in manifest file from image '{image}': {e}",
                        "YAML_PARSING_ERROR",
                        e,
                    )
                except (IOError, OSError) as e:
                    raise ManifestExtractionError(
                        f"Failed to read manifest file from image '{image}': {e}",
                        "FILE_READ_ERROR",
                        e,
                    )

                if manifest_data is None:
                    raise ManifestExtractionError(
                        f"Manifest file from image '{image}' is empty or contains only null values",
                        "EMPTY_MANIFEST_FILE",
                    )

                try:
                    return Manifest(**manifest_data)
                except (TypeError, ValueError) as e:
                    raise ManifestExtractionError(
                        f"Failed to validate manifest schema from image '{image}': {e}",
                        "SCHEMA_VALIDATION_ERROR",
                        e,
                    )

        except ManifestExtractionError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            raise ManifestExtractionError(
                f"Unexpected error while extracting manifest from image '{image}': {e}",
                "UNEXPECTED_ERROR",
                e,
            )
        finally:
            # Clean up container
            if container:
                try:
                    container.remove()
                except (docker_errors.APIError, docker_errors.NotFound) as e:
                    logger.warning(f"Failed to remove container during cleanup: {e}")


def _resolve_abs(p: str) -> str:
    """
    Resolve a given path string to an absolute normalized path.

    - Expands '~' (user home) if present.
    - Converts relative paths to absolute.
    - Ensures normalization (resolves '..' and symlinks).
    - Unlike realpath, it doesn’t require the path to exist.

    Args:
        p: Input path string.

    Returns:
        Absolute normalized path string.
    """
    return str(Path(p).expanduser().resolve())


def _devcontainer_host_path(client, maybe_dev_path: str) -> str:
    """
    Translate a devcontainer path to its corresponding host path if possible.

    - If the path starts with `/workspaces/...`, attempts to resolve it
      to the host machine’s mount path using Docker inspection.
    - Otherwise, assumes it's already a host path and just normalizes it.

    Args:
        client: Docker client used for inspecting container mounts.
        maybe_dev_path: Path string that may belong to the devcontainer.

    Returns:
        Host path string corresponding to the given path, or a normalized fallback.
    """
    try:
        # First resolve to absolute path to handle relative paths
        abs_path = _resolve_abs(maybe_dev_path)

        # Short-circuit if it's clearly a host path (macOS /Users, Windows drive, etc.)
        if not abs_path.startswith("/workspaces/"):
            return abs_path

        # Inspect *this* container, then map Destination -> Source
        cid = Path("/etc/hostname").read_text().strip()
        info = client.api.inspect_container(cid)
        for m in info.get("Mounts", []):
            dest = m.get("Destination") or m.get("Target")
            src = m.get("Source")
            if dest and src and abs_path.startswith(dest):
                rel = abs_path[len(dest) :]
                return _resolve_abs(src + rel)
    except Exception as e:
        logger.error("Failed to resolve devcontainer path '%s': %s", maybe_dev_path, e)
    return _resolve_abs(maybe_dev_path)


def _extract_mounts_from_args(
    client, args: List[str]
) -> Tuple[List[str], Optional[List[Mount]]]:
    """
    Extract and validate volume mount definitions from the '--test-params' CLI argument.

    The '--test-params' JSON may include a special key:
      "__volumes": {
          "input": <host_path>,
          "output": <host_path>
      }

    - The input path is mounted read-only at `INPUT_MOUNT_PATH` (e.g., "/input").
    - The output path is mounted read-write at `OUTPUT_MOUNT_PATH` (e.g., "/output").
    - The '__volumes' key is removed from the final '--test-params' JSON passed to the container.

    Args:
        client: Docker client used to resolve devcontainer → host paths.
        args: List of CLI arguments (potentially containing '--test-params').

    Returns:
        Tuple (new_args, mounts):
          - new_args: The CLI args with a cleaned/updated '--test-params' JSON.
          - mounts: A list of Docker `Mount` objects if volumes were specified, otherwise None.

    Raises:
        MountExtractionError: If '--test-params' is present but malformed, if JSON parsing fails,
            if '__volumes' is invalid, or if mount resolution/creation fails.
            Absence of '--test-params' does not raise and is treated as no mounts.
    """
    if not args:
        return args, None

    new_args = list(args)
    mounts: List[Mount] = []

    try:
        PARAM_FLAGS = ("--test-params", "--generation-params")
        idx = next(i for i, v in enumerate(new_args) if v in PARAM_FLAGS)
        raw = new_args[idx + 1]
        tp = json.loads(raw)

        vols = tp.pop("__volumes", None)
        if vols:
            inp = vols.get("input")
            outp = vols.get("output")

            if inp:
                host_in = _devcontainer_host_path(client, inp)
                mounts.append(
                    Mount(
                        target=str(INPUT_MOUNT_PATH),
                        source=host_in,
                        type="bind",
                        read_only=True,
                    )
                )

            if outp:
                host_out = _devcontainer_host_path(client, outp)
                mounts.append(
                    Mount(
                        target=str(OUTPUT_MOUNT_PATH),
                        source=host_out,
                        type="bind",
                        read_only=False,
                    )
                )

            # write back cleaned test-params
            new_args[idx + 1] = json.dumps(tp)

    except StopIteration:
        return args, None  # no --test-params found is fine
    except Exception as e:
        raise MountExtractionError(f"Failed to extract mounts from args: {e}") from e

    return new_args, (mounts or None)


def run_container_with_args(
    image: str,
    args: List[str],
    container_config: ContainerConfig,
    environment: Optional[Dict[str, str]] = None,
    name: Optional[str] = None,
    workflow_id: str = "",
) -> Dict[str, Any]:
    """
    Run a Docker container with specified arguments and return results.

    Args:
        image: Docker image to run
        args: Command line arguments to pass to container
        container_config: Container execution configurations
        environment: Optional dictionary of environment variables to pass to container
        name: Optional name for the container (will be used as container name in Docker)
        workflow_id: Workflow identifier to uniquely associate the container with a workflow

    Returns:
        Dictionary with execution results including exit_code, output, success, etc.
    """
    result = {
        "success": False,
        "exit_code": -1,
        "output": "",
        "error": "",
        "container_id": "",
    }
    # e.g. my-registry/deepteam:latest -> deepteam
    image_name_only = image.split("/")[-1].split(":")[0]
    container_logger = create_container_logger(display_name=image_name_only)

    with _active_lock:
        if _shutdown_in_progress:
            logger.warning(
                f"Attempting to run container '{image}' during shutdown, skipping..."
            )
            return result
    with docker_client() as client:
        container = None
        try:
            # Run container
            args, mounts = _extract_mounts_from_args(client, args)
            # Container labels
            labels = {"workflow_id": workflow_id, "service": "asqi_engineer"}
            logger.info(f"Running container for image '{image}' with args: {args}")
            if mounts:
                logger.info(f"Mounts: {mounts}")

            # Prepare run parameters
            run_kwargs = {
                "image": image,
                "command": args,
                "environment": environment or {},
                "mounts": mounts,
                "labels": labels,
                **container_config.run_params,
            }

            # Add name if provided
            if name:
                run_kwargs["name"] = name

            container = client.containers.run(**run_kwargs)

            with _active_lock:
                _active_containers.add(container.id)  # type: ignore

            result["container_id"] = container.id or ""
            output_lines = []
            if container_config.stream_logs:
                try:
                    for log_line in container.logs(stream=True, follow=True):
                        line = log_line.decode("utf-8", errors="replace").rstrip()
                        if line:  # Only process non-empty lines
                            output_lines.append(line)
                            container_logger.info(line)
                except (
                    UnicodeDecodeError,
                    docker_errors.APIError,
                    *TIMEOUT_EXCEPTIONS,
                ) as e:
                    logger.warning(f"Failed to stream container logs: {e}")

            # Wait for completion
            try:
                check_interval = 2
                max_execution_time = container_config.timeout_seconds
                elapsed = 0

                while elapsed < max_execution_time:
                    # Wait on shutdown event - wakes immediately if shutdown signaled
                    # Otherwise timeout after check_interval to poll container status
                    if _shutdown_event.wait(timeout=check_interval):
                        logger.info(
                            f"Shutdown detected, stopping container {container.id}"
                        )
                        try:
                            container.kill()
                        except (docker_errors.APIError, docker_errors.NotFound):
                            pass
                        result["exit_code"] = 130  # 128 + SIGINT
                        result["error"] = "Container stopped due to shutdown signal"
                        break

                    # Event timeout expired - check if container has finished
                    try:
                        exit_status = container.wait(timeout=check_interval)
                        result["exit_code"] = exit_status["StatusCode"]
                        break  # Container finished successfully
                    except TIMEOUT_EXCEPTIONS:
                        # Container still running (Docker API timed out), continue waiting
                        elapsed += check_interval
                        continue
                else:
                    # Overall execution time exceeded - kill container immediately
                    try:
                        container.kill()
                    except (docker_errors.APIError, docker_errors.NotFound) as e:
                        logger.warning(f"Failed to kill container {container.id}: {e}")
                    raise TimeoutError(
                        f"Container exceeded timeout of {max_execution_time}s"
                    )

            except docker_errors.APIError as api_error:
                # Docker API error during wait; attempt to kill and report
                try:
                    container.kill()
                except (docker_errors.APIError, docker_errors.NotFound) as e:
                    logger.warning(f"Failed to kill container {container.id}: {e}")
                result["error"] = (
                    f"Container execution failed with API error: {api_error}"
                )
                return result

            # Get output (use streamed output if available, otherwise get all logs)
            try:
                if output_lines:
                    result["output"] = "\n".join(output_lines)
                else:
                    logger.debug(f"container.logs() : {container.logs()}")
                    result["output"] = container.logs().decode(
                        "utf-8", errors="replace"
                    )
            except (
                UnicodeDecodeError,
                docker_errors.APIError,
                *TIMEOUT_EXCEPTIONS,
            ) as e:
                result["output"] = "\n".join(output_lines) if output_lines else ""
                logger.warning(f"Failed to retrieve container logs: {e}")

            result["success"] = result["exit_code"] == 0

        except docker_errors.ImageNotFound as e:
            result["error"] = f"Docker image '{image}' not found: {e}"
        except docker_errors.ContainerError as e:
            result["error"] = f"Container execution failed for image '{image}': {e}"
        except docker_errors.APIError as e:
            result["error"] = f"Docker API error running image '{image}': {e}"
        except TimeoutError as e:
            # Built-in TimeoutError during container start
            result["exit_code"] = 137
            result["error"] = (
                f"Container execution timed out after {container_config.timeout_seconds}s for image '{image}': {e}"
            )
        except TIMEOUT_EXCEPTIONS as e:
            # Network/HTTP timeout or connection error to Docker daemon.
            # Report gracefully without crashing the workflow.
            result["exit_code"] = 137
            result["error"] = (
                f"Container execution timed out after {container_config.timeout_seconds}s for image '{image}': {e}"
            )
        except ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to Docker daemon while running image '{image}': {e}"
            )
        finally:
            _decommission_container(container, container_config)
    return result


def shutdown_containers() -> None:
    """Force-remove any containers that are still tracked as active.

    This is intended to run during atexit or signal handling to ensure
    worker containers do not linger if the main process is interrupted.
    """
    # Make a snapshot under lock to avoid long-held lock during Docker calls
    with _active_lock:
        global _shutdown_in_progress
        _shutdown_in_progress = True
        _shutdown_event.set()  # signal all waiting threads
        active_ids = list(_active_containers)
    if not active_ids:
        return

    with docker_client() as client:
        for cid in active_ids:
            try:
                c = client.containers.get(cid)
                _decommission_container(c)
            except (docker_errors.NotFound, docker_errors.APIError):
                pass


def _decommission_container(
    container,
    container_config: Optional[ContainerConfig] = None,
) -> None:
    if not container:
        return

    # Try graceful stop first
    try:
        container.stop(timeout=1)
    except (docker_errors.APIError, docker_errors.NotFound) as e:
        logger.debug(f"Failed to gracefully stop container: {e}")

    if container_config is not None:
        # Respect config only when provided
        auto_remove = bool(container_config.run_params.get("remove", False))
        if container_config.cleanup_on_finish and not auto_remove:
            try:
                container.remove(force=bool(container_config.cleanup_force))
            except (docker_errors.APIError, docker_errors.NotFound) as e:
                logger.warning(f"Failed to remove container during cleanup: {e}")
    else:
        # Legacy fallback: always force-remove
        try:
            container.remove(force=True)
        except (docker_errors.APIError, docker_errors.NotFound) as e:
            logger.warning(f"Failed to remove container during cleanup: {e}")

    # Remove from active set
    with _active_lock:
        _active_containers.discard(container.id)
