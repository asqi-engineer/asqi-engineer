import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import docker
import yaml
from docker import errors as docker_errors

from asqi.logging_config import create_container_logger
from asqi.schemas import Manifest

logger = logging.getLogger(__name__)


class ManifestExtractionError(Exception):
    """Exception raised when manifest extraction fails."""

    def __init__(
        self, message: str, error_type: str, original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


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

    Args:
        images: List of image names to check

    Returns:
        Dictionary mapping image names to availability status
    """
    availability = {}

    with docker_client() as client:
        for image in images:
            try:
                client.images.get(image)
                availability[image] = True
            except docker_errors.ImageNotFound:
                availability[image] = False
            except docker_errors.APIError as e:
                # Log specific Docker API errors but continue checking other images
                logger.warning(f"Docker API error checking image {image}: {e}")
                availability[image] = False
            except ConnectionError as e:
                raise ConnectionError(
                    f"Failed to connect to Docker daemon while checking image {image}: {e}"
                )

    return availability


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


def run_container_with_args(
    image: str,
    args: List[str],
    timeout_seconds: int = 300,
    memory_limit: str = "512m",
    cpu_quota: int = 50000,
    cpu_period: int = 100000,
    environment: Optional[Dict[str, str]] = None,
    stream_logs: bool = False,
    network: str = "host",
) -> Dict[str, Any]:
    """
    Run a Docker container with specified arguments and return results.

    Args:
        image: Docker image to run
        args: Command line arguments to pass to container
        timeout_seconds: Container execution timeout
        memory_limit: Memory limit for container
        cpu_quota: CPU quota for container
        cpu_period: CPU period for container
        environment: Optional dictionary of environment variables to pass to container
        stream_logs: If True, stream logs in real-time
        network: Docker network mode (default: "host")

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
    container_logger = create_container_logger(display_name=image)

    with docker_client() as client:
        container = None
        try:
            # Run container
            logger.info(
                f"Running container for image '{image}' with args: {args} on network: {network}"
            )
            container = client.containers.run(
                image,
                command=args,
                detach=True,
                remove=False,
                network_mode=network,
                mem_limit=memory_limit,
                cpu_period=cpu_period,
                cpu_quota=cpu_quota,
                environment=environment or {},
            )

            result["container_id"] = container.id or ""

            output_lines = []
            if stream_logs:
                try:
                    for log_line in container.logs(stream=True, follow=True):
                        line = log_line.decode("utf-8", errors="replace").rstrip()
                        if line:  # Only process non-empty lines
                            output_lines.append(line)
                            container_logger.info(line)
                except (UnicodeDecodeError, docker_errors.APIError) as e:
                    logger.warning(f"Failed to stream container logs: {e}")

            # Wait for completion
            try:
                exit_status = container.wait(timeout=timeout_seconds)
                result["exit_code"] = exit_status["StatusCode"]
            except docker_errors.APIError as api_error:
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
                    result["output"] = container.logs().decode(
                        "utf-8", errors="replace"
                    )
            except (UnicodeDecodeError, docker_errors.APIError) as e:
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
            result["error"] = (
                f"Container execution timed out after {timeout_seconds}s for image '{image}': {e}"
            )
        except ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to Docker daemon while running image '{image}': {e}"
            )
        finally:
            # Clean up container
            if container:
                try:
                    container.remove(force=True)
                except (docker_errors.APIError, docker_errors.NotFound) as e:
                    logger.warning(f"Failed to remove container during cleanup: {e}")

    return result
