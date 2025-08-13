import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import docker
import yaml
from docker import errors as docker_errors

from asqi.schemas import Manifest


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
            except (docker_errors.APIError, Exception):
                availability[image] = False

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
    """
    with docker_client() as client:
        container = None
        try:
            # Create container without starting it
            container = client.containers.create(
                image, command="echo 'manifest extraction'", detach=True
            )

            # Create temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                local_manifest_path = temp_path / "manifest.yaml"

                try:
                    # Copy manifest from container
                    bits, _ = container.get_archive(manifest_path)
                except docker_errors.NotFound:
                    return None

                # Extract tar data
                import io
                import tarfile

                tar_stream = io.BytesIO()
                for chunk in bits:
                    tar_stream.write(chunk)
                tar_stream.seek(0)

                with tarfile.open(fileobj=tar_stream, mode="r") as tar:
                    tar.extractall(temp_path)

                # Read and parse manifest
                if local_manifest_path.exists():
                    with open(local_manifest_path, "r") as f:
                        manifest_data = yaml.safe_load(f)

                    try:
                        return Manifest(**manifest_data)
                    except Exception:
                        return None

        except (docker_errors.ImageNotFound, docker_errors.APIError, Exception):
            return None
        finally:
            # Clean up container
            if container:
                try:
                    container.remove()
                except Exception:
                    pass

    return None


def run_container_with_args(
    image: str,
    args: List[str],
    timeout_seconds: int = 300,
    memory_limit: str = "512m",
    cpu_quota: int = 50000,
    cpu_period: int = 100000,
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

    with docker_client() as client:
        container = None
        try:
            # Run container
            container = client.containers.run(
                image,
                command=args,
                detach=True,
                remove=False,
                network_mode="bridge",
                mem_limit=memory_limit,
                cpu_period=cpu_period,
                cpu_quota=cpu_quota,
            )

            result["container_id"] = container.id or ""

            # Wait for completion
            try:
                exit_status = container.wait(timeout=timeout_seconds)
                result["exit_code"] = exit_status["StatusCode"]
            except docker_errors.APIError as e:
                try:
                    container.kill()
                except Exception:
                    pass
                result["error"] = f"Container execution failed: {e}"
                return result

            # Get output
            try:
                result["output"] = container.logs().decode("utf-8", errors="replace")
            except Exception:
                result["output"] = ""

            result["success"] = result["exit_code"] == 0

        except docker_errors.ImageNotFound:
            result["error"] = f"Docker image not found: {image}"
        except docker_errors.ContainerError as e:
            result["error"] = f"Container execution failed: {e}"
        except docker_errors.APIError as e:
            result["error"] = f"Docker API error: {e}"
        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
        finally:
            # Clean up container
            if container:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    return result
