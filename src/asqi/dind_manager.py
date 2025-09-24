import logging
import time
import uuid
from typing import Dict, Optional

import docker.errors
from docker.models.containers import Container
from docker.models.networks import Network

import docker

logger = logging.getLogger(__name__)


class DinDServiceManager:
    """Manages Docker-in-Docker service containers for secure test execution."""

    def __init__(self, docker_client: Optional[docker.DockerClient] = None):
        self.client = docker_client or docker.from_env()
        self._active_services: Dict[str, Container] = {}
        self._network: Optional[Network] = None

    def create_isolated_network(self) -> Network:
        """Create isolated network for DinD communication."""
        if not self._network:
            network_name = f"asqi-dind-{uuid.uuid4().hex[:8]}"
            self._network = self.client.networks.create(
                network_name,
                driver="bridge",
                options={
                    "com.docker.network.bridge.enable_icc": "true",
                    "com.docker.network.bridge.enable_ip_masquerade": "true",
                },
            )
            logger.info(f"Created isolated network: {network_name}")
        return self._network

    def start_dind_service(
        self,
        service_name: Optional[str] = None,
        test_name: Optional[str] = None,
        image: Optional[str] = None,
        host_volumes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Start a Docker-in-Docker service container.

        Args:
            service_name: Optional custom name for the service
            test_name: Optional test name for meaningful service naming
            image: Optional image name for context in service naming
            host_volumes: Optional dict of host paths to mount in DinD service
                         Format: {"host_path": "container_path"}

        Returns:
            Dict containing service connection details
        """
        if not service_name:
            # Generate meaningful service name from test context
            if test_name and image:
                # Extract clean names for container naming
                clean_test = test_name.lower().replace(" ", "-").replace("_", "-")[:20]
                clean_image = image.split("/")[-1].split(":")[0].replace("_", "-")[:15]
                service_name = (
                    f"asqi-dind-{clean_test}-{clean_image}-{uuid.uuid4().hex[:6]}"
                )
            elif test_name:
                clean_test = test_name.lower().replace(" ", "-").replace("_", "-")[:25]
                service_name = f"asqi-dind-{clean_test}-{uuid.uuid4().hex[:6]}"
            else:
                service_name = f"asqi-dind-{uuid.uuid4().hex[:8]}"

        # Check if service already exists
        if service_name in self._active_services:
            container = self._active_services[service_name]
            if container.status == "running":
                return self._get_service_info(container)

        network = self.create_isolated_network()

        try:
            # Prepare volume mounts for DinD service if provided
            volumes = {}
            if host_volumes:
                for host_path, container_path in host_volumes.items():
                    volumes[host_path] = {"bind": container_path, "mode": "rw"}

            # Start Docker-in-Docker service container
            dind_container = self.client.containers.run(
                "docker:dind",
                detach=True,
                name=service_name,
                privileged=True,  # DinD service needs privileges, not test containers
                environment={
                    "DOCKER_TLS_CERTDIR": "",  # Disable TLS for simplicity
                    "DOCKER_DRIVER": "overlay2",
                },
                ports={"2376/tcp": None},  # Auto-assign host port
                volumes=volumes
                if volumes
                else None,  # Mount host volumes into DinD service
                remove=False,  # Keep for reuse
                mem_limit="1g",  # Limit DinD service resources
                cpu_quota=100000,  # 100% CPU limit
            )

            # Connect to the isolated network after container creation
            network.connect(dind_container)

            # Wait for Docker daemon to be ready
            self._wait_for_docker_ready(dind_container)

            self._active_services[service_name] = dind_container

            service_info = self._get_service_info(dind_container)
            logger.info(
                f"Started DinD service: {service_name} on port {service_info['host_port']}"
            )

            return service_info

        except docker.errors.APIError as e:
            logger.error(f"Failed to start DinD service: {e}")
            raise RuntimeError(f"DinD service creation failed: {e}")

    def _wait_for_docker_ready(self, container: Container, timeout: int = 30):
        """Wait for Docker daemon inside DinD container to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Test Docker daemon connectivity
                result = container.exec_run("docker version")
                if result.exit_code == 0:
                    logger.debug("Docker daemon is ready in DinD container")
                    return
            except docker.errors.APIError:
                pass

            time.sleep(2)

        raise RuntimeError("Docker daemon failed to start in DinD container")

    def _get_service_info(self, container: Container) -> Dict[str, str]:
        """Extract service connection information."""
        container.reload()  # Refresh container state

        # Get dynamically assigned port
        port_info = container.ports.get("2376/tcp", [])
        host_port = port_info[0]["HostPort"] if port_info else "2376"
        network_name = self._network.name if self._network else "bridge"

        return {
            "container_name": container.name,
            "container_id": container.id,
            "network_name": network_name,
            "docker_host": f"tcp://{container.name}:2376",  # For internal network
            "host_port": str(host_port),  # For external access if needed
        }

    def stop_dind_service(self, service_name: str):
        """Stop and remove a DinD service container."""
        if service_name in self._active_services:
            container = self._active_services[service_name]
            try:
                container.stop(timeout=10)
                container.remove()
                del self._active_services[service_name]
                logger.info(f"Stopped DinD service: {service_name}")
            except docker.errors.APIError as e:
                logger.error(f"Failed to stop DinD service {service_name}: {e}")

    def cleanup_all_services(self):
        """Stop all active DinD services and clean up network."""
        for service_name in list(self._active_services.keys()):
            self.stop_dind_service(service_name)

        if self._network:
            try:
                self._network.remove()
                self._network = None
                logger.info("Cleaned up DinD network")
            except docker.errors.APIError as e:
                logger.error(f"Failed to cleanup DinD network: {e}")

    def get_container_config_for_dind(
        self, service_info: Dict[str, str]
    ) -> Dict[str, any]:
        """
        Generate container configuration to connect to DinD service.

        Args:
            service_info: Service information from start_dind_service()

        Returns:
            Docker container configuration dict
        """
        # Use host port for connection instead of internal network for simplicity
        docker_host = f"tcp://host.docker.internal:{service_info['host_port']}"

        return {
            "environment": {
                "DOCKER_HOST": docker_host,
                "DOCKER_TLS_CERTDIR": "",  # No TLS for internal communication
                # Add environment variable to help containers with log forwarding
                "ASQI_LOG_FORWARDING": "true",
                "ASQI_DIND_SERVICE": service_info["container_name"],
            },
            "privileged": False,  # Test container doesn't need privileges
            "extra_hosts": {"host.docker.internal": "host-gateway"},
            # Apply minimal capability restrictions for security
            "cap_drop": ["ALL"],
            "cap_add": ["SYS_ADMIN"],  # Only capability needed for Docker-in-Docker
        }

    def get_nested_container_logs(self, service_name: str) -> Dict[str, str]:
        """
        Retrieve logs from all containers running inside a DinD service.

        Args:
            service_name: Name of the DinD service

        Returns:
            Dictionary mapping container names to their logs
        """
        if service_name not in self._active_services:
            return {}

        dind_container = self._active_services[service_name]
        nested_logs = {}

        try:
            # Get list of containers running inside DinD
            result = dind_container.exec_run("docker ps -a --format '{{.Names}}'")
            if result.exit_code == 0:
                container_names = result.output.decode("utf-8").strip().split("\n")

                # Get logs from each nested container
                for container_name in container_names:
                    if container_name.strip():
                        log_result = dind_container.exec_run(
                            f"docker logs {container_name.strip()}"
                        )
                        if log_result.exit_code == 0:
                            nested_logs[container_name.strip()] = (
                                log_result.output.decode("utf-8", errors="replace")
                            )

        except docker.errors.APIError as e:
            logger.warning(
                f"Failed to retrieve nested container logs from {service_name}: {e}"
            )

        return nested_logs


# Global DinD manager instance
dind_manager = DinDServiceManager()
