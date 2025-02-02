"""Main session module for LLM Sandbox."""

from typing import Optional, Union

import docker
from kubernetes import client as k8s_client

from .const import SupportedLanguage
from .monitoring import ResourceLimits
from .factory import UnifiedSessionFactory
from .docker import SandboxDockerSession
from .kubernetes import SandboxKubernetesSession
from .podman import SandboxPodmanSession


class SandboxSession:
    """Factory function for creating sandbox sessions."""

    def __new__(
        cls,
        client: Optional[Union[docker.DockerClient, k8s_client.CoreV1Api]] = None,
        image: Optional[str] = None,
        dockerfile: Optional[str] = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        commit_container: bool = True,
        verbose: bool = False,
        use_kubernetes: bool = False,
        use_podman: bool = False,
        kube_namespace: Optional[str] = "default",
        resource_limits: Optional[ResourceLimits] = None,
        strict_security: bool = True,
        container_configs: Optional[dict] = None,
    ) -> Union[SandboxDockerSession, SandboxKubernetesSession, SandboxPodmanSession]:
        """
        Create a new sandbox session.

        Args:
            client: Docker, Kubernetes or Podman client
            image: Container image to use
            dockerfile: Path to Dockerfile
            lang: Programming language
            keep_template: Whether to keep the container template
            commit_container: Whether to commit container changes
            verbose: Enable verbose logging
            use_kubernetes: Use Kubernetes backend
            use_podman: Use Podman backend
            kube_namespace: Kubernetes namespace
            resource_limits: Resource limits for the container
            strict_security: Enable strict security checks
            container_configs: Additional container configurations

        Returns:
            A new sandbox session
        """
        factory = UnifiedSessionFactory(
            docker_client=client if isinstance(client, docker.DockerClient) else None,
            k8s_client=client if isinstance(client, k8s_client.CoreV1Api) else None,
            default_resource_limits=resource_limits,
            default_k8s_namespace=kube_namespace,
        )

        if use_kubernetes:
            backend = "kubernetes"
        elif use_podman:
            backend = "podman"
        else:
            backend = "docker"

        return factory.create_session(
            backend=backend,
            image=image,
            dockerfile=dockerfile,
            lang=lang,
            keep_template=keep_template,
            commit_container=commit_container,
            verbose=verbose,
            kube_namespace=kube_namespace,
            strict_security=strict_security,
            container_configs=container_configs,
        )
