"""Factory for creating sandbox sessions."""

from typing import Optional, Union
from abc import ABC, abstractmethod

import docker
from kubernetes import client as k8s_client

from .docker import SandboxDockerSession
from .kubernetes import SandboxKubernetesSession
from .podman import SandboxPodmanSession
from .monitoring import ResourceLimits
from .exceptions import ValidationError


class SessionFactory(ABC):
    """Abstract base factory for creating sandbox sessions."""

    @abstractmethod
    def create_session(self, **kwargs):
        """Create a new sandbox session."""
        pass


class DockerSessionFactory(SessionFactory):
    """Factory for creating Docker-based sandbox sessions."""

    def __init__(
        self,
        client: Optional[docker.DockerClient] = None,
        default_resource_limits: Optional[ResourceLimits] = None,
    ):
        self.client = client or docker.from_env()
        self.default_resource_limits = default_resource_limits or ResourceLimits()

    def create_session(self, **kwargs) -> SandboxDockerSession:
        """Create a new Docker sandbox session."""
        # Merge default resource limits with provided ones
        resource_limits = kwargs.pop("resource_limits", self.default_resource_limits)

        # Convert resource limits to Docker container configs
        container_configs = {
            "cpu_count": resource_limits.max_cpu_percent / 100.0,
            "mem_limit": str(resource_limits.max_memory_bytes),
            "network_disabled": kwargs.pop("network_disabled", False),
        }

        if kwargs.get("container_configs", {}):
            kwargs["container_configs"] = (
                kwargs.get("container_configs", {}) | container_configs
            )
        else:
            kwargs["container_configs"] = container_configs

        return SandboxDockerSession(client=self.client, **kwargs)


class KubernetesSessionFactory(SessionFactory):
    """Factory for creating Kubernetes-based sandbox sessions."""

    def __init__(
        self,
        client: Optional[k8s_client.CoreV1Api] = None,
        default_namespace: str = "default",
    ):
        if not client:
            k8s_client.Configuration.set_default(k8s_client.Configuration())
            self.client = k8s_client.CoreV1Api()
        else:
            self.client = client
        self.default_namespace = default_namespace

    def create_session(self, **kwargs) -> SandboxKubernetesSession:
        """Create a new Kubernetes sandbox session."""
        namespace = kwargs.pop("kube_namespace", self.default_namespace)

        return SandboxKubernetesSession(
            client=self.client, kube_namespace=namespace, **kwargs
        )


class PodmanSessionFactory(SessionFactory):
    """Factory for creating Podman-based sandbox sessions."""

    def create_session(self, **kwargs) -> SandboxPodmanSession:
        """Create a new Podman sandbox session."""
        return SandboxPodmanSession(**kwargs)


class UnifiedSessionFactory:
    """Unified factory for creating sandbox sessions of any type."""

    def __init__(
        self,
        docker_client: Optional[docker.DockerClient] = None,
        k8s_client: Optional[k8s_client.CoreV1Api] = None,
        default_resource_limits: Optional[ResourceLimits] = None,
        default_k8s_namespace: str = "default",
    ):
        self.factories = {
            "docker": DockerSessionFactory(
                client=docker_client, default_resource_limits=default_resource_limits
            ),
            "kubernetes": KubernetesSessionFactory(
                client=k8s_client, default_namespace=default_k8s_namespace
            ),
            "podman": PodmanSessionFactory(),
        }

    def create_session(
        self, backend: str = "docker", **kwargs
    ) -> Union[SandboxDockerSession, SandboxKubernetesSession, SandboxPodmanSession]:
        """
        Create a new sandbox session.

        Args:
            backend: The backend to use ('docker', 'kubernetes', or 'podman')
            **kwargs: Additional arguments for session creation

        Returns:
            A new sandbox session

        Raises:
            ValidationError: If backend is not supported
        """
        if backend not in self.factories:
            raise ValidationError(
                f"Unsupported backend: {backend}. "
                f"Must be one of: {list(self.factories.keys())}"
            )

        return self.factories[backend].create_session(**kwargs)
