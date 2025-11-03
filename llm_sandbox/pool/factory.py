"""Factory for creating container pool managers."""

from typing import Any

from llm_sandbox.const import SandboxBackend, SupportedLanguage
from llm_sandbox.exceptions import UnsupportedBackendError
from llm_sandbox.pool.base import ContainerPoolManager
from llm_sandbox.pool.config import PoolConfig


def create_pool_manager(
    backend: SandboxBackend = SandboxBackend.DOCKER,
    config: PoolConfig | None = None,
    lang: SupportedLanguage = SupportedLanguage.PYTHON,
    **kwargs: Any,
) -> ContainerPoolManager:
    """Create a container pool manager for the specified backend.

    Args:
        backend: Container backend to use (docker, kubernetes, podman)
        config: Pool configuration (uses defaults if None)
        lang: Programming language for containers
        **kwargs: Additional backend-specific arguments

    Returns:
        ContainerPoolManager instance for the specified backend

    Raises:
        UnsupportedBackendError: If the backend is not supported
        MissingDependencyError: If required backend dependency is not installed

    Examples:
        Create a Docker pool manager:
        ```python
        from llm_sandbox.pool import create_pool_manager, PoolConfig
        from llm_sandbox.const import SandboxBackend, SupportedLanguage

        pool_config = PoolConfig(
            max_pool_size=10,
            min_pool_size=3,
        )

        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=pool_config,
            lang=SupportedLanguage.PYTHON,
        )

        # Use the pool
        with pool:
            container = pool.acquire()
            try:
                # Use container...
                pass
            finally:
                pool.release(container)
        ```

        Create a Kubernetes pool manager:
        ```python
        from kubernetes import client, config as k8s_config

        k8s_config.load_kube_config()
        k8s_client = client.CoreV1Api()

        pool = create_pool_manager(
            backend=SandboxBackend.KUBERNETES,
            config=pool_config,
            lang=SupportedLanguage.PYTHON,
            client=k8s_client,
            namespace="my-namespace",
        )
        ```

        Create a Podman pool manager:
        ```python
        from podman import PodmanClient

        podman_client = PodmanClient()

        pool = create_pool_manager(
            backend=SandboxBackend.PODMAN,
            config=pool_config,
            lang=SupportedLanguage.PYTHON,
            client=podman_client,
        )
        ```

    """
    # Use default config if not provided
    if config is None:
        config = PoolConfig()

    # Create appropriate pool manager based on backend
    match backend:
        case SandboxBackend.DOCKER:
            from llm_sandbox.pool.docker_pool import DockerPoolManager

            return DockerPoolManager(config=config, lang=lang, **kwargs)

        case SandboxBackend.KUBERNETES:
            from llm_sandbox.pool.kubernetes_pool import KubernetesPoolManager

            return KubernetesPoolManager(config=config, lang=lang, **kwargs)

        case SandboxBackend.PODMAN:
            from llm_sandbox.pool.podman_pool import PodmanPoolManager

            return PodmanPoolManager(config=config, lang=lang, **kwargs)

        case _:
            raise UnsupportedBackendError(backend=backend)
