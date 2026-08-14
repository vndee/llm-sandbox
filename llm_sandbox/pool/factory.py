"""Factory for creating container pool managers."""

from typing import Any

from llm_sandbox.backends.plugin import BackendCapability
from llm_sandbox.const import SandboxBackend, SupportedLanguage
from llm_sandbox.pool.base import ContainerPoolManager
from llm_sandbox.pool.config import PoolConfig
from llm_sandbox.registry import get_backend


def create_pool_manager(
    client: Any | None = None,
    backend: SandboxBackend | str = SandboxBackend.DOCKER,
    config: PoolConfig | None = None,
    lang: SupportedLanguage | str = SupportedLanguage.PYTHON,
    **kwargs: Any,
) -> ContainerPoolManager:
    """Create a container pool manager for the specified backend.

    Args:
        client: Client to use for container creation (optional)
        backend: Container backend to use (docker, kubernetes, podman), or the name of a
            plugin backend that declares the ``pooling`` capability
        config: Pool configuration (uses defaults if None)
        lang: Programming language for containers
        **kwargs: Additional backend-specific arguments

    Returns:
        ContainerPoolManager instance for the specified backend

    Raises:
        UnsupportedBackendError: If the backend is not supported, or does not declare the
            ``pooling`` capability
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

    provider = get_backend(str(backend))
    provider.require(BackendCapability.POOLING)
    manager = provider.create_pool_manager(client=client, config=config, lang=lang, **kwargs)

    # Stamp the resolved name so a PooledSandboxSession can route back to this backend
    # without having to guess it from the manager's class name.
    if not getattr(manager, "backend_name", ""):
        manager.backend_name = provider.name
    return manager
