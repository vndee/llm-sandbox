"""Kubernetes-specific container pool manager."""

from typing import Any

from kubernetes import client as k8s_client
from kubernetes.client import CoreV1Api
from kubernetes.client.exceptions import ApiException

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.k8s_utils import retry_k8s_api_call
from llm_sandbox.pool.base import ContainerPoolManager
from llm_sandbox.pool.config import PoolConfig

NOT_FOUND_ERROR_CODE = 404


class KubernetesPoolManager(ContainerPoolManager):
    """Container pool manager for Kubernetes backend.

    This manager creates and manages a pool of Kubernetes pods,
    reusing the standard session logic for pod initialization and
    environment setup (venv, pip, library installation, etc.).
    """

    def __init__(
        self,
        config: PoolConfig,
        lang: SupportedLanguage | str,
        image: str | None = None,
        client: CoreV1Api | None = None,
        namespace: str = "default",
        pod_manifest: dict | None = None,
        **session_kwargs: Any,
    ) -> None:
        """Initialize Kubernetes pool manager.

        Args:
            config: Pool configuration
            lang: Programming language
            image: Container image to use
            client: Kubernetes client instance
            namespace: Kubernetes namespace
            pod_manifest: Custom pod manifest template
            **session_kwargs: Additional session arguments

        """
        from kubernetes import config as k8s_config

        if client is None:
            try:
                k8s_config.load_kube_config()
            except Exception:  # noqa: BLE001
                k8s_config.load_incluster_config()
            client = CoreV1Api()

        self.namespace = namespace
        self.pod_manifest_template = pod_manifest

        # Resolve image using helper
        from llm_sandbox.pool.base import resolve_default_image

        image = resolve_default_image(lang, image)

        super().__init__(client=client, config=config, lang=lang, image=image, **session_kwargs)

    def _create_session_for_container(self) -> Any:
        """Create a Kubernetes session for initializing a pod.

        This creates a session that, when opened, will:
        1. Create a pod
        2. Wait for pod to be running
        3. Set up the environment (venv, pip, libraries, etc.)

        Returns:
            KubernetesSession instance (not yet opened)

        """
        from llm_sandbox.kubernetes import SandboxKubernetesSession

        # Create session with same configuration as the pool
        # The session handles all initialization automatically
        return SandboxKubernetesSession(
            client=self.client,
            image=self.image,
            lang=str(self.lang),
            namespace=self.namespace,
            pod_manifest=self.pod_manifest_template,
            **self.session_kwargs,
        )

    def _destroy_container_impl(self, container: Any) -> None:
        """Destroy a Kubernetes pod.

        Args:
            container: Pod object or pod name string to destroy

        """
        try:
            # Handle both pod name strings and pod objects
            pod_name = container if isinstance(container, str) else container.metadata.name

            # Use retry logic with thread-safe API calls
            retry_k8s_api_call(
                self.client.delete_namespaced_pod,
                name=pod_name,
                namespace=self.namespace,
                body=k8s_client.V1DeleteOptions(),
                logger=self.logger,
            )
        except ApiException as e:
            if e.status != NOT_FOUND_ERROR_CODE:  # Ignore not found errors
                self.logger.exception("Failed to destroy pod")
        except Exception:
            self.logger.exception("Failed to destroy pod")

    def _get_container_id(self, container: Any) -> str:
        """Get Kubernetes pod name.

        Args:
            container: Pod object or pod name string

        Returns:
            Pod name

        """
        # Handle both pod name strings (from session.container) and pod objects
        if isinstance(container, str):
            return container
        return str(container.metadata.name)

    def _health_check_impl(self, container: Any) -> bool:
        """Perform health check on Kubernetes pod.

        Args:
            container: Pod object or pod name string to check

        Returns:
            True if healthy, False otherwise

        """
        try:
            # Handle both pod name strings and pod objects
            pod_name = container if isinstance(container, str) else container.metadata.name

            # Get current pod status with retry logic
            pod = retry_k8s_api_call(
                self.client.read_namespaced_pod,
                name=pod_name,
                namespace=self.namespace,
                logger=self.logger,
            )

            # Check if pod is running
            if pod.status.phase != "Running":
                return False

            # Check if all containers are ready
            if pod.status.container_statuses:
                for status in pod.status.container_statuses:
                    if not status.ready:
                        return False

        except ApiException as e:
            if e.status == NOT_FOUND_ERROR_CODE:
                return False
            self.logger.exception("Health check error")
            return False
        except Exception:
            self.logger.exception("Health check error")
            return False
        else:
            return True
