"""Kubernetes Pod Pool for pre-warmed sandbox environments."""

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from kubernetes import client as k8s_client
from kubernetes.client import AppsV1Api, CoreV1Api
from kubernetes.client.exceptions import ApiException

from llm_sandbox.const import DefaultImage, SandboxBackend, SupportedLanguage
from llm_sandbox.exceptions import ContainerError
from llm_sandbox.session import create_session

logger = logging.getLogger(__name__)

POD_ACQUISITION_TIMEOUT = 30  # seconds
POD_STARTUP_TIMEOUT = 300  # 5 minutes
POOL_HEALTH_CHECK_INTERVAL = 60  # seconds


class KubernetesPodPool:
    """Manages a pool of pre-warmed Kubernetes pods for fast sandbox execution.

    This class maintains a deployment of always-running pods that can be quickly
    acquired for code execution. After use, pods are deleted and automatically
    replaced by the deployment's replica set, ensuring fresh isolation for each run.
    """

    def __init__(
        self,
        namespace: str = "default",
        pool_size: int = 5,
        deployment_name: str = "llm-sandbox-pool",
        image: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        client: CoreV1Api | None = None,
        pod_template: dict | None = None,
        warmup_timeout: int = POD_STARTUP_TIMEOUT,
        acquisition_timeout: int = POD_ACQUISITION_TIMEOUT,
        verbose: bool = False,
        **pod_kwargs: Any,
    ) -> None:
        """Initialize Kubernetes Pod Pool.

        Args:
            namespace: Kubernetes namespace for the pool
            pool_size: Number of pods to maintain in the pool
            deployment_name: Name of the deployment managing the pool
            image: Container image to use for pods
            lang: Programming language for the sandbox
            client: Kubernetes CoreV1Api client (optional)
            pod_template: Custom pod template specification
            warmup_timeout: Maximum time to wait for pods to start (seconds)
            acquisition_timeout: Maximum time to wait when acquiring a pod (seconds)
            verbose: Enable verbose logging
            **pod_kwargs: Additional arguments passed to pod creation

        """
        self.namespace = namespace
        self.pool_size = pool_size
        self.deployment_name = deployment_name
        self.image = image or DefaultImage.__dict__[lang.upper()]
        self.lang = lang
        self.pod_template = pod_template
        self.warmup_timeout = warmup_timeout
        self.acquisition_timeout = acquisition_timeout
        self.verbose = verbose
        self.pod_kwargs = pod_kwargs

        if not client:
            from kubernetes import config as k8s_config

            k8s_config.load_kube_config()
            self.core_v1 = CoreV1Api()
        else:
            self.core_v1 = client

        self.apps_v1 = AppsV1Api()

        self._lock = threading.Lock()
        self._health_check_thread: threading.Thread | None = None
        self._stop_health_check = threading.Event()
        self._acquired_pods: set[str] = set()  # Track pods currently being used

    def _get_default_pod_template(self) -> dict:
        """Generate default pod template for the deployment."""
        return {
            "metadata": {
                "labels": {
                    "app": "llm-sandbox-pool",
                    "pool": self.deployment_name,
                    "lang": self.lang.lower(),
                }
            },
            "spec": {
                "containers": [
                    {
                        "name": "sandbox",
                        "image": self.image,
                        "command": ["tail", "-f", "/dev/null"],  # Keep pod running
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "256Mi"},
                            "limits": {"cpu": "1", "memory": "1Gi"},
                        },
                        "securityContext": {
                            "runAsUser": 0,
                            "runAsGroup": 0,
                        },
                    }
                ],
                "securityContext": {
                    "runAsUser": 0,
                    "runAsGroup": 0,
                },
                "restartPolicy": "Always",
            },
        }

    def _create_deployment_manifest(self) -> dict:
        """Create deployment manifest for the pod pool."""
        pod_spec = self.pod_template or self._get_default_pod_template()

        # Ensure pod template has required labels for selector matching
        if "metadata" not in pod_spec:
            pod_spec["metadata"] = {}
        if "labels" not in pod_spec["metadata"]:
            pod_spec["metadata"]["labels"] = {}

        # Add required labels for selector matching
        required_labels = {"app": "llm-sandbox-pool", "pool": self.deployment_name}
        pod_spec["metadata"]["labels"].update(required_labels)

        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.deployment_name,
                "namespace": self.namespace,
                "labels": {"app": "llm-sandbox-pool", "managed-by": "llm-sandbox"},
            },
            "spec": {
                "replicas": self.pool_size,
                "selector": {"matchLabels": required_labels},
                "template": pod_spec,
            },
        }

    def setup(self) -> None:
        """Create the pod pool deployment."""
        try:
            # Check if deployment already exists
            try:
                existing = self.apps_v1.read_namespaced_deployment(name=self.deployment_name, namespace=self.namespace)
                if self.verbose:
                    logger.info("Deployment %s already exists", self.deployment_name)

                # Update replica count if different
                if existing.spec.replicas != self.pool_size:
                    self.scale(self.pool_size)

            except ApiException as e:
                if e.status == 404:
                    # Create new deployment
                    manifest = self._create_deployment_manifest()
                    self.apps_v1.create_namespaced_deployment(namespace=self.namespace, body=manifest)
                    if self.verbose:
                        logger.info("Created deployment %s with %s replicas", self.deployment_name, self.pool_size)
                else:
                    raise

            # Wait for pods to be ready
            self._wait_for_ready_pods()

            # Start health check thread
            self._start_health_check()

        except Exception as e:
            raise ContainerError(f"Failed to setup pod pool: {e}") from e

    def _wait_for_ready_pods(self) -> None:
        """Wait for pods in the pool to be ready."""
        if self.verbose:
            logger.info("Waiting for %s pods to be ready...", self.pool_size)

        start_time = time.time()
        while time.time() - start_time < self.warmup_timeout:
            ready_pods = self._get_ready_pods()
            if len(ready_pods) >= self.pool_size:
                if self.verbose:
                    logger.info("All %s pods are ready", self.pool_size)
                return

            if self.verbose:
                logger.info("Waiting for pods... (%s/%s ready)", len(ready_pods), self.pool_size)
            time.sleep(5)

        ready_pods = self._get_ready_pods()
        if len(ready_pods) < self.pool_size:
            logger.warning("Only %s/%s pods ready after %ss", len(ready_pods), self.pool_size, self.warmup_timeout)

    def _get_ready_pods(self) -> list[str]:
        """Get list of ready pod names from the pool."""
        try:
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"app=llm-sandbox-pool,pool={self.deployment_name}"
            )

            ready_pods = []
            for pod in pods.items:
                if pod.status.phase == "Running" and all(
                    condition.status == "True" for condition in pod.status.conditions or [] if condition.type == "Ready"
                ):
                    ready_pods.append(pod.metadata.name)

            return ready_pods

        except ApiException as e:
            logger.error(f"Failed to list pods: {e}")
            return []

    def acquire_pod(self) -> str:
        """Acquire a pod from the pool for use.

        Returns:
            str: Pod name that can be used for code execution

        Raises:
            ContainerError: If no pod is available within the timeout

        """
        with self._lock:
            start_time = time.time()

            while time.time() - start_time < self.acquisition_timeout:
                ready_pods = self._get_ready_pods()

                # Filter out pods already being used by other threads
                available_pods = [pod for pod in ready_pods if pod not in self._acquired_pods]

                if available_pods:
                    pod_name = available_pods[0]
                    self._acquired_pods.add(pod_name)  # Reserve this pod
                    if self.verbose:
                        logger.info("Acquired pod: %s", pod_name)
                    return pod_name

                if self.verbose:
                    logger.info("No ready pods available, waiting...")
                time.sleep(1)

            error_msg = f"No pod available after {self.acquisition_timeout} seconds"
            raise ContainerError(error_msg)

    def release_pod(self, pod_name: str) -> None:
        """Release (delete) a pod after use.

        The deployment will automatically create a new pod to replace it.

        Args:
            pod_name: Name of the pod to delete

        """
        try:
            # Remove from acquired set first
            with self._lock:
                self._acquired_pods.discard(pod_name)

            self.core_v1.delete_namespaced_pod(
                name=pod_name, namespace=self.namespace, body=k8s_client.V1DeleteOptions()
            )
            if self.verbose:
                logger.info("Released pod: %s", pod_name)

        except ApiException as e:
            if e.status != 404:  # Ignore if pod already deleted
                logger.warning("Failed to delete pod %s: %s", pod_name, e)

    @contextmanager
    def get_session(self, **session_kwargs: Any) -> Iterator[Any]:
        """Get a sandbox session using a pod from the pool.

        Args:
            **session_kwargs: Additional arguments passed to create_session()

        Yields:
            Session: A sandbox session connected to a pool pod

        Example:
            ```python
            pool = KubernetesPodPool(pool_size=10)
            pool.setup()

            with pool.get_session(lang="python") as session:
                result = session.run("print('Hello from pool!')")
                print(result.stdout)
            # Pod is automatically released after use
            ```

        """
        pod_name = self.acquire_pod()

        try:
            # Merge session kwargs with pool defaults
            session_config = {
                "backend": SandboxBackend.KUBERNETES,
                "client": self.core_v1,
                "kube_namespace": self.namespace,
                "container_id": pod_name,
                "lang": self.lang,
                "verbose": self.verbose,
                **self.pod_kwargs,
                **session_kwargs,
            }

            with create_session(**session_config) as session:
                yield session

        finally:
            # Always release the pod after use
            self.release_pod(pod_name)

    def scale(self, new_size: int) -> None:
        """Scale the pod pool to a new size.

        Args:
            new_size: New number of pods to maintain in the pool

        """
        try:
            # Update deployment replica count
            deployment = self.apps_v1.read_namespaced_deployment(name=self.deployment_name, namespace=self.namespace)

            deployment.spec.replicas = new_size

            self.apps_v1.patch_namespaced_deployment(
                name=self.deployment_name, namespace=self.namespace, body=deployment
            )

            self.pool_size = new_size
            if self.verbose:
                logger.info("Scaled pool to %s replicas", new_size)

        except ApiException as e:
            raise ContainerError(f"Failed to scale pool: {e}") from e

    def get_pool_status(self) -> dict:
        """Get current status of the pod pool.

        Returns:
            dict: Pool status information including ready/total pods

        """
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name=self.deployment_name, namespace=self.namespace)

            ready_pods = self._get_ready_pods()

            return {
                "deployment_name": self.deployment_name,
                "namespace": self.namespace,
                "desired_replicas": deployment.spec.replicas,
                "ready_replicas": deployment.status.ready_replicas or 0,
                "available_replicas": deployment.status.available_replicas or 0,
                "ready_pods": len(ready_pods),
                "ready_pod_names": ready_pods,
            }

        except ApiException as e:
            logger.exception("Failed to get pool status: %s")
            return {"error": str(e)}

    def _start_health_check(self) -> None:
        """Start background health check thread."""
        if self._health_check_thread and self._health_check_thread.is_alive():
            return

        self._stop_health_check.clear()
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()

    def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._stop_health_check.wait(POOL_HEALTH_CHECK_INTERVAL):
            try:
                status = self.get_pool_status()
                ready_pods = status.get("ready_pods", 0)

                if ready_pods < self.pool_size * 0.8:  # Alert if less than 80% ready
                    logger.warning("Pool health warning: only %s/%s pods ready", ready_pods, self.pool_size)

            except Exception:
                logger.exception("Health check failed: %s")

    def teardown(self) -> None:
        """Tear down the pod pool deployment."""
        try:
            # Stop health check
            if self._health_check_thread:
                self._stop_health_check.set()
                self._health_check_thread.join(timeout=5)

            # Delete deployment
            self.apps_v1.delete_namespaced_deployment(
                name=self.deployment_name, namespace=self.namespace, body=k8s_client.V1DeleteOptions()
            )

            if self.verbose:
                logger.info("Deleted deployment %s", self.deployment_name)

        except ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                logger.warning("Failed to delete deployment: %s", e)

    def __enter__(self) -> "KubernetesPodPool":
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.teardown()
