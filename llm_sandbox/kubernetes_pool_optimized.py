"""Alternative optimized patterns for Kubernetes Pod Pool.

This module provides different strategies for managing pod pools with
better performance characteristics for different use cases.
"""

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


class OptimizedKubernetesPodPool:
    """Optimized Kubernetes Pod Pool with multiple performance strategies.

    This version implements several optimization strategies:
    1. Over-provisioning to reduce wait times
    2. Background pod replacement
    3. Pod warming/pre-heating
    4. Smarter resource allocation
    """

    def __init__(
        self,
        namespace: str = "default",
        pool_size: int = 5,
        buffer_size: int | None = None,  # Extra pods beyond pool_size
        deployment_name: str = "llm-sandbox-pool-optimized",
        image: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        client: CoreV1Api | None = None,
        pod_template: dict | None = None,
        warmup_timeout: int = 300,
        acquisition_timeout: int = 60,
        enable_background_replacement: bool = True,
        verbose: bool = False,
        **pod_kwargs: Any,
    ) -> None:
        """Initialize optimized pod pool.

        Args:
            buffer_size: Extra pods to maintain beyond pool_size for faster response
            enable_background_replacement: Start replacement pods immediately when acquired

        """
        self.namespace = namespace
        self.pool_size = pool_size
        self.buffer_size = buffer_size or max(2, pool_size // 2)  # Default: 50% buffer
        self.total_size = pool_size + self.buffer_size
        self.deployment_name = deployment_name
        self.image = image or DefaultImage.__dict__[lang.upper()]
        self.lang = lang
        self.pod_template = pod_template
        self.warmup_timeout = warmup_timeout
        self.acquisition_timeout = acquisition_timeout
        self.enable_background_replacement = enable_background_replacement
        self.verbose = verbose
        self.pod_kwargs = pod_kwargs

        # Initialize Kubernetes clients
        if not client:
            from kubernetes import config as k8s_config

            k8s_config.load_kube_config()
            self.core_v1 = CoreV1Api()
        else:
            self.core_v1 = client

        self.apps_v1 = AppsV1Api()

        # Thread safety and tracking
        self._lock = threading.Lock()
        self._acquired_pods: set[str] = set()
        self._replacement_queue: list[str] = []  # Pods pending replacement
        self._background_thread: threading.Thread | None = None
        self._stop_background = threading.Event()

    def setup(self) -> None:
        """Setup the optimized pod pool."""
        try:
            # Create deployment with buffer size
            manifest = self._create_deployment_manifest()

            try:
                existing = self.apps_v1.read_namespaced_deployment(name=self.deployment_name, namespace=self.namespace)
                if existing.spec.replicas != self.total_size:
                    self.scale(self.total_size)
            except ApiException as e:
                if e.status == 404:
                    self.apps_v1.create_namespaced_deployment(namespace=self.namespace, body=manifest)
                    if self.verbose:
                        logger.info(
                            "Created optimized deployment %s with %s total pods (%s active + %s buffer)",
                            self.deployment_name,
                            self.total_size,
                            self.pool_size,
                            self.buffer_size,
                        )
                else:
                    raise

            # Wait for pods to be ready
            self._wait_for_ready_pods(self.total_size)

            # Start background replacement thread
            if self.enable_background_replacement:
                self._start_background_replacement()

        except Exception as e:
            error_msg = f"Failed to setup optimized pod pool: {e}"
            raise ContainerError(error_msg) from e

    def _create_deployment_manifest(self) -> dict:
        """Create deployment manifest for the optimized pod pool."""
        pod_spec = self.pod_template or self._get_default_pod_template()

        # Ensure pod template has required labels
        if "metadata" not in pod_spec:
            pod_spec["metadata"] = {}
        if "labels" not in pod_spec["metadata"]:
            pod_spec["metadata"]["labels"] = {}

        required_labels = {"app": "llm-sandbox-pool-optimized", "pool": self.deployment_name}
        pod_spec["metadata"]["labels"].update(required_labels)

        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.deployment_name,
                "namespace": self.namespace,
                "labels": {"app": "llm-sandbox-pool-optimized", "managed-by": "llm-sandbox"},
            },
            "spec": {
                "replicas": self.total_size,  # Include buffer
                "selector": {"matchLabels": required_labels},
                "template": pod_spec,
            },
        }

    def _get_default_pod_template(self) -> dict:
        """Generate default pod template."""
        return {
            "metadata": {
                "labels": {
                    "app": "llm-sandbox-pool-optimized",
                    "lang": self.lang.lower(),
                }
            },
            "spec": {
                "containers": [
                    {
                        "name": "sandbox",
                        "image": self.image,
                        "command": ["tail", "-f", "/dev/null"],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "256Mi"},
                            "limits": {"cpu": "1", "memory": "1Gi"},
                        },
                        "securityContext": {
                            "runAsUser": 0,
                            "runAsGroup": 0,
                        },
                        # Pre-warm containers with common tools
                        "lifecycle": {
                            "postStart": {
                                "exec": {
                                    "command": [
                                        "/bin/sh",
                                        "-c",
                                        "echo 'Pod ready for sandbox execution' && python --version",
                                    ]
                                }
                            }
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

    def _wait_for_ready_pods(self, expected_count: int) -> None:
        """Wait for pods to be ready."""
        if self.verbose:
            logger.info("Waiting for %s pods to be ready...", expected_count)

        start_time = time.time()
        while time.time() - start_time < self.warmup_timeout:
            ready_pods = self._get_ready_pods()
            if len(ready_pods) >= expected_count:
                if self.verbose:
                    logger.info("All %s pods are ready", expected_count)
                return

            if self.verbose:
                logger.info("Waiting for pods... (%s/%s ready)", len(ready_pods), expected_count)
            time.sleep(5)

    def _get_ready_pods(self) -> list[str]:
        """Get list of ready pod names."""
        try:
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"app=llm-sandbox-pool-optimized,pool={self.deployment_name}"
            )

            ready_pods = []
            for pod in pods.items:
                if pod.status.phase == "Running" and all(
                    condition.status == "True" for condition in pod.status.conditions or [] if condition.type == "Ready"
                ):
                    ready_pods.append(pod.metadata.name)

            return ready_pods

        except ApiException as e:
            logger.exception("Failed to list pods: %s", e)
            return []

    def acquire_pod(self) -> str:
        """Acquire a pod with optimized selection."""
        with self._lock:
            start_time = time.time()

            while time.time() - start_time < self.acquisition_timeout:
                ready_pods = self._get_ready_pods()
                available_pods = [pod for pod in ready_pods if pod not in self._acquired_pods]

                if available_pods:
                    pod_name = available_pods[0]
                    self._acquired_pods.add(pod_name)

                    # Queue for background replacement if enabled
                    if self.enable_background_replacement:
                        self._replacement_queue.append(pod_name)

                    if self.verbose:
                        logger.info("Acquired optimized pod: %s", pod_name)
                    return pod_name

                if self.verbose:
                    logger.info("No ready pods available, waiting...")
                time.sleep(0.5)  # Shorter polling interval

            error_msg = f"No pod available after {self.acquisition_timeout} seconds"
            raise ContainerError(error_msg)

    def release_pod(self, pod_name: str) -> None:
        """Release pod with optimized cleanup."""
        try:
            with self._lock:
                self._acquired_pods.discard(pod_name)

            self.core_v1.delete_namespaced_pod(
                name=pod_name, namespace=self.namespace, body=k8s_client.V1DeleteOptions()
            )
            if self.verbose:
                logger.info("Released optimized pod: %s", pod_name)

        except ApiException as e:
            if e.status != 404:
                logger.warning("Failed to delete pod %s: %s", pod_name, e)

    def _start_background_replacement(self) -> None:
        """Start background thread for pod replacement."""
        if self._background_thread and self._background_thread.is_alive():
            return

        self._stop_background.clear()
        self._background_thread = threading.Thread(target=self._background_replacement_loop, daemon=True)
        self._background_thread.start()

    def _background_replacement_loop(self) -> None:
        """Background loop to maintain optimal pod count."""
        while not self._stop_background.wait(10):  # Check every 10 seconds
            try:
                ready_count = len(self._get_ready_pods())
                acquired_count = len(self._acquired_pods)
                available_count = ready_count - acquired_count

                # If we're running low on available pods, trigger scaling
                if available_count < self.buffer_size:
                    target_size = self.total_size + (self.buffer_size - available_count)
                    if self.verbose:
                        logger.info(
                            "Background scaling: %s available < %s buffer, scaling to %s",
                            available_count,
                            self.buffer_size,
                            target_size,
                        )
                    self._scale_deployment(target_size)

            except Exception:
                logger.exception("Error in background replacement loop")

    def _scale_deployment(self, new_size: int) -> None:
        """Scale deployment to new size."""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name=self.deployment_name, namespace=self.namespace)
            deployment.spec.replicas = new_size

            self.apps_v1.patch_namespaced_deployment(
                name=self.deployment_name, namespace=self.namespace, body=deployment
            )

        except ApiException as e:
            logger.warning("Failed to scale deployment: %s", e)

    def scale(self, new_pool_size: int) -> None:
        """Scale the pool to new size (including buffer)."""
        self.pool_size = new_pool_size
        new_total = new_pool_size + self.buffer_size
        self._scale_deployment(new_total)
        self.total_size = new_total

    @contextmanager
    def get_session(self, **session_kwargs: Any) -> Iterator[Any]:
        """Get optimized session with faster pod acquisition."""
        pod_name = self.acquire_pod()

        try:
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
            self.release_pod(pod_name)

    def get_pool_status(self) -> dict:
        """Get detailed pool status."""
        try:
            deployment = self.apps_v1.read_namespaced_deployment(name=self.deployment_name, namespace=self.namespace)
            ready_pods = self._get_ready_pods()
            acquired_count = len(self._acquired_pods)
            available_count = len(ready_pods) - acquired_count

            return {
                "deployment_name": self.deployment_name,
                "namespace": self.namespace,
                "pool_size": self.pool_size,
                "buffer_size": self.buffer_size,
                "total_desired": self.total_size,
                "deployment_replicas": deployment.spec.replicas,
                "ready_replicas": deployment.status.ready_replicas or 0,
                "ready_pods": len(ready_pods),
                "acquired_pods": acquired_count,
                "available_pods": available_count,
                "utilization_percent": round((acquired_count / self.pool_size) * 100, 1),
                "ready_pod_names": ready_pods[:5],  # Show first 5 for brevity
                "total_pod_count": len(ready_pods),
            }

        except ApiException as e:
            logger.exception("Failed to get pool status")
            return {"error": str(e)}

    def teardown(self) -> None:
        """Teardown optimized pool."""
        try:
            if self._background_thread:
                self._stop_background.set()
                self._background_thread.join(timeout=5)

            self.apps_v1.delete_namespaced_deployment(
                name=self.deployment_name, namespace=self.namespace, body=k8s_client.V1DeleteOptions()
            )

            if self.verbose:
                logger.info("Deleted optimized deployment %s", self.deployment_name)

        except ApiException as e:
            if e.status != 404:
                logger.warning("Failed to delete deployment: %s", e)

    def __enter__(self) -> "OptimizedKubernetesPodPool":
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.teardown()


class PersistentPodPool:
    """Alternative: Keep pods alive and reset them instead of deleting.

    This approach trades some security isolation for much better performance.
    Pods are reused but reset between executions.
    """

    def __init__(
        self,
        namespace: str = "default",
        pool_size: int = 5,
        deployment_name: str = "llm-sandbox-persistent",
        reset_command: str = "rm -rf /tmp/* /sandbox/* || true",
        **kwargs: Any,
    ):
        """Initialize persistent pod pool.

        Args:
            reset_command: Command to reset pod state between uses

        """
        self.reset_command = reset_command
        # Use the optimized pool as base but don't delete pods
        # Instead, reset them for reuse

    # Implementation would reset pods instead of deleting them
    # This provides much faster turnaround but less security isolation
