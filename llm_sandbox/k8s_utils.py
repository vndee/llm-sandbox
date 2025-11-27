"""Kubernetes-specific utility functions for thread-safe API calls and retry logic."""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from kubernetes.client.exceptions import ApiException

# Constants for retry logic
K8S_API_MAX_RETRIES = 3
K8S_API_RETRY_DELAY = 0.5  # seconds

# Global lock for thread-safe Kubernetes API calls
# The Kubernetes Python client is not fully thread-safe, especially for concurrent operations
_k8s_api_lock = threading.Lock()

# Type variable for return type
T = TypeVar("T")


def retry_k8s_api_call(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = K8S_API_MAX_RETRIES,
    retry_delay: float = K8S_API_RETRY_DELAY,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> T:
    """Retry Kubernetes API calls with exponential backoff and thread-safety.

    This function wraps Kubernetes API calls with:
    1. Thread-safety via a global lock (Kubernetes client is not thread-safe)
    2. Automatic retry with exponential backoff for transient WebSocket errors
    3. Detailed logging of retry attempts

    Args:
        func: The Kubernetes API function to call
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 0.5)
        logger: Logger instance for retry warnings (optional)
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the API call

    Raises:
        ApiException: If the API call fails after all retries
        Exception: For any other non-retryable errors

    Example:
        ```python
        from llm_sandbox.k8s_utils import retry_k8s_api_call

        pod = retry_k8s_api_call(
            client.read_namespaced_pod,
            name="my-pod",
            namespace="default"
        )
        ```

    """
    if logger is None:
        logger = logging.getLogger(__name__)

    last_exception = None

    for attempt in range(max_retries):
        try:
            # Use global lock for thread-safety
            with _k8s_api_lock:
                return func(*args, **kwargs)

        except ApiException as e:
            last_exception = e

            # Check if this is a WebSocket handshake error (status=0)
            # This error occurs when the Kubernetes client tries to use WebSocket for regular HTTP calls
            is_websocket_error = e.status == 0 and "Handshake status" in str(e.reason)

            if is_websocket_error and attempt < max_retries - 1:
                # Exponential backoff: 0.5s, 1.0s, 2.0s, ...
                delay = retry_delay * (2**attempt)
                logger.warning(
                    "Kubernetes API WebSocket error (attempt %d/%d), retrying in %.2fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    str(e.reason)[:100],  # Truncate long error messages
                )
                time.sleep(delay)
                continue

            # For other API exceptions, don't retry
            raise

        except Exception:
            # For non-API exceptions, don't retry
            raise

    # All retries failed
    if last_exception:
        raise last_exception

    # This should never happen, but satisfy type checker
    msg = "Unexpected state: no result and no exception"
    raise RuntimeError(msg)
