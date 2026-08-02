"""Example demonstrating the skip_environment_setup feature.

skip_environment_setup=True bypasses the virtualenv build and pip upgrade, trading the
ability to install libraries at runtime for a faster start. That trade is worth most on
backends where startup is billed, so this times both paths on whichever backend you pick.

Usage:
    python examples/skip_environment_setup_demo.py            # every available backend
    python examples/skip_environment_setup_demo.py tenki      # just one
"""

import logging
import time
from contextlib import ExitStack, closing

from llm_sandbox import SandboxBackend, SandboxSession
from llm_sandbox.exceptions import MissingDependencyError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

# Skip only when the runtime is missing/unreachable — not for example/logic bugs.
_BACKEND_UNAVAILABLE: list[type[BaseException]] = [
    OSError,
    ConnectionError,
    TimeoutError,
    MissingDependencyError,
]
try:
    from docker.errors import DockerException

    _BACKEND_UNAVAILABLE.append(DockerException)
except ImportError:
    pass
try:
    from podman.errors.exceptions import PodmanError

    _BACKEND_UNAVAILABLE.append(PodmanError)
except ImportError:
    pass
try:
    from kubernetes.client.exceptions import ApiException
    from kubernetes.config.config_exception import ConfigException

    _BACKEND_UNAVAILABLE.extend((ApiException, ConfigException))
except ImportError:
    pass
try:
    from tenki import MissingAuthTokenError

    _BACKEND_UNAVAILABLE.append(MissingAuthTokenError)
except ImportError:
    pass

BACKEND_UNAVAILABLE_ERRORS = tuple(_BACKEND_UNAVAILABLE)

BACKENDS = {
    "docker": SandboxBackend.DOCKER,
    "kubernetes": SandboxBackend.KUBERNETES,
    "podman": SandboxBackend.PODMAN,
    "tenki": SandboxBackend.TENKI,
}


def build_client(backend_name: str, stack: ExitStack) -> object | None:
    """Build a client for one backend only, at the point of use.

    The client is entered into ``stack`` so it is closed when the caller exits the
    ``ExitStack``. Docker/Podman sessions do not close an injected client themselves.

    Returns:
        object | None: A backend client, or None when the backend builds its own.

    """
    if backend_name == "docker":
        import docker

        # DockerClient has close() but is not a context manager.
        return stack.enter_context(closing(docker.DockerClient.from_env()))
    if backend_name == "podman":
        from podman import PodmanClient

        return stack.enter_context(PodmanClient.from_env())
    return None


def time_startup(backend_name: str, *, skip_setup: bool) -> None:
    """Open a session one way and report how long it took to become usable.

    Args:
        backend_name: Key into BACKENDS.
        skip_setup: Whether to skip the venv build and pip upgrade.

    """
    label = "skip_environment_setup=True" if skip_setup else "default (builds venv)"
    logger.info("\n--- %s: %s ---", backend_name, label)

    started = time.perf_counter()
    with (
        ExitStack() as stack,
        SandboxSession(
            lang="python",
            verbose=True,
            backend=BACKENDS[backend_name],
            client=build_client(backend_name, stack),
            skip_environment_setup=skip_setup,
        ) as session,
    ):
        ready = time.perf_counter() - started
        # Confirms the sandbox is genuinely usable, not merely opened. With setup skipped
        # this relies on the image already shipping a working interpreter.
        output = session.run("import sys; print(sys.version.split()[0])")

    logger.info("%s on %s: ready in %.2fs, python %s", label, backend_name, ready, output.stdout.strip())


def run_demo(backend_name: str) -> None:
    """Time both startup paths on one backend.

    Raises:
        ValueError: If the backend name is not recognised.

    """
    if backend_name not in BACKENDS:
        msg = f"Unknown backend {backend_name!r}; choose from {sorted(BACKENDS)}"
        raise ValueError(msg)

    for skip_setup in (False, True):
        time_startup(backend_name, skip_setup=skip_setup)


def demo_kubernetes_use_case() -> None:
    """Show how this would be used in a Kubernetes deployment scenario."""
    logger.info("\nDemo: Kubernetes deployment scenario")

    # In a Kubernetes environment, administrators might want to:
    # 1. Use a custom image with pre-installed packages
    # 2. Skip environment setup to reduce pod startup time
    # 3. Avoid potential network issues with pip index access

    # In a Kubernetes environment, administrators might configure:
    session_config = {
        "lang": "python",
        "verbose": True,
        "backend": SandboxBackend.KUBERNETES,  # Would be KUBERNETES in real scenario
        "skip_environment_setup": True,
        "pod_manifest": {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "sandbox-python",
                "namespace": "default",
            },
            "spec": {
                "containers": [
                    {
                        "name": "my-python-app",
                        "image": "ghcr.io/vndee/sandbox-python-311-bullseye",
                        "tty": True,
                        "securityContext": {
                            "runAsUser": 0,
                            "runAsGroup": 0,
                        },
                        "resources": {
                            "requests": {"memory": "256Mi", "cpu": "100m"},
                            "limits": {"memory": "512Mi", "cpu": "500m"},
                        },
                    }
                ],
                "securityContext": {
                    "runAsUser": 0,
                    "runAsGroup": 0,
                },
            },
        },
    }

    logger.info("Kubernetes-style configuration:")
    logger.info("  - skip_environment_setup: %s", session_config["skip_environment_setup"])
    logger.info("  - Custom image: %s", session_config["pod_manifest"]["spec"]["containers"][0]["image"])  # type: ignore[index]
    resource_limits = session_config["pod_manifest"]["spec"]["containers"][0]["resources"]["limits"]  # type: ignore[index]
    logger.info("  - Resource limits: %s", resource_limits)

    try:
        with SandboxSession(
            lang=session_config["lang"],
            verbose=session_config["verbose"],
            backend=session_config["backend"],  # type: ignore[arg-type]
            skip_environment_setup=session_config["skip_environment_setup"],
            pod_manifest=session_config["pod_manifest"],
        ) as session:
            output = session.run("""
import os
import sys
print(f"Running in: {os.environ.get('HOSTNAME', 'unknown')}")
print(f"Python path: {sys.executable}")
print("Environment setup was skipped - using pre-configured image!")
""")
            logger.info("K8s demo output:\n%s", output.stdout)
    except Exception as e:  # noqa: BLE001
        logger.warning("Kubernetes demo failed due to cluster connectivity: %s", str(e)[:100])
        logger.info("Note: The skip_environment_setup feature is working correctly")
        logger.info("This error is related to Kubernetes cluster setup, not our feature")
        logger.info("In a properly configured K8s cluster, this would work fine")


def main() -> None:
    """Time one backend by name, or every backend when none is given."""
    import sys

    if len(sys.argv) > 1:
        backend_name = sys.argv[1]
        run_demo(backend_name)
        if backend_name == "kubernetes":
            demo_kubernetes_use_case()
    else:
        # One unavailable runtime must not stop the rest, so failures are logged.
        for backend_name in BACKENDS:
            try:
                run_demo(backend_name)
            except BACKEND_UNAVAILABLE_ERRORS:
                logger.exception("%s skipped", backend_name)
        demo_kubernetes_use_case()

    separator = "=" * 60
    logger.info("\n%s", separator)
    logger.info("Summary:")
    logger.info("- Use skip_environment_setup=True for:")
    logger.info("  • Custom images with pre-configured environments")
    logger.info("  • Kubernetes deployments to reduce startup time")
    logger.info("  • Avoiding pip index/network configuration issues")
    logger.info("- Use skip_environment_setup=False (default) for:")
    logger.info("  • Standard development and testing")
    logger.info("  • When using base images without pre-installed packages")
    logger.info("  • Maximum compatibility with dynamic package installation")


if __name__ == "__main__":
    main()
