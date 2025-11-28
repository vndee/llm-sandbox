"""Fix for issue #121: Using emptyDir volume with read-only root filesystem.

This script demonstrates the proper way to use Kubernetes with strict security settings:
- readOnlyRootFilesystem: True (security best practice)
- runAsUser: 1000 (non-root)
- emptyDir volume mounted at /sandbox (provides writable workspace)

This allows secure, read-only containers while still providing a writable workspace.
"""

import logging

from llm_sandbox import SandboxBackend, SandboxSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def demonstrate_fix() -> bool:
    """Demonstrate the fix using emptyDir volume mount."""
    logger.info("Creating session with read-only root filesystem + emptyDir volume...")

    # Pod manifest with proper volume configuration
    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "test-readonly-fixed",
            "labels": {"app": "llm-sandbox-test"},
        },
        "spec": {
            "containers": [
                {
                    "name": "sandbox",
                    "image": "python:3.11-slim",
                    "tty": True,
                    "resources": {
                        "requests": {"memory": "256Mi", "cpu": "250m"},
                        "limits": {"memory": "512Mi", "cpu": "500m"},
                    },
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "readOnlyRootFilesystem": True,  # Still read-only for security
                        "allowPrivilegeEscalation": False,
                    },
                    "volumeMounts": [
                        {
                            "name": "sandbox-writable",
                            "mountPath": "/sandbox",
                        },
                        {
                            "name": "tmp-writable",
                            "mountPath": "/tmp",
                        },
                    ],
                }
            ],
            "securityContext": {
                "runAsNonRoot": True,
                "fsGroup": 2000,  # Ensures volume is writable by user 1000
            },
            "volumes": [
                {
                    "name": "sandbox-writable",
                    "emptyDir": {},  # Ephemeral writable storage for /sandbox
                },
                {
                    "name": "tmp-writable",
                    "emptyDir": {},  # Ephemeral writable storage for /tmp (needed by pip)
                },
            ],
        },
    }

    try:
        with SandboxSession(
            backend=SandboxBackend.KUBERNETES,
            lang="python",
            kube_namespace="default",
            pod_manifest=pod_manifest,
            verbose=True,
        ) as session:
            # Test basic execution
            result = session.run("print('Hello from Kubernetes!')")
            logger.info("Basic execution: %s", result.stdout.strip())

            # Test file operations in /sandbox
            result = session.run("import os; os.makedirs('/sandbox/test', exist_ok=True); print('Directory created')")
            logger.info("Directory creation: %s", result.stdout.strip())

            # Test writing files
            result = session.run("with open('/sandbox/test.txt', 'w') as f: f.write('test'); print('File written')")
            logger.info("File write: %s", result.stdout.strip())

            # Verify root filesystem is still read-only
            result = session.run("import os; os.makedirs('/root-test', exist_ok=True); print('Should fail')")
            logger.warning("Root filesystem test: %s", result.stderr)

    except Exception:
        logger.exception("Unexpected error")
        return False

    logger.info("Session completed successfully!")
    return True


def demonstrate_alternative_workdir() -> bool:
    """Alternative solution: Use a different workdir that's writable."""
    logger.info("Alternative solution: Configure workdir to writable location")
    logger.info("=" * 70)

    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "test-custom-workdir",
            "labels": {"app": "llm-sandbox-test"},
        },
        "spec": {
            "containers": [
                {
                    "name": "sandbox",
                    "image": "python:3.11-slim",
                    "tty": True,
                    "resources": {
                        "requests": {"memory": "256Mi", "cpu": "250m"},
                        "limits": {"memory": "512Mi", "cpu": "500m"},
                    },
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "readOnlyRootFilesystem": True,
                        "allowPrivilegeEscalation": False,
                    },
                    # Mount emptyDir at custom location
                    "volumeMounts": [
                        {
                            "name": "workspace",
                            "mountPath": "/workspace",  # Custom workdir
                        },
                        {
                            "name": "tmp-writable",
                            "mountPath": "/tmp",  # Also need writable /tmp
                        },
                    ],
                }
            ],
            "securityContext": {
                "runAsNonRoot": True,
                "fsGroup": 2000,
            },
            "volumes": [
                {
                    "name": "workspace",
                    "emptyDir": {},
                },
                {
                    "name": "tmp-writable",
                    "emptyDir": {},
                },
            ],
        },
    }

    try:
        with SandboxSession(
            backend=SandboxBackend.KUBERNETES,
            lang="python",
            kube_namespace="default",
            pod_manifest=pod_manifest,
            workdir="/workspace",  # Configure workdir to match volumeMount
            verbose=True,
        ) as session:
            result = session.run("import os; print(f'Working directory: {os.getcwd()}')")
            logger.info("Custom workdir: %s", result.stdout.strip())

    except Exception:
        logger.exception("Unexpected error")
        return False

    logger.info("Alternative solution works!")
    return True


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Fix for issue #121: emptyDir volume solution")
    logger.info("=" * 70)

    # Primary solution
    success1 = demonstrate_fix()

    # Alternative solution
    success2 = demonstrate_alternative_workdir()

    if success1 and success2:
        logger.info("Both solutions work!")
        logger.info("\nKey takeaways:")
        logger.info("1. Use emptyDir volume when readOnlyRootFilesystem: True")
        logger.info("2. Mount it at /sandbox (default workdir) or configure custom workdir")
        logger.info("3. Set fsGroup to ensure volume is writable by container user")
    else:
        logger.info("One or more solutions failed")
    logger.info("=" * 70)
