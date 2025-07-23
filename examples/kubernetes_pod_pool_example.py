# ruff: noqa: T201
"""Example usage of KubernetesPodPool for fast sandbox execution.

This example demonstrates how to use the KubernetesPodPool to maintain
a pool of pre-warmed Kubernetes pods for faster code execution.

Prerequisites:
- Kubernetes cluster access configured (kubectl works)
- llm-sandbox installed with kubernetes extras: pip install llm-sandbox[k8s]
- Appropriate RBAC permissions for creating/managing deployments and pods

Usage:
    python kubernetes_pod_pool_example.py
"""

import time

from llm_sandbox import KubernetesPodPool


def basic_pool_example() -> None:
    """Example of basic usage of KubernetesPodPool."""
    print("=== Basic KubernetesPodPool Example ===")

    # Create and setup a pool of 3 Python pods
    with KubernetesPodPool(
        namespace="default", pool_size=3, deployment_name="my-sandbox-pool", lang="python", verbose=True
    ) as pool:
        print(f"\nPool status: {pool.get_pool_status()}")

        # Execute code using pods from the pool
        for i in range(5):
            print(f"\n--- Execution {i + 1} ---")
            start_time = time.time()

            with pool.get_session() as session:
                result = session.run(f"""
import time
print(f"Hello from pod! Execution {i + 1}")
print(f"Current time: {{time.strftime('%H:%M:%S')}}")
print("This pod will be deleted after use for security.")
""")
                print(f"Result: {result.stdout}")

            execution_time = time.time() - start_time
            print(f"Execution time: {execution_time:.2f}s (includes pod cleanup)")

        print(f"\nFinal pool status: {pool.get_pool_status()}")


def concurrent_execution_example() -> None:
    """Example of concurrent execution using the pool."""
    print("\n=== Concurrent Execution Example ===")
    import concurrent.futures

    def execute_code(pool: KubernetesPodPool, worker_id: int) -> str:
        """Execute code in a worker thread."""
        try:
            with pool.get_session() as session:
                result = session.run(f"""
import os
import threading
print(f"Worker {worker_id} executing in pod")
print(f"Thread ID: {{threading.get_ident()}}")
print(f"Process ID: {{os.getpid()}}")
""")
                return f"Worker {worker_id}: {result.stdout.strip()}"
        except Exception as e:
            return f"Worker {worker_id}: Error - {e}"

    # Create pool and execute multiple tasks concurrently
    with KubernetesPodPool(pool_size=5, deployment_name="concurrent-pool", verbose=True) as pool:
        print(f"Pool status: {pool.get_pool_status()}")

        # Submit concurrent tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(execute_code, pool, i) for i in range(8)]

            for future in concurrent.futures.as_completed(futures):
                print(future.result())


def custom_configuration_example() -> None:
    """Example with custom pod template and configuration."""
    print("\n=== Custom Configuration Example ===")

    # Custom pod template with resource limits
    custom_template = {
        "metadata": {"labels": {"app": "llm-sandbox-pool", "environment": "development", "custom": "template"}},
        "spec": {
            "containers": [
                {
                    "name": "sandbox",
                    "image": "python:3.11-slim",
                    "command": ["tail", "-f", "/dev/null"],
                    "resources": {
                        "requests": {"cpu": "200m", "memory": "512Mi"},
                        "limits": {"cpu": "500m", "memory": "1Gi"},
                    },
                    "env": [
                        {"name": "CUSTOM_VAR", "value": "custom_value"},
                        {"name": "ENVIRONMENT", "value": "development"},
                    ],
                    "securityContext": {
                        "runAsUser": 1000,
                        "runAsGroup": 1000,
                    },
                }
            ],
            "securityContext": {
                "runAsUser": 1000,
                "runAsGroup": 1000,
            },
            "restartPolicy": "Always",
        },
    }

    with KubernetesPodPool(
        namespace="default",
        pool_size=2,
        deployment_name="custom-sandbox-pool",
        pod_template=custom_template,
        verbose=True,
    ) as pool:
        with pool.get_session() as session:
            result = session.run("""
import os
print(f"Custom environment variable: {os.environ.get('CUSTOM_VAR', 'Not set')}")
print(f"Environment: {os.environ.get('ENVIRONMENT', 'Not set')}")
print(f"User ID: {os.getuid()}")
print(f"Group ID: {os.getgid()}")
""")
            print("Custom pod execution result:")
            print(result.stdout)


def monitoring_example() -> None:
    """Example of monitoring pool health and scaling."""
    print("\n=== Monitoring and Scaling Example ===")

    pool = KubernetesPodPool(pool_size=2, deployment_name="monitoring-pool", verbose=True)

    try:
        pool.setup()

        # Monitor initial status
        print("Initial pool status:")
        status = pool.get_pool_status()
        for key, value in status.items():
            print(f"  {key}: {value}")

        # Scale up the pool
        print("\nScaling pool to 4 replicas...")
        pool.scale(4)
        time.sleep(10)  # Wait for scaling

        print("Status after scaling:")
        status = pool.get_pool_status()
        for key, value in status.items():
            print(f"  {key}: {value}")

        # Scale down the pool
        print("\nScaling pool to 1 replica...")
        pool.scale(1)
        time.sleep(5)

        print("Final status:")
        status = pool.get_pool_status()
        for key, value in status.items():
            print(f"  {key}: {value}")

    finally:
        pool.teardown()


def error_handling_example() -> None:
    """Example of error handling with the pool."""
    print("\n=== Error Handling Example ===")

    try:
        with KubernetesPodPool(
            pool_size=2,
            deployment_name="error-handling-pool",
            acquisition_timeout=5,  # Short timeout for demo
            verbose=True,
        ) as pool:
            # Try to acquire more pods than available
            acquired_pods = []
            try:
                for i in range(5):  # Try to acquire 5 pods from pool of 2
                    pod = pool.acquire_pod()
                    acquired_pods.append(pod)
                    print(f"Acquired pod: {pod}")
            except Exception as e:
                print(f"Expected error when pool exhausted: {e}")
            finally:
                # Release acquired pods
                for pod in acquired_pods:
                    pool.release_pod(pod)
                    print(f"Released pod: {pod}")

    except Exception as e:
        print(f"Pool setup error: {e}")


def main() -> None:
    """Run all examples."""
    print("KubernetesPodPool Examples")
    print("=" * 40)

    try:
        basic_pool_example()
        concurrent_execution_example()
        custom_configuration_example()
        monitoring_example()
        error_handling_example()

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure to install llm-sandbox with kubernetes support:")
        print("pip install llm-sandbox[k8s]")
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have:")
        print("1. Kubernetes cluster access (kubectl configured)")
        print("2. Appropriate RBAC permissions")
        print("3. llm-sandbox[k8s] installed")


if __name__ == "__main__":
    main()
