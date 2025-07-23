# ruff: noqa: T201
"""Example comparing different Kubernetes Pod Pool strategies.

This example demonstrates:
1. Fixed concurrency issues from the original implementation
2. Optimized pool with buffer for better performance
3. Performance comparisons between strategies
"""

import concurrent.futures
import time

from llm_sandbox import KubernetesPodPool
from llm_sandbox.kubernetes_pool_optimized import OptimizedKubernetesPodPool


def performance_comparison_example() -> None:
    """Compare performance between standard and optimized pools."""
    print("=== Performance Comparison ===")

    # Test parameters
    concurrent_tasks = 8
    pool_size = 3

    def execute_simple_code(pool: KubernetesPodPool | OptimizedKubernetesPodPool, pool_type: str, task_id: int) -> dict:
        """Execute simple code and measure timing."""
        start_time = time.time()
        try:
            with pool.get_session() as session:
                result = session.run(f"""
import time
print(f"Task {task_id} from {pool_type} pool")
print(f"Execution time: {{time.strftime('%H:%M:%S')}}")
# Simulate some work
time.sleep(0.1)
print("Task completed successfully")
""")
                success = True
                output = result.stdout
        except Exception as e:
            success = False
            output = str(e)

        total_time = time.time() - start_time
        return {
            "task_id": task_id,
            "pool_type": pool_type,
            "success": success,
            "total_time": total_time,
            "output": output[:100] + "..." if len(output) > 100 else output,
        }

    # Test Standard Pool
    print("\n--- Testing Standard KubernetesPodPool ---")
    with KubernetesPodPool(
        pool_size=pool_size,
        deployment_name="perf-test-standard",
        acquisition_timeout=60,  # Longer timeout for congestion
        verbose=False,
    ) as standard_pool:
        print(f"Standard pool status: {standard_pool.get_pool_status()}")

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_tasks) as executor:
            futures = [
                executor.submit(execute_simple_code, standard_pool, "standard", i) for i in range(concurrent_tasks)
            ]

            standard_results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                standard_results.append(result)
                if result["success"]:
                    print(f"✓ Task {result['task_id']}: {result['total_time']:.2f}s")
                else:
                    print(f"✗ Task {result['task_id']}: {result['output']}")

        standard_total_time = time.time() - start_time
        standard_success_count = sum(1 for r in standard_results if r["success"])

    # Test Optimized Pool
    print("\n--- Testing Optimized KubernetesPodPool ---")
    with OptimizedKubernetesPodPool(
        pool_size=pool_size,
        buffer_size=2,  # 2 extra pods for buffer
        deployment_name="perf-test-optimized",
        acquisition_timeout=60,
        enable_background_replacement=True,
        verbose=False,
    ) as optimized_pool:
        print(f"Optimized pool status: {optimized_pool.get_pool_status()}")

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_tasks) as executor:
            futures = [
                executor.submit(execute_simple_code, optimized_pool, "optimized", i) for i in range(concurrent_tasks)
            ]

            optimized_results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                optimized_results.append(result)
                if result["success"]:
                    print(f"✓ Task {result['task_id']}: {result['total_time']:.2f}s")
                else:
                    print(f"✗ Task {result['task_id']}: {result['output']}")

        optimized_total_time = time.time() - start_time
        optimized_success_count = sum(1 for r in optimized_results if r["success"])

    # Print comparison
    print("\n=== Performance Comparison Results ===")
    print("Standard Pool:")
    print(f"  Success rate: {standard_success_count}/{concurrent_tasks}")
    print(f"  Total time: {standard_total_time:.2f}s")
    print(f"  Avg time per task: {standard_total_time / concurrent_tasks:.2f}s")

    print("Optimized Pool:")
    print(f"  Success rate: {optimized_success_count}/{concurrent_tasks}")
    print(f"  Total time: {optimized_total_time:.2f}s")
    print(f"  Avg time per task: {optimized_total_time / concurrent_tasks:.2f}s")

    if optimized_success_count > 0 and standard_success_count > 0:
        improvement = ((standard_total_time - optimized_total_time) / standard_total_time) * 100
        print(f"  Performance improvement: {improvement:.1f}%")


def buffer_sizing_example() -> None:
    """Demonstrate optimal buffer sizing for different workloads."""
    print("\n=== Buffer Sizing Strategy ===")

    configurations = [
        {"pool_size": 3, "buffer_size": 0, "name": "No Buffer"},
        {"pool_size": 3, "buffer_size": 1, "name": "Small Buffer"},
        {"pool_size": 3, "buffer_size": 3, "name": "Equal Buffer"},
        {"pool_size": 3, "buffer_size": 5, "name": "Large Buffer"},
    ]

    for config in configurations:
        print(f"\n--- {config['name']}: {config['pool_size']} + {config['buffer_size']} ---")

        with OptimizedKubernetesPodPool(
            pool_size=config["pool_size"],
            buffer_size=config["buffer_size"],
            deployment_name=f"buffer-test-{config['buffer_size']}",
            verbose=False,
        ) as pool:
            status = pool.get_pool_status()
            print(
                f"Total pods: {status['total_desired']} (Active: {config['pool_size']}, Buffer: {config['buffer_size']})"
            )
            print(f"Ready pods: {status['ready_pods']}")
            print(f"Resource overhead: {status['total_desired'] - config['pool_size']} extra pods")

            # Quick test of 5 concurrent executions
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(lambda i: pool.get_session().__enter__().run(f"print('Quick test {i}')"))
                    for i in range(5)
                ]
                results = [f.result() for f in futures]

            test_time = time.time() - start_time
            print(f"5 concurrent executions: {test_time:.2f}s")


def recommended_patterns() -> None:
    """Show recommended patterns for different use cases."""
    print("\n=== Recommended Patterns for Different Use Cases ===")

    patterns = [
        {
            "name": "Low Latency API (< 1s response time)",
            "config": {
                "pool_size": 10,
                "buffer_size": 15,  # 150% buffer
                "acquisition_timeout": 5,
                "enable_background_replacement": True,
            },
            "description": "Large buffer for instant pod availability",
        },
        {
            "name": "Batch Processing (cost-optimized)",
            "config": {
                "pool_size": 5,
                "buffer_size": 1,  # Minimal buffer
                "acquisition_timeout": 120,
                "enable_background_replacement": False,
            },
            "description": "Minimal overhead, longer timeouts acceptable",
        },
        {
            "name": "Mixed Workload (balanced)",
            "config": {
                "pool_size": 8,
                "buffer_size": 4,  # 50% buffer
                "acquisition_timeout": 30,
                "enable_background_replacement": True,
            },
            "description": "Balanced performance and resource usage",
        },
    ]

    for pattern in patterns:
        print(f"\n{pattern['name']}:")
        print(f"  Description: {pattern['description']}")
        print("  Configuration:")
        for key, value in pattern["config"].items():
            print(f"    {key}: {value}")

        # Show resource calculation
        total_pods = pattern["config"]["pool_size"] + pattern["config"]["buffer_size"]
        overhead = (pattern["config"]["buffer_size"] / pattern["config"]["pool_size"]) * 100
        print(f"  Resource usage: {total_pods} total pods ({overhead:.0f}% overhead)")


def troubleshooting_guide() -> None:
    """Provide troubleshooting guidance for common issues."""
    print("\n=== Troubleshooting Guide ===")

    issues = [
        {
            "problem": "Concurrent execution failures",
            "symptoms": "API errors, pod acquisition timeouts",
            "solutions": [
                "Increase acquisition_timeout (e.g., 60s)",
                "Add buffer_size for over-provisioning",
                "Check Kubernetes API rate limits",
                "Verify RBAC permissions",
            ],
        },
        {
            "problem": "Selector/label mismatch errors",
            "symptoms": "Deployment validation failures",
            "solutions": [
                "Use fixed KubernetesPodPool (automatically handles labels)",
                "Ensure custom templates include required labels",
                "Verify deployment selector matches pod labels",
            ],
        },
        {
            "problem": "Slow execution times (>3s per task)",
            "symptoms": "High latency, poor user experience",
            "solutions": [
                "Use OptimizedKubernetesPodPool with buffer",
                "Enable background_replacement",
                "Consider persistent pod patterns for dev/test",
                "Scale up cluster resources",
            ],
        },
    ]

    for issue in issues:
        print(f"\nProblem: {issue['problem']}")
        print(f"Symptoms: {issue['symptoms']}")
        print("Solutions:")
        for solution in issue["solutions"]:
            print(f"  • {solution}")


def main() -> None:
    """Run optimized examples and comparisons."""
    print("Optimized Kubernetes Pod Pool Examples")
    print("=" * 50)

    try:
        performance_comparison_example()
        buffer_sizing_example()
        recommended_patterns()
        troubleshooting_guide()

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure to install llm-sandbox with kubernetes support:")
        print("pip install llm-sandbox[k8s]")
    except Exception as e:
        print(f"Error running examples: {e}")
        print("\nCommon solutions:")
        print("1. Ensure kubectl is configured and working")
        print("2. Check RBAC permissions for deployments and pods")
        print("3. Verify cluster has sufficient resources")
        print("4. Try with smaller pool_size values first")


if __name__ == "__main__":
    main()
