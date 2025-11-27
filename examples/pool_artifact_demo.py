# ruff: noqa: T201, S603, S607

"""Unified artifact extraction demo with container pooling for all backends.

This example demonstrates artifact extraction (plots, CSV files, text files)
using container pooling across Docker, Podman, and Kubernetes backends.

Key features demonstrated:
1. Matplotlib plot extraction with pooling
2. CSV file generation and extraction
3. Text file artifacts
4. Performance comparison with/without pooling
5. Multi-backend support (Docker, Podman, Kubernetes)
"""

import argparse
import base64
import sys
import textwrap
import time
from pathlib import Path

import docker
import podman

from llm_sandbox import SandboxBackend, SupportedLanguage
from llm_sandbox.pool import ArtifactPooledSandboxSession, ExhaustionStrategy, PoolConfig, create_pool_manager

# Output directory for artifacts (for manual inspection)
OUTPUT_DIR = Path("pool_artifact_output")
docker_client = docker.DockerClient.from_env()
podman_client = podman.PodmanClient.from_env()


def demo_matplotlib_plots(backend: SandboxBackend, num_plots: int = 5) -> None:
    """Demonstrate matplotlib plot extraction with pooling.

    Args:
        backend: Backend to use (docker, podman, kubernetes)
        num_plots: Number of different plots to generate

    """
    print(f"\n1. Matplotlib Plot Extraction with Pooling ({backend.upper()}):")
    print("-" * 60)

    # Create plot code
    plot_code = textwrap.dedent("""
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate data
    x = np.linspace(0, 10, 100)
    y = np.sin(x * {multiplier}) * np.exp(-x / 10)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='f(x)')
    plt.fill_between(x, 0, y, alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('Plot #{plot_num}: Damped Sine Wave', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print('Plot #{plot_num} generated successfully!')
    """)
    client = None
    if backend == SandboxBackend.DOCKER:
        client = docker_client
    elif backend == SandboxBackend.PODMAN:
        client = podman_client

    # Create pool
    pool_manager = create_pool_manager(
        backend=backend,
        config=PoolConfig(
            max_pool_size=3,
            min_pool_size=2,
            enable_prewarming=True,
            exhaustion_strategy=ExhaustionStrategy.WAIT,
        ),
        lang=SupportedLanguage.PYTHON,
        libraries=["matplotlib", "numpy"],
        client=client,
    )

    try:
        print(f"  Generating {num_plots} plots using pooled sessions...")
        start_time = time.time()

        # Create output directory
        output_dir = OUTPUT_DIR / "plots" / backend.value
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, num_plots + 1):
            iteration_start = time.time()

            code = plot_code.format(multiplier=i, plot_num=i)

            with ArtifactPooledSandboxSession(
                pool_manager=pool_manager,
                enable_plotting=True,
                verbose=False,
            ) as session:
                result = session.run(code)

                # Extract plots
                if result.plots:
                    for j, plot in enumerate(result.plots):
                        plot_path = output_dir / f"plot_{i}_{j}.{plot.format.value}"
                        plot_path.write_bytes(base64.b64decode(plot.content_base64))
                        print(
                            f"    Plot {i}: Generated in {time.time() - iteration_start:.2f}s "
                            f"(saved to {plot_path.relative_to(OUTPUT_DIR)})"
                        )
                else:
                    print(f"    Plot {i}: No plots detected")

                print(f"      Output: {result.stdout.strip()}")

        total_time = time.time() - start_time
        print(f"\n  Total time: {total_time:.2f}s")
        print(f"  Average per plot: {total_time / num_plots:.2f}s")

    finally:
        pool_manager.close()


def demo_csv_generation(backend: SandboxBackend, num_datasets: int = 3) -> None:
    """Demonstrate CSV file generation and extraction with pooling.

    Args:
        backend: Backend to use (docker, podman, kubernetes)
        num_datasets: Number of CSV datasets to generate

    """
    print(f"\n2. CSV File Generation and Extraction ({backend.upper()}):")
    print("-" * 60)

    csv_code = textwrap.dedent("""
    import pandas as pd
    import numpy as np

    # Generate random dataset
    np.random.seed({seed})
    data = {{
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
        'value': np.random.randn(100).cumsum(),
        'category': np.random.choice(['A', 'B', 'C'], 100),
    }}

    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv('/sandbox/dataset_{num}.csv', index=False)

    # Print summary
    print(f'Dataset {num}: {{len(df)}} rows, mean={{df["value"].mean():.2f}}')
    """)

    client = None
    if backend == SandboxBackend.DOCKER:
        client = docker_client
    elif backend == SandboxBackend.PODMAN:
        client = podman_client

    pool_manager = create_pool_manager(
        backend=backend,
        config=PoolConfig(
            max_pool_size=2,
            min_pool_size=1,
            enable_prewarming=True,
        ),
        lang=SupportedLanguage.PYTHON,
        libraries=["pandas", "numpy"],
        client=client,
    )

    try:
        print(f"  Generating {num_datasets} CSV datasets using pooled sessions...")
        start_time = time.time()

        # Create output directory
        output_dir = OUTPUT_DIR / "csv" / backend.value
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, num_datasets + 1):
            code = csv_code.format(seed=i * 42, num=i)

            with ArtifactPooledSandboxSession(
                pool_manager=pool_manager,
                enable_plotting=False,
                verbose=False,
            ) as session:
                result = session.run(code)
                print(f"    Dataset {i}: {result.stdout.strip()}")

                # Extract CSV file
                csv_path = output_dir / f"dataset_{i}.csv"
                session.copy_from_runtime(f"/sandbox/dataset_{i}.csv", csv_path.as_posix())
                file_size = csv_path.stat().st_size
                print(f"      Saved to {csv_path.relative_to(OUTPUT_DIR)} ({file_size} bytes)")

        total_time = time.time() - start_time
        print(f"\n  Total time: {total_time:.2f}s")
        print(f"  Average per dataset: {total_time / num_datasets:.2f}s")
        print(f"  CSV files saved to: {output_dir.resolve().relative_to(Path.cwd())}")

    finally:
        pool_manager.close()


def demo_mixed_artifacts(backend: SandboxBackend) -> None:
    """Demonstrate mixed artifact types in a single session.

    Args:
        backend: Backend to use (docker, podman, kubernetes)

    """
    print(f"\n3. Mixed Artifact Types ({backend.upper()}):")
    print("-" * 60)

    mixed_code = textwrap.dedent("""
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import json

    # 1. Generate a plot
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.title('Sine Wave')
    plt.grid(True)
    plt.savefig('/sandbox/sine_plot.png')
    plt.show()

    # 2. Create a CSV file
    df = pd.DataFrame({
        'x': x,
        'y': y,
    })
    df.to_csv('/sandbox/data.csv', index=False)

    # 3. Create a JSON file
    metadata = {
        'title': 'Sine Wave Analysis',
        'points': len(x),
        'min': float(y.min()),
        'max': float(y.max()),
    }
    with open('/sandbox/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # 4. Create a text report
    with open('/sandbox/report.txt', 'w') as f:
        f.write('Sine Wave Analysis Report\\n')
        f.write('=' * 50 + '\\n')
        f.write(f'Data points: {len(x)}\\n')
        f.write(f'Min value: {y.min():.4f}\\n')
        f.write(f'Max value: {y.max():.4f}\\n')
        f.write(f'Mean value: {y.mean():.4f}\\n')

    print('All artifacts generated successfully!')
    print(f'- Plot: sine_plot.png')
    print(f'- Data: data.csv ({len(df)} rows)')
    print(f'- Metadata: metadata.json')
    print(f'- Report: report.txt')
    """)

    client = None
    if backend == SandboxBackend.DOCKER:
        client = docker_client
    elif backend == SandboxBackend.PODMAN:
        client = podman_client

    pool_manager = create_pool_manager(
        backend=backend,
        config=PoolConfig(
            max_pool_size=2,
            min_pool_size=1,
        ),
        lang=SupportedLanguage.PYTHON,
        libraries=["matplotlib", "pandas", "numpy"],
        client=client,
    )

    try:
        print("  Generating multiple artifact types in a single session...")

        # Create output directory
        output_dir = OUTPUT_DIR / "mixed" / backend.value
        output_dir.mkdir(parents=True, exist_ok=True)

        with ArtifactPooledSandboxSession(
            pool_manager=pool_manager,
            enable_plotting=True,
            verbose=False,
        ) as session:
            result = session.run(mixed_code)
            print("\n  Execution output:")
            for line in result.stdout.strip().split("\n"):
                print(f"    {line}")

            # Extract all artifacts
            artifacts = {
                "sine_plot.png": "PNG image",
                "data.csv": "CSV data",
                "metadata.json": "JSON metadata",
                "report.txt": "Text report",
            }

            print("\n  Extracting artifacts:")
            for filename, description in artifacts.items():
                artifact_path = output_dir / filename
                session.copy_from_runtime(f"/sandbox/{filename}", artifact_path.as_posix())
                file_size = artifact_path.stat().st_size
                print(f"    ✓ {filename} ({description}): {file_size} bytes")

            # Also extract plots from result
            if result.plots:
                print(f"\n  Matplotlib plots extracted: {len(result.plots)}")
                for i, plot in enumerate(result.plots):
                    print(f"    Plot {i + 1}: {plot.format.value} format")

    finally:
        pool_manager.close()


def demo_performance_comparison(backend: SandboxBackend, num_tasks: int = 10) -> None:  # noqa: PLR0912, PLR0915
    """Compare artifact extraction performance with and without pooling.

    Args:
        backend: Backend to use (docker, podman, kubernetes)
        num_tasks: Number of tasks to run for comparison

    """
    print(f"\n4. Performance Comparison: Pooling vs No Pooling ({backend.upper()}):")
    print("-" * 60)

    simple_plot_code = textwrap.dedent("""
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    plt.plot(x, y)
    plt.title('Simple Plot')
    plt.show()
    print('Plot generated')
    """)

    client = None
    if backend == SandboxBackend.DOCKER:
        client = docker_client
    elif backend == SandboxBackend.PODMAN:
        client = podman_client

    print(f"  Running {num_tasks} tasks WITHOUT pooling...")
    from llm_sandbox import ArtifactSandboxSession

    no_pool_times = []
    start_time = time.time()

    for i in range(1, num_tasks + 1):
        task_start = time.time()
        try:
            with ArtifactSandboxSession(
                client=client,
                lang="python",
                backend=backend,
                verbose=False,
                enable_plotting=True,
                libraries=["matplotlib", "numpy"],
            ) as session:
                result = session.run(simple_plot_code)
                if result.plots:
                    task_time = time.time() - task_start
                    no_pool_times.append(task_time)
                    if i == 1 or i % 5 == 0:
                        print(f"    Task {i:2d}: {task_time:.2f}s")
        except Exception as e:  # noqa: BLE001
            print(f"    Task {i:2d}: FAILED - {e!s}")

    no_pool_total = time.time() - start_time

    print(f"\n  Running {num_tasks} tasks WITH pooling...")
    client = None
    if backend == SandboxBackend.DOCKER:
        client = docker_client
    elif backend == SandboxBackend.PODMAN:
        client = podman_client

    pool_manager = create_pool_manager(
        backend=backend,
        config=PoolConfig(
            max_pool_size=3,
            min_pool_size=2,
            enable_prewarming=True,
        ),
        lang=SupportedLanguage.PYTHON,
        libraries=["matplotlib", "numpy"],
        client=client,
    )

    try:
        pool_times = []
        start_time = time.time()

        for i in range(1, num_tasks + 1):
            task_start = time.time()
            try:
                with ArtifactPooledSandboxSession(
                    pool_manager=pool_manager,
                    enable_plotting=True,
                    verbose=False,
                ) as session:
                    result = session.run(simple_plot_code)
                    if result.plots:
                        task_time = time.time() - task_start
                        pool_times.append(task_time)
                        if i == 1 or i % 5 == 0:
                            print(f"    Task {i:2d}: {task_time:.2f}s")
            except Exception as e:  # noqa: BLE001
                print(f"    Task {i:2d}: FAILED - {e!s}")

        pool_total = time.time() - start_time

        # Show comparison
        if no_pool_times and pool_times:
            import statistics

            speedup = no_pool_total / pool_total
            time_saved = no_pool_total - pool_total

            print("\n  📊 Results:")
            print("     Without pooling:")
            print(f"       Total time: {no_pool_total:.2f}s")
            print(f"       Average per task: {statistics.mean(no_pool_times):.2f}s")
            print("     With pooling:")
            print(f"       Total time: {pool_total:.2f}s")
            print(f"       Average per task: {statistics.mean(pool_times):.2f}s")
            print(f"       Warmup (first 3): {statistics.mean(pool_times[:3]):.2f}s")
            print(f"       Steady-state: {statistics.mean(pool_times[3:]):.2f}s")
            print(f"\n     Speedup: {speedup:.2f}x faster")
            print(f"     Time saved: {time_saved:.2f}s")

    finally:
        pool_manager.close()


def main() -> None:
    """Run all artifact extraction demonstrations."""
    parser = argparse.ArgumentParser(
        description="Artifact extraction demo with container pooling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Docker backend
  python pool_artifact_demo.py --backend docker

  # Run with Kubernetes backend
  python pool_artifact_demo.py --backend kubernetes

  # Run with Podman backend
  python pool_artifact_demo.py --backend podman

  # Run specific demos only
  python pool_artifact_demo.py --backend docker --demo plots,csv
        """,
    )
    parser.add_argument(
        "--backend",
        choices=[b.value for b in SandboxBackend],
        default=SandboxBackend.DOCKER,
        help="Backend to use (default: docker)",
    )
    parser.add_argument(
        "--demo",
        help="Comma-separated list of demos to run (plots, csv, mixed, perf). Default: all",
    )
    parser.add_argument(
        "--num-plots",
        type=int,
        default=5,
        help="Number of plots to generate (default: 5)",
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=3,
        help="Number of CSV datasets to generate (default: 3)",
    )
    parser.add_argument(
        "--num-perf-tasks",
        type=int,
        default=10,
        help="Number of tasks for performance comparison (default: 10)",
    )

    args = parser.parse_args()

    # Determine which demos to run
    demos_to_run = set(args.demo.lower().split(",")) if args.demo else {"plots", "csv", "mixed", "perf"}

    print("=" * 60)
    print("Artifact Extraction Demo with Container Pooling")
    print("=" * 60)
    print(f"Backend: {args.backend.upper()}")
    print(f"Demos: {', '.join(sorted(demos_to_run))}")
    print("=" * 60)

    try:
        if "plots" in demos_to_run:
            demo_matplotlib_plots(SandboxBackend(args.backend), args.num_plots)

        if "csv" in demos_to_run:
            demo_csv_generation(SandboxBackend(args.backend), args.num_datasets)

        if "mixed" in demos_to_run:
            demo_mixed_artifacts(SandboxBackend(args.backend))

        if "perf" in demos_to_run:
            demo_performance_comparison(SandboxBackend(args.backend), args.num_perf_tasks)

        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
