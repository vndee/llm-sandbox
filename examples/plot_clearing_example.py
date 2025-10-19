# ruff: noqa: T201, F841

"""Example demonstrating plot clearing functionality in ArtifactSandboxSession."""

import textwrap

import docker

from llm_sandbox import ArtifactSandboxSession, SandboxBackend

client = docker.DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")


def main() -> None:
    """Demonstrate plot clearing between runs."""
    # This example shows how to use the clear_plots feature
    # Note: Docker must be running for this to work

    example_code = textwrap.dedent("""
        import matplotlib.pyplot as plt
        import numpy as np

        # Create a simple plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        plt.figure(figsize=(8, 6))
        plt.plot(x, y, label='sin(x)')
        plt.title('Example Plot')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()
    """).strip()

    non_plot_code = textwrap.dedent("""
        print("Hello, world! No plots here.")
    """).strip()

    print("Example: Plot Clearing Feature")
    print("=" * 40)

    with ArtifactSandboxSession(
        client=client,
        lang="python",
        backend=SandboxBackend.DOCKER,
        enable_plotting=True,
        verbose=True,
        keep_template=True,
    ) as session:
        print("1. First run - should generate 1 plot")
        result1 = session.run(example_code)
        print(f"   Generated {len(result1.plots)} plots")

        print("\\n2. Second run - should generate another plot (total 2)")
        result2 = session.run(example_code)
        print(f"   Generated {len(result2.plots)} plots")

        print("\\n3. Third run with clear_plots=True - should generate 1 plot")
        result3 = session.run(example_code, clear_plots=True)
        print(f"   Generated {len(result3.plots)} plots")

        print("\\n4. Fourth run - no plotting code")
        result4 = session.run(non_plot_code, clear_plots=True)
        print(f"   Generated {len(result4.plots)} plots")
        print(f"   Output: {result4.stdout.strip()}")

        print("\\n5. Manual plot clearing")
        session.clear_plots()
        result5 = session.run(example_code)
        print(f"   Generated {len(result5.plots)} plots")

    print("To run this example:")
    print("1. Make sure Docker is running")
    print("2. Uncomment the session code above")
    print("3. Run this script")

    print("\\nFeatures demonstrated:")
    print("- clear_plots parameter in run() method")
    print("- Manual clear_plots() method call")
    print("- Plot persistence vs clearing behavior")


if __name__ == "__main__":
    main()
