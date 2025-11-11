# ruff: noqa: T201, F841
"""Example demonstrating plot clearing functionality with R.

This example shows:
1. Plot accumulation across multiple runs
2. Manual plot clearing
3. Plot counter persistence

Run this with:
    python examples/r_plot_clearing_example.py
"""

import textwrap

import docker

from llm_sandbox import ArtifactSandboxSession, SandboxBackend

client = docker.DockerClient.from_env()


def main() -> None:
    """Demonstrate R plot clearing feature."""
    # Initialize Docker client

    # Create sandbox session with plotting enabled
    with ArtifactSandboxSession(
        client=client,
        backend=SandboxBackend.DOCKER,
        lang="r",
        enable_plotting=True,
        verbose=True,
        keep_template=True,
    ) as session:
        print("\n" + "=" * 60)
        print("R PLOT CLEARING DEMO")
        print("=" * 60)

        # First run: Generate some plots
        print("\n[Step 1] Running first R code - generating 3 plots...")
        first_code = textwrap.dedent("""
            # Base R plots
            plot(1:10, main='Plot 1: Line')
            hist(rnorm(100), main='Plot 2: Histogram')
            boxplot(rnorm(50), main='Plot 3: Boxplot')
        """).strip()

        result1 = session.run(first_code)
        print(f"✓ First run completed: {len(result1.plots)} plots generated")
        print(f"  Plot formats: {[p.format for p in result1.plots]}")

        # Second run: Generate more plots (should accumulate)
        print("\n[Step 2] Running second R code - generating 2 more plots...")
        second_code = textwrap.dedent("""
            # More base R plots
            barplot(c(3,5,7,9), main='Plot 4: Barplot')
            plot(sin, -pi, pi, main='Plot 5: Sine Wave')
        """).strip()

        result2 = session.run(second_code)
        print(f"✓ Second run completed: {len(result2.plots)} total plots (accumulated)")
        print(f"  Plot formats: {[p.format for p in result2.plots]}")

        # Third run: Run non-plotting code
        print("\n[Step 3] Running non-plotting R code...")
        non_plot_code = textwrap.dedent("""
            # Simple computation
            x <- 1:10
            result <- sum(x)
            cat('Sum:', result, '\\n')
        """).strip()

        result3 = session.run(non_plot_code)
        print(f"✓ Third run completed: {len(result3.plots)} total plots (still accumulated)")

        # Manually clear plots
        print("\n[Step 4] Manually clearing all plots...")
        session.clear_plots()
        print("✓ Plots cleared and counter reset")

        # Fourth run: Generate plots after clearing (should start from 1)
        print("\n[Step 5] Running R code after clearing - generating 2 new plots...")
        fourth_code = textwrap.dedent("""
            # New plots after clearing
            plot(cos, -pi, pi, main='New Plot 1: Cosine')
            hist(runif(100), main='New Plot 2: Uniform Distribution')
        """).strip()

        result4 = session.run(fourth_code)
        print(f"✓ Fourth run completed: {len(result4.plots)} plots (reset counter)")
        print(f"  Plot formats: {[p.format for p in result4.plots]}")

        # Fifth run with clear_plots=True
        print("\n[Step 6] Running with clear_plots=True parameter...")
        fifth_code = textwrap.dedent("""
            # Plot with auto-clear
            plot(exp, 0, 2, main='Auto-cleared Plot')
        """).strip()

        result5 = session.run(fifth_code, clear_plots=True)
        print(f"✓ Fifth run completed: {len(result5.plots)} plot (auto-cleared before run)")
        print(f"  Plot formats: {[p.format for p in result5.plots]}")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Step 1: Generated 3 plots    -> Total: {len(result1.plots)}")
        print(f"Step 2: Generated 2 plots    -> Total: {len(result2.plots)} (accumulated)")
        print(f"Step 3: No plots             -> Total: {len(result3.plots)} (accumulated)")
        print("Step 4: Manual clear         -> Counter reset")
        print(f"Step 5: Generated 2 plots    -> Total: {len(result4.plots)} (fresh start)")
        print(f"Step 6: Auto-clear + 1 plot  -> Total: {len(result5.plots)} (isolated)")
        print("=" * 60)

        print("\n✓ R Plot clearing demo completed successfully!")


if __name__ == "__main__":
    main()
