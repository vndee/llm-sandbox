# ruff: noqa: T201

"""Interactive session demo using Podman backend.

This example demonstrates using InteractiveSandboxSession with Podman
instead of Docker. The API is identical, you just specify backend='podman'.
"""

import logging

from podman import PodmanClient

from llm_sandbox.const import SandboxBackend
from llm_sandbox.interactive import InteractiveSandboxSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

client = PodmanClient(
    base_url="unix:///var/folders/lh/rjbzw60n1fv7xr9kffn7gr840000gn/T/podman/podman-machine-default-api.sock"
)


def main() -> None:
    """Demonstrate interactive session with Podman backend."""
    print("=" * 60)
    print("Interactive Session Demo - Podman Backend")
    print("=" * 60)

    # Create an interactive session with Podman backend
    with InteractiveSandboxSession(
        backend=SandboxBackend.PODMAN,
        verbose=True,
        timeout=60.0,
        client=client,
    ) as session:
        # Example 1: Basic arithmetic
        print("\n[Example 1] Basic arithmetic")
        result = session.run("x = 10\nprint(f'x = {x}')")
        print(f"Output: {result.stdout}")

        # Example 2: State persists between runs
        print("\n[Example 2] State persists between runs")
        result = session.run("y = x + 5\nprint(f'y = {y}')")
        print(f"Output: {result.stdout}")

        # Example 3: Installing and using libraries
        print("\n[Example 3] Installing and using numpy")
        result = session.run("%pip install numpy")
        print(f"Output: {result.stdout}")
        code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f'Array: {arr}')
print(f'Mean: {arr.mean()}')
"""
        result = session.run(code)
        print(f"Output: {result.stdout}")

        # Example 4: Using previously imported libraries
        print("\n[Example 4] Using previously imported numpy")
        result = session.run("print(f'Sum: {arr.sum()}')")
        print(f"Output: {result.stdout}")

        # Example 5: Error handling
        print("\n[Example 5] Error handling")
        result = session.run("print(1 / 0)")
        print(f"Exit code: {result.exit_code}")
        print(f"Error: {result.stderr}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
