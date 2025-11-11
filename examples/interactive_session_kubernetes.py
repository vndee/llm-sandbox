# ruff: noqa: T201

"""Interactive session demo using Kubernetes backend.

This example demonstrates using InteractiveSandboxSession with Kubernetes
instead of Docker. The API is identical, you just specify backend='kubernetes'.
"""

from llm_sandbox.const import SandboxBackend
from llm_sandbox.interactive import InteractiveSandboxSession


def main() -> None:
    """Demonstrate interactive session with Kubernetes backend."""
    print("=" * 60)
    print("Interactive Session Demo - Kubernetes Backend")
    print("=" * 60)

    # Create an interactive session with Kubernetes backend
    # Note: This requires a Kubernetes cluster to be available
    with InteractiveSandboxSession(
        backend=SandboxBackend.KUBERNETES,
        verbose=True,
        timeout=60.0,
        kube_namespace="default",  # Kubernetes-specific parameter
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
        result = session.run("%pip install pandas")
        print("\n[Example 3] Installing and using pandas")
        code = """
import pandas as pd
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35]
})
print(df)
"""
        result = session.run(code)
        print(f"Output: {result.stdout}")

        # Example 4: Using previously imported libraries
        print("\n[Example 4] Using previously imported pandas")
        result = session.run("print(f'Average age: {df[\"Age\"].mean()}')")
        print(f"Output: {result.stdout}")

        # Example 5: Complex computation with state
        print("\n[Example 5] Complex computation with state")
        code = """
# Define a function
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Calculate and store result
fib_10 = fibonacci(10)
print(f'Fibonacci(10) = {fib_10}')
"""
        result = session.run(code)
        print(f"Output: {result.stdout}")

        # Example 6: Use the stored result
        print("\n[Example 6] Use stored result")
        result = session.run("print(f'Stored value * 2 = {fib_10 * 2}')")
        print(f"Output: {result.stdout}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
