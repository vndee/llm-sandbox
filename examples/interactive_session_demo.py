# ruff: noqa: T201

"""Interactive Session Demo - Demonstrates persistent state execution.

This example showcases the InteractiveSandboxSession feature, which maintains
Python interpreter state across multiple run() calls, similar to Jupyter notebooks.
"""

import textwrap

import docker

from llm_sandbox import InteractiveSandboxSession, KernelType
from llm_sandbox.interactive import InteractiveSettings

client = docker.DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")


def demo_basic_state_persistence() -> None:
    """Demonstrate basic variable persistence across runs."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic State Persistence")
    print("=" * 60)

    with InteractiveSandboxSession(lang="python", client=client) as session:
        # First execution: define variables
        print("\n1. Defining variables...")
        result = session.run("x = 21\ny = 2")
        print(f"   Exit code: {result.exit_code}")

        # Second execution: use the variables
        print("\n2. Using previously defined variables...")
        result = session.run("result = x * y\nprint(f'Result: {result}')")
        print(f"   Output: {result.stdout.strip()}")

        # Third execution: modify state
        print("\n3. Modifying state...")
        result = session.run("x = x + 10\nprint(f'New x: {x}')")
        print(f"   Output: {result.stdout.strip()}")


def demo_function_definitions() -> None:
    """Demonstrate persistent function definitions."""
    print("\n" + "=" * 60)
    print("Demo 2: Function Definitions Persist")
    print("=" * 60)

    with InteractiveSandboxSession(lang="python", client=client) as session:
        # Define a function
        print("\n1. Defining utility functions...")
        session.run(
            textwrap.dedent("""
            def fibonacci(n):
                if n <= 1:
                    return n
                return fibonacci(n-1) + fibonacci(n-2)

            def factorial(n):
                if n <= 1:
                    return 1
                return n * factorial(n-1)
        """)
        )
        print("   Functions defined")

        # Use the functions
        print("\n2. Using the defined functions...")
        result = session.run(
            textwrap.dedent("""
            fib_10 = fibonacci(10)
            fact_5 = factorial(5)
            print(f"Fibonacci(10) = {fib_10}")
            print(f"Factorial(5) = {fact_5}")
        """)
        )
        print(f"   Output:\n{result.stdout.strip()}")


def demo_class_definitions() -> None:
    """Demonstrate persistent class definitions and instances."""
    print("\n" + "=" * 60)
    print("Demo 3: Class Definitions and Instances")
    print("=" * 60)

    with InteractiveSandboxSession(lang="python", client=client) as session:
        # Define a class
        print("\n1. Defining a DataProcessor class...")
        session.run(
            textwrap.dedent("""
            class DataProcessor:
                def __init__(self):
                    self.data = []

                def add(self, value):
                    self.data.append(value)

                def get_stats(self):
                    if not self.data:
                        return {'count': 0, 'sum': 0, 'mean': 0}
                    return {
                        'count': len(self.data),
                        'sum': sum(self.data),
                        'mean': sum(self.data) / len(self.data)
                    }

            processor = DataProcessor()
            print("DataProcessor initialized")
        """)
        )

        # Use the instance
        print("\n2. Adding data to the processor...")
        session.run(
            textwrap.dedent("""
            for i in range(1, 11):
                processor.add(i * 2)
            print(f"Added {len(processor.data)} values")
        """)
        )

        # Get statistics
        print("\n3. Computing statistics...")
        result = session.run(
            textwrap.dedent("""
            stats = processor.get_stats()
            print(f"Count: {stats['count']}")
            print(f"Sum: {stats['sum']}")
            print(f"Mean: {stats['mean']:.2f}")
        """)
        )
        print(f"   Output:\n{result.stdout.strip()}")


def demo_ipython_magic() -> None:
    """Demonstrate IPython magic commands."""
    print("\n" + "=" * 60)
    print("Demo 4: IPython Magic Commands")
    print("=" * 60)

    with InteractiveSandboxSession(lang="python", client=client) as session:
        # Create some variables
        session.run("x = 10\ny = 20\nz = 'hello'")

        # Use %who magic to list variables
        print("\n1. Listing variables with %who...")
        result = session.run("%who")
        print(f"   Variables: {result.stdout.strip()}")

        # Use %pwd magic to show current directory
        print("\n2. Showing current directory with %pwd...")
        result = session.run("%pwd")
        print(f"   Directory: {result.stdout.strip()}")

        # Use shell command
        print("\n3. Running shell command with !...")
        result = session.run("!echo 'Hello from shell'")
        print(f"   Output: {result.stdout.strip()}")


def demo_multi_step_data_analysis() -> None:
    """Demonstrate a multi-step data analysis workflow."""
    print("\n" + "=" * 60)
    print("Demo 5: Multi-Step Data Analysis")
    print("=" * 60)

    with InteractiveSandboxSession(
        lang="python",
        client=client,
    ) as session:
        # Step 0: Install libraries with magic command
        print("\n0. Installing libraries with magic command...")
        result = session.run("%pip install pandas numpy")
        print(f"   {result.stdout.strip()}")

        # Step 1: Import libraries and create data
        print("\n1. Importing libraries and creating dataset...")
        result = session.run(
            textwrap.dedent("""
            import pandas as pd
            import numpy as np

            # Create sample dataset
            np.random.seed(42)
            data = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=50),
                'sales': np.random.randint(100, 1000, 50),
                'category': np.random.choice(['A', 'B', 'C'], 50)
            })
            print(f"Created dataset with {len(data)} rows")
            """),
        )
        print(f"   {result.stdout.strip()}")

        # Step 2: Compute basic statistics
        print("\n2. Computing basic statistics...")
        result = session.run(
            textwrap.dedent("""
            total_sales = data['sales'].sum()
            avg_sales = data['sales'].mean()
            print(f"Total Sales: ${total_sales:,}")
            print(f"Average Sales: ${avg_sales:.2f}")
        """)
        )
        print(f"   {result.stdout.strip()}")

        # Step 3: Group by category
        print("\n3. Analyzing by category...")
        result = session.run(
            textwrap.dedent("""
            category_stats = data.groupby('category')['sales'].agg(['sum', 'mean', 'count'])
            print("\\nSales by Category:")
            print(category_stats.to_string())
        """)
        )
        print(f"   {result.stdout.strip()}")

        # Step 4: Find top 5 sales days
        print("\n4. Finding top 5 sales days...")
        result = session.run(
            textwrap.dedent("""
            top_days = data.nlargest(5, 'sales')[['date', 'sales', 'category']]
            print("\\nTop 5 Sales Days:")
            for idx, row in top_days.iterrows():
                print(f"  {row['date'].strftime('%Y-%m-%d')}: ${row['sales']} (Category {row['category']})")
        """)
        )
        print(f"   {result.stdout.strip()}")


def demo_custom_settings() -> None:
    """Demonstrate custom InteractiveSettings configuration."""
    print("\n" + "=" * 60)
    print("Demo 6: Custom Settings Configuration")
    print("=" * 60)

    # Configure custom settings
    settings = InteractiveSettings(
        kernel_type=KernelType.IPYTHON,
        max_memory="2GB",
        history_size=200,
        timeout=600,
        poll_interval=0.1,
    )

    print("\nSettings:")
    print(f"  Kernel Type: {settings.kernel_type}")
    print(f"  Max Memory: {settings.max_memory}")
    print(f"  History Size: {settings.history_size}")
    print(f"  Timeout: {settings.timeout}s")
    print(f"  Poll Interval: {settings.poll_interval}s")

    with InteractiveSandboxSession(lang="python", interactive_settings=settings, client=client) as session:
        print("\n1. Running code with custom settings...")
        result = session.run(
            textwrap.dedent("""
            import sys
            print(f"Python version: {sys.version.split()[0]}")
            print("Custom settings applied!")
        """)
        )
        print(f"   {result.stdout.strip()}")


def demo_error_handling() -> None:
    """Demonstrate error handling while maintaining state."""
    print("\n" + "=" * 60)
    print("Demo 7: Error Handling with State Preservation")
    print("=" * 60)

    with InteractiveSandboxSession(lang="python", client=client) as session:
        # Set up some state
        print("\n1. Setting up initial state...")
        result = session.run("x = 42\ny = 100")
        print(f"   Variables defined: x={result.exit_code == 0}")

        # Try to run code that will fail
        print("\n2. Running code that causes an error...")
        result = session.run("z = x / 0")  # Division by zero
        print(f"   Exit code: {result.exit_code}")
        print(f"   Error: {result.stderr.strip()[:100]}...")

        # Verify state is preserved
        print("\n3. Verifying state is preserved after error...")
        result = session.run("print(f'x = {x}, y = {y}')")
        print(f"   Output: {result.stdout.strip()}")
        print("   State preserved successfully!")


def demo_ai_agent_workflow() -> None:
    """Demonstrate a typical AI agent workflow."""
    print("\n" + "=" * 60)
    print("Demo 8: AI Agent Workflow Simulation")
    print("=" * 60)

    with InteractiveSandboxSession(lang="python", client=client) as session:
        # Agent explores environment
        print("\n1. Agent explores the environment...")
        result = session.run("%pwd")
        print(f"   Current directory: {result.stdout.strip()}")

        # Agent creates some data
        print("\n2. Agent creates initial data...")
        session.run(
            textwrap.dedent("""
            data = {
                'temperatures': [20, 22, 19, 23, 21, 24, 22],
                'days': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            }
            print(f"Created temperature data for {len(data['days'])} days")
        """)
        )

        # Agent analyzes the data
        print("\n3. Agent analyzes the data...")
        result = session.run(
            textwrap.dedent("""
            avg_temp = sum(data['temperatures']) / len(data['temperatures'])
            max_temp = max(data['temperatures'])
            min_temp = min(data['temperatures'])

            print(f"Average temperature: {avg_temp:.1f}°C")
            print(f"Max temperature: {max_temp}°C")
            print(f"Min temperature: {min_temp}°C")
        """)
        )
        print(f"   {result.stdout.strip()}")

        # Agent makes a decision
        print("\n4. Agent makes a decision...")
        result = session.run(
            textwrap.dedent("""
            if avg_temp > 22:
                decision = "It's warm - recommend light clothing"
            elif avg_temp > 18:
                decision = "It's moderate - recommend layered clothing"
            else:
                decision = "It's cool - recommend warm clothing"

            print(f"Decision: {decision}")
        """)
        )
        print(f"   {result.stdout.strip()}")


def main() -> None:
    """Run all interactive session demos."""
    print("\n" + "=" * 60)
    print("LLM Sandbox - Interactive Session Demo")
    print("=" * 60)
    print("\nThis demo showcases InteractiveSandboxSession features:")
    print("- State persistence across multiple run() calls")
    print("- Function and class definitions")
    print("- IPython magic commands")
    print("- Multi-step data analysis workflows")
    print("- Error handling with state preservation")
    print("- AI agent simulation")

    try:
        # Run all demos
        demo_basic_state_persistence()
        demo_function_definitions()
        demo_class_definitions()
        demo_ipython_magic()
        demo_multi_step_data_analysis()
        demo_custom_settings()
        demo_error_handling()
        demo_ai_agent_workflow()

        # Summary
        print("\n" + "=" * 60)
        print("Demo Complete!")
        print("=" * 60)

    except Exception as e:  # noqa: BLE001
        print(f"\nError running demo: {e}")
        print("Make sure Docker is running and you have the required dependencies.")
        print("Install with: pip install 'llm-sandbox[docker]'")


if __name__ == "__main__":
    main()
