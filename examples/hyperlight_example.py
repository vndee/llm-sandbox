"""Example demonstrating Hyperlight backend usage.

This example shows how to use the Hyperlight backend for running code in
micro VMs instead of containers. Hyperlight provides faster startup times
and lighter weight isolation compared to container-based backends.

Requirements:
    - Rust toolchain (cargo, rustc)
    - KVM support (Linux) or Windows Hypervisor Platform (Windows)
    - Hyperlight library (automatically pulled from GitHub during build)

Note: This is an experimental backend. The first run will compile the guest
binary which can take several minutes. Subsequent runs will be faster.
"""

from llm_sandbox import SandboxBackend, create_session


def main() -> None:
    """Run example code using Hyperlight backend."""
    print("=" * 80)
    print("Hyperlight Backend Example")
    print("=" * 80)
    print()

    # Example 1: Basic usage with automatic guest binary compilation
    print("Example 1: Basic usage (first run compiles guest binary)")
    print("-" * 80)

    try:
        with create_session(
            backend=SandboxBackend.HYPERLIGHT, lang="python", verbose=True, keep_template=True
        ) as session:
            print("Session opened successfully!")

            # Note: Currently, the Hyperlight backend is in early development
            # and code execution is not yet fully implemented.
            # This example demonstrates the session creation and setup.

            print("\nSession properties:")
            print(f"  - Backend: Hyperlight")
            print(f"  - Language: {session.config.lang}")
            print(f"  - Verbose: {session.config.verbose}")
            print(f"  - Keep template: {session.config.keep_template}")

            print("\nNote: Code execution via Hyperlight is not yet fully implemented.")
            print("The backend currently supports session management and will execute")
            print("code via Rust-based guest binaries in future versions.")

    except Exception as e:
        print(f"Error: {e}")
        print("\nThis is expected if Rust toolchain is not installed.")
        print("Install Rust from: https://www.rust-lang.org/tools/install")

    print()

    # Example 2: Using pre-compiled guest binary
    print("Example 2: Using pre-compiled guest binary (faster startup)")
    print("-" * 80)
    print("If you have a pre-compiled guest binary, you can use it directly:")
    print()
    print("```python")
    print("with create_session(")
    print("    backend=SandboxBackend.HYPERLIGHT,")
    print("    lang='python',")
    print("    guest_binary_path='/path/to/precompiled/guest',")
    print("    verbose=True")
    print(") as session:")
    print("    # Use the session")
    print("    pass")
    print("```")
    print()

    # Example 3: Limitations and considerations
    print("Example 3: Hyperlight Backend Limitations")
    print("-" * 80)
    print("The Hyperlight backend has some limitations compared to containers:")
    print()
    print("1. Stateless Execution:")
    print("   - No persistent environment between runs")
    print("   - Each execution creates a new micro VM")
    print()
    print("2. Library Installation:")
    print("   - Libraries cannot be installed at runtime")
    print("   - Must be compiled into the guest binary")
    print()
    print("3. File Operations:")
    print("   - copy_to_runtime() and copy_from_runtime() not supported")
    print("   - VMs are stateless and destroyed after execution")
    print()
    print("4. Performance:")
    print("   - First run: Slow (guest binary compilation ~5 minutes)")
    print("   - Subsequent runs: Very fast (microsecond VM startup)")
    print()
    print("5. Platform Support:")
    print("   - Linux: Requires KVM or mshv")
    print("   - Windows: Requires Windows 11/Server 2022+ with WHP")
    print()

    # Example 4: When to use Hyperlight
    print("Example 4: When to Use Hyperlight Backend")
    print("-" * 80)
    print("Hyperlight is best suited for:")
    print()
    print("✓ Function-as-a-Service workloads with high request rates")
    print("✓ Scenarios requiring microsecond-level VM startup")
    print("✓ Trusted code that doesn't need library installation")
    print("✓ Maximum isolation with minimal overhead")
    print()
    print("Use container backends (Docker/Podman) for:")
    print()
    print("✓ Interactive development with library installation")
    print("✓ Persistent environments across multiple runs")
    print("✓ Complex workflows requiring file operations")
    print("✓ Standard containerized deployments")
    print()


if __name__ == "__main__":
    main()
