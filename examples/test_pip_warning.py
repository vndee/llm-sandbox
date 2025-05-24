#!/usr/bin/env python3
"""Test script to verify pip warnings are suppressed."""

from llm_sandbox import SandboxSession


def test_pip_no_warning():
    """Test that pip doesn't show root user warnings."""
    print("=== Testing Pip Root User Warning Suppression ===")

    with SandboxSession(lang="python", verbose=True) as session:
        # Test installing a simple package
        try:
            result = session.run(
                "print('Testing package installation...')",
                libraries=[
                    "requests"
                ],  # Common package that's likely not pre-installed
            )
            print(f"Installation result: {result.stdout.strip()}")
            print(f"Any warnings/errors: {result.stderr.strip()}")

            # Verify the package can be imported
            result = session.run("""
import requests
print("requests imported successfully!")
print(f"requests version: {requests.__version__}")
""")
            print(f"Import test: {result.stdout.strip()}")

        except Exception as e:
            print(f"Test failed: {e}")


def test_custom_user_pip():
    """Test pip with custom user (should still work)."""
    print("\n=== Testing Pip with Custom User ===")

    try:
        with SandboxSession(
            lang="python",
            runtime_configs={"user": "1000:1000"},
            workdir="/tmp/sandbox",
            verbose=True,
        ) as session:
            # Test installing a package as non-root user
            result = session.run(
                "print('Testing non-root package installation...')",
                libraries=["urllib3"],
            )
            print(f"Custom user installation: {result.stdout.strip()}")
            print(f"Any errors: {result.stderr.strip()}")

    except Exception as e:
        print(f"Custom user test failed: {e}")


if __name__ == "__main__":
    test_pip_no_warning()
    test_custom_user_pip()
    print("\n=== Pip warning test completed ===")
