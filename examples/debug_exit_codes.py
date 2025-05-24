#!/usr/bin/env python3
"""Debug exit codes for virtual environment tests."""

from llm_sandbox import SandboxSession


def debug_exit_codes():
    """Debug what exit codes we're getting."""
    print("=== Debugging Exit Codes ===")

    with SandboxSession(backend="docker", lang="python", verbose=True) as session:
        result = session.run("print('Test successful!')", libraries=["urllib3"])
        print(f"Exit code: {result.exit_code}")
        print(f"Exit code type: {type(result.exit_code)}")
        print(f"Stdout: {result.stdout!r}")
        print(f"Stderr: {result.stderr!r}")
        print(f"Comparison result.exit_code == 0: {result.exit_code == 0}")
        print(f"Comparison result.exit_code is None: {result.exit_code is None}")


if __name__ == "__main__":
    debug_exit_codes()
