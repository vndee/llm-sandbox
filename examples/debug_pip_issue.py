#!/usr/bin/env python3
"""Debug script for pip cache issue."""

from llm_sandbox import SandboxSession


def debug_pip_cache():
    """Debug pip cache directory issue."""
    print("=== Debugging Pip Cache Issue ===")

    with SandboxSession(
        lang="python",
        runtime_configs={"user": "1000:1000"},
        workdir="/tmp/sandbox",
        verbose=True,
    ) as session:
        # Check current user
        result = session.execute_command("whoami")
        print(f"Current user: {result.stdout.strip()}")

        # Check venv ownership
        result = session.execute_command("ls -la /tmp/venv/")
        print(f"Venv directory ownership:\n{result.stdout}")

        # Check pip cache ownership
        result = session.execute_command("ls -la /tmp/pip_cache/")
        print(f"Pip cache ownership:\n{result.stdout}")

        # Check what the pip command looks like
        result = session.execute_command("which pip")
        print(f"Pip location: {result.stdout.strip()}")

        result = session.execute_command("/tmp/venv/bin/pip --version")
        print(f"Pip version: {result.stdout.strip()}")

        # Check if pip respects our cache dir
        result = session.execute_command(
            "/tmp/venv/bin/pip install --cache-dir /tmp/pip_cache --dry-run urllib3"
        )
        print(f"Dry run result: {result.stdout.strip()}")
        print(f"Dry run errors: {result.stderr.strip()}")


if __name__ == "__main__":
    debug_pip_cache()
