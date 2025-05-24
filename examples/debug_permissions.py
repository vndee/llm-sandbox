#!/usr/bin/env python3
"""Debug script to understand permission issues."""

from llm_sandbox import SandboxSession


def debug_file_permissions():
    """Debug what's happening with file permissions."""
    print("=== Debugging File Permissions ===")

    with SandboxSession(
        lang="python",
        runtime_configs={"user": "1000:1000"},
        workdir="/tmp/sandbox",
        verbose=True,
    ) as session:
        # Check current user
        result = session.execute_command("id")
        print(f"Current user info: {result.stdout.strip()}")

        # Check if user 1000 exists
        result = session.execute_command(
            "getent passwd 1000 2>/dev/null || echo 'User 1000 does not exist'"
        )
        print(f"User 1000 existence: {result.stdout.strip()}")

        # Try to run some code to see where it fails
        print("\n=== Attempting to run code ===")
        try:
            result = session.run("print('Hello from user 1000!')")
            print(f"Code execution result: {result.stdout.strip()}")
            print(f"Any errors: {result.stderr.strip()}")
        except Exception as e:
            print(f"Code execution failed: {e}")

        # Check workdir after attempting to run code
        result = session.execute_command("ls -la /tmp/")
        print(f"Contents of /tmp/ after run attempt:\n{result.stdout}")

        # Check if sandbox dir exists and its permissions
        result = session.execute_command(
            "test -d /tmp/sandbox && ls -la /tmp/sandbox/ || echo 'Directory does not exist'"
        )
        print(f"Contents of /tmp/sandbox/:\n{result.stdout}")

        # Check specific file if it exists
        result = session.execute_command(
            "test -f /tmp/sandbox/code.py && ls -la /tmp/sandbox/code.py || echo 'code.py does not exist'"
        )
        print(f"code.py file info:\n{result.stdout}")

        # Try to manually create directory and see if we can write
        print("\n=== Manual directory creation test ===")
        result = session.execute_command(
            "mkdir -p /tmp/sandbox_test && echo 'Directory created successfully'"
        )
        print(f"Manual mkdir result: {result.stdout.strip()}")

        result = session.execute_command(
            "echo 'test content' > /tmp/sandbox_test/test.txt && echo 'File created successfully'"
        )
        print(f"Manual file creation: {result.stdout.strip()}")

        result = session.execute_command("ls -la /tmp/sandbox_test/")
        print(f"Manual test directory contents:\n{result.stdout}")


if __name__ == "__main__":
    debug_file_permissions()
