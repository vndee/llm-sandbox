#!/usr/bin/env python3
"""Test to see if chmod is really needed or if chown alone is sufficient."""

from llm_sandbox import SandboxSession


def test_chown_only():
    """Test if chown alone is sufficient."""
    print("=== Testing chown-only approach ===")

    with SandboxSession(
        lang="python",
        runtime_configs={"user": "1000:1000"},
        workdir="/tmp/sandbox",
        verbose=True,
    ) as session:
        # Check current user
        result = session.execute_command("whoami")
        print(f"Current user: {result.stdout.strip()}")

        # Check what happens with normal file operations
        result = session.execute_command("echo 'test content' > /tmp/test_normal.txt")
        print(f"Normal file creation result: {result.exit_code}")

        result = session.execute_command("ls -la /tmp/test_normal.txt")
        print(f"Normal file permissions: {result.stdout.strip()}")

        # Try to run code to see original file permissions
        try:
            result = session.run("print('Testing chown-only!')")
            print(f"Code execution success: {result.stdout.strip()}")

            # Check the actual file permissions after our copy process
            result = session.execute_command("ls -la /tmp/sandbox/code.py")
            print(f"Code file permissions after copy: {result.stdout.strip()}")

            # Try to read the file as current user
            result = session.execute_command("cat /tmp/sandbox/code.py")
            print(f"File readable: {'Yes' if result.exit_code == 0 else 'No'}")
            if result.exit_code != 0:
                print(f"Cat error: {result.stderr.strip()}")

            # Check who owns the file and current user
            result = session.execute_command("stat -c '%U %G %n' /tmp/sandbox/code.py")
            print(f"File ownership: {result.stdout.strip()}")

            result = session.execute_command("id")
            print(f"Current user ID: {result.stdout.strip()}")

            # Try reading with different permissions check
            result = session.execute_command(
                "test -r /tmp/sandbox/code.py && echo 'readable' || echo 'not readable'"
            )
            print(f"Test -r result: {result.stdout.strip()}")

            # Check if Python can read it directly
            result = session.execute_command(
                "python3 -c \"with open('/tmp/sandbox/code.py', 'r') as f: print('Python can read:', len(f.read()), 'chars')\""
            )
            print(f"Python read test: {result.stdout.strip()}")
            if result.stderr:
                print(f"Python read error: {result.stderr.strip()}")

        except Exception as e:
            print(f"Code execution failed: {e}")


if __name__ == "__main__":
    test_chown_only()
