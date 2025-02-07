"""Integration tests for the SandboxSession class."""

import os
import pytest
import tempfile
from pathlib import Path
from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import SecurityError, ResourceError

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)

class TestDockerIntegration:
    def test_full_session_lifecycle(self):
        with SandboxSession(
            backend="docker",
            image="python:3.9.19-bullseye",
            lang="python",
            keep_template=False,
            verbose=True,
        ) as session:
            result = session.run("print('Hello, World!')")
            assert result.exit_code == 0
            assert result.output.strip() == "Hello, World!"
            assert result.resource_usage is not None

    def test_custom_dockerfile(self, temp_dir):
        dockerfile_content = """
        FROM python:3.9.19-bullseye
        RUN pip install numpy pandas
        WORKDIR /sandbox
        """
        dockerfile_path = temp_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)

        with SandboxSession(
            backend="docker",
            dockerfile=str(dockerfile_path),
            lang="python",
            keep_template=True,
        ) as session:
            code = """
            import numpy as np
            import pandas as pd
            arr = np.array([1, 2, 3])
            print(f'Sum: {arr.sum()}')
            """
            result = session.run(code)
            assert result.exit_code == 0
            assert "Sum: 6" in result.output

    def test_file_operations(self, temp_dir):
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_content = "Hello from host!"
        test_file.write_text(test_content)

        with SandboxSession(
            backend="docker",
            image="python:3.9.19-bullseye",
            lang="python",
        ) as session:
            # Copy file to container
            session.copy_to_runtime(str(test_file), "/sandbox/test.txt")

            # Read and verify file content in container
            code = """
            with open('/sandbox/test.txt', 'r') as f:
                content = f.read()
            print(f'Content: {content}')
            """
            result = session.run(code)
            assert result.exit_code == 0
            assert test_content in result.output

            # Create a new file in container
            code = """
            with open('/sandbox/output.txt', 'w') as f:
                f.write('Hello from container!')
            """
            result = session.run(code)
            assert result.exit_code == 0

            # Copy file back to host
            output_file = temp_dir / "output.txt"
            session.copy_from_runtime("/sandbox/output.txt", str(output_file))
            assert output_file.read_text() == "Hello from container!"

    @pytest.mark.parametrize("lang,code,expected", [
        ("python", "print('Hello from Python!')", "Hello from Python!"),
        ("javascript", "console.log('Hello from Node!')", "Hello from Node!"),
        ("ruby", "puts 'Hello from Ruby!'", "Hello from Ruby!"),
    ])
    def test_multiple_languages(self, lang, code, expected):
        with SandboxSession(lang=lang) as session:
            result = session.run(code)
            assert result.exit_code == 0
            assert expected in result.output

    def test_resource_limits(self):
        with SandboxSession(
            lang="python",
            max_cpu_percent=50.0,
            max_memory_bytes=100 * 1024 * 1024,  # 100MB
        ) as session:
            # Test CPU-intensive operation
            cpu_code = """
            def fibonacci(n):
                if n <= 1:
                    return n
                return fibonacci(n-1) + fibonacci(n-2)
            
            print(fibonacci(35))  # CPU intensive
            """
            result = session.run(cpu_code)
            assert result.exit_code == 0
            assert result.resource_usage["cpu_percent"] <= 50.0

            # Test memory-intensive operation
            memory_code = """
            import numpy as np
            # Try to allocate more than allowed memory
            try:
                arr = np.zeros((200, 1024, 1024))  # Should be more than 100MB
            except Exception as e:
                print(f'Memory allocation failed: {e}')
            """
            with pytest.raises(ResourceError, match="Memory usage exceeded"):
                session.run(memory_code)

    def test_network_isolation(self):
        with SandboxSession(lang="python", strict_security=True) as session:
            network_code = """
            import socket
            try:
                socket.create_connection(('google.com', 80))
                print('Connected')
            except Exception as e:
                print(f'Connection failed: {e}')
            """
            with pytest.raises(SecurityError):
                session.run(network_code)

    def test_persistent_environment(self):
        with SandboxSession(
            lang="python",
            keep_template=True,
            image="python:3.9.19-bullseye"
        ) as session:
            # Install a package
            install_code = """
            import sys
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'requests'])
            """
            result = session.run(install_code)
            assert result.exit_code == 0

            # Use the installed package
            use_package_code = """
            import requests
            print('Package available!')
            """
            result = session.run(use_package_code)
            assert result.exit_code == 0
            assert "Package available!" in result.output

    def test_environment_variables(self):
        env_vars = {
            "TEST_VAR": "test_value",
            "PYTHON_ENV": "production"
        }
        
        with SandboxSession(
            lang="python",
            environment=env_vars
        ) as session:
            code = """
            import os
            print(f"TEST_VAR: {os.getenv('TEST_VAR')}")
            print(f"PYTHON_ENV: {os.getenv('PYTHON_ENV')}")
            """
            result = session.run(code)
            assert result.exit_code == 0
            assert "TEST_VAR: test_value" in result.output
            assert "PYTHON_ENV: production" in result.output 