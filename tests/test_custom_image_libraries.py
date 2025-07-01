"""Tests for custom Docker image libraries support (Issue #79).

This module tests that Python libraries pre-installed in custom Docker images
are accessible without needing to specify them in the libraries parameter.

Note: This issue is Python-specific due to virtual environment isolation.
Other languages (Go, R, Java, etc.) don't have this issue.
"""

from unittest.mock import MagicMock, patch

from llm_sandbox import SandboxSession
from llm_sandbox.data import ConsoleOutput


class TestCustomImageLibraries:
    """Test custom Docker image libraries functionality."""

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_custom_image_with_preinstalled_libraries(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that custom images with pre-installed libraries work without libraries parameter."""
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler.name = "python"
        mock_handler.file_extension = "py"
        mock_handler.get_execution_commands.return_value = ["/tmp/venv/bin/python test.py"]
        mock_handler.is_support_library_installation = True
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        mock_image = MagicMock()
        mock_image.tags = ["custom-python:latest"]
        mock_client.images.get.return_value = mock_image

        mock_container = MagicMock()
        mock_client.containers.create.return_value = mock_container

        # Mock successful execution
        mock_container.exec_run.return_value = MagicMock(exit_code=0, output=(b'{"a":[1,2,3],"b":[4,5,6]}\n', None))

        # Test with custom image (should work without libraries parameter)
        session = SandboxSession(client=mock_client, lang="python", image="custom-python:latest", verbose=True)

        with (
            patch.object(session, "copy_to_runtime"),
            patch.object(session, "execute_command") as mock_exec,
            patch("tempfile.NamedTemporaryFile"),
        ):
            # Mock environment setup commands
            mock_exec.side_effect = [
                ConsoleOutput(exit_code=0, stdout="", stderr=""),  # mkdir
                ConsoleOutput(exit_code=0, stdout="", stderr=""),  # venv creation
                ConsoleOutput(exit_code=0, stdout="", stderr=""),  # pip cache
                ConsoleOutput(exit_code=0, stdout="", stderr=""),  # pip upgrade
                ConsoleOutput(exit_code=0, stdout='{"a":[1,2,3],"b":[4,5,6]}', stderr=""),  # code execution
            ]

            session.open()

            # This should work without specifying pandas in libraries
            # because it's pre-installed in the custom image
            result = session.run("""
import pandas as pd
import numpy as np
df = pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})
print(df.to_json())
""")

            session.close()

        # Verify the code executed successfully
        assert result.exit_code == 0
        assert not any(
            "pip install pandas" in call[0][0] for call in mock_exec.call_args_list
        )  # No pandas installation

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_custom_dockerfile_with_preinstalled_libraries(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that custom Dockerfiles with pre-installed libraries work."""
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler.name = "python"
        mock_handler.file_extension = "py"
        mock_handler.get_execution_commands.return_value = ["/tmp/venv/bin/python test.py"]
        mock_handler.is_support_library_installation = True
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        mock_image = MagicMock()
        mock_image.tags = ["sandbox-python-custom"]
        mock_client.images.build.return_value = (mock_image, [])

        mock_container = MagicMock()
        mock_client.containers.create.return_value = mock_container

        # Test with custom dockerfile
        session = SandboxSession(client=mock_client, lang="python", dockerfile="./custom/Dockerfile", verbose=True)

        with (
            patch.object(session, "copy_to_runtime"),
            patch.object(session, "execute_command") as mock_exec,
            patch("tempfile.NamedTemporaryFile"),
            patch("llm_sandbox.docker.Path") as mock_path,
        ):
            # Mock Path behavior for dockerfile
            mock_dockerfile_path = MagicMock()
            mock_dockerfile_path.parent = "/custom"
            mock_dockerfile_path.name = "Dockerfile"
            mock_path.return_value = mock_dockerfile_path

            # Mock successful execution
            mock_exec.side_effect = [
                ConsoleOutput(exit_code=0, stdout="", stderr=""),  # mkdir
                ConsoleOutput(exit_code=0, stdout="", stderr=""),  # venv creation
                ConsoleOutput(exit_code=0, stdout="", stderr=""),  # pip cache
                ConsoleOutput(exit_code=0, stdout="", stderr=""),  # pip upgrade
                ConsoleOutput(exit_code=0, stdout="3", stderr=""),  # code execution
            ]

            session.open()

            # This should work because numpy is pre-installed in the Dockerfile
            result = session.run("""
import numpy as np
arr = np.array([1, 2, 3])
print(len(arr))
""")

            session.close()

        # Verify the code executed successfully
        assert result.exit_code == 0
        # Verify the image was built from dockerfile
        mock_client.images.build.assert_called_once()

    def test_venv_command_includes_system_site_packages(self) -> None:
        """Test that the virtual environment creation command includes --system-site-packages."""
        from llm_sandbox.core.session_base import PYTHON_CREATE_VENV_COMMAND

        # This ensures the fix for issue #79 is in place
        assert "--system-site-packages" in PYTHON_CREATE_VENV_COMMAND
        # Ensure the command includes the required flag without enforcing an exact match
        assert "/tmp/venv" in PYTHON_CREATE_VENV_COMMAND

    @patch("llm_sandbox.docker.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_libraries_parameter_still_works_with_custom_image(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that the libraries parameter still works when using custom images."""
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler.name = "python"
        mock_handler.file_extension = "py"
        mock_handler.get_execution_commands.return_value = ["/tmp/venv/bin/python test.py"]
        mock_handler.get_library_installation_command.return_value = (
            "/tmp/venv/bin/pip install --cache-dir /tmp/pip_cache requests"
        )
        mock_handler.is_support_library_installation = True
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        mock_image = MagicMock()
        mock_image.tags = ["custom-python:latest"]
        mock_client.images.get.return_value = mock_image

        mock_container = MagicMock()
        mock_client.containers.create.return_value = mock_container

        session = SandboxSession(client=mock_client, lang="python", image="custom-python:latest")

        with (
            patch.object(session, "copy_to_runtime"),
            patch.object(session, "execute_command") as mock_exec,
            patch("tempfile.NamedTemporaryFile"),
        ):
            mock_exec.side_effect = [
                ConsoleOutput(exit_code=0, stdout="", stderr=""),  # mkdir
                ConsoleOutput(exit_code=0, stdout="", stderr=""),  # venv creation
                ConsoleOutput(exit_code=0, stdout="", stderr=""),  # pip cache
                ConsoleOutput(exit_code=0, stdout="", stderr=""),  # pip upgrade
                ConsoleOutput(exit_code=0, stdout="", stderr=""),  # library installation
                ConsoleOutput(exit_code=0, stdout="2.28.1", stderr=""),  # code execution
            ]

            session.open()

            # This should still install additional libraries when specified
            result = session.run(
                """
import requests
print(requests.__version__)
""",
                libraries=["requests"],
            )

            session.close()

        # Verify library installation was called
        install_calls = [call for call in mock_exec.call_args_list if call[0][0] and "pip install" in call[0][0]]
        assert len(install_calls) >= 1  # At least one pip install call
        assert result.exit_code == 0
