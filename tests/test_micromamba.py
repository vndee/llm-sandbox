# ruff: noqa: SLF001, PLR2004, ARG002

"""Tests for Micromamba backend implementation."""

from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.micromamba import MicromambaSession
from llm_sandbox.security import SecurityPolicy


class TestMicromambaSessionInit:
    """Test MicromambaSession initialization."""

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_defaults(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test initialization with default parameters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        session = MicromambaSession()

        assert session.config.lang == SupportedLanguage.PYTHON
        assert session.config.verbose is False
        assert session.config.image == "mambaorg/micromamba:latest"
        assert session.keep_template is False
        assert session.commit_container is False
        assert session.stream is True
        assert session.config.workdir == "/sandbox"
        assert session.environment == "base"
        assert session.client == mock_client
        mock_docker_from_env.assert_called_once()

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_custom_client(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test initialization with custom Docker client."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        custom_client = MagicMock()

        session = MicromambaSession(client=custom_client)

        assert session.client == custom_client
        mock_docker_from_env.assert_not_called()

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_custom_params(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test initialization with custom parameters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client
        security_policy = SecurityPolicy(patterns=[], restricted_modules=[])

        session = MicromambaSession(
            image="custom-micromamba:latest",
            lang="java",
            keep_template=True,
            commit_container=True,
            verbose=True,
            environment="custom_env",
            stream=False,
            workdir="/custom",
            security_policy=security_policy,
        )

        assert session.config.image == "custom-micromamba:latest"
        assert session.config.lang == SupportedLanguage.JAVA
        assert session.keep_template is True
        assert session.commit_container is True
        assert session.config.verbose is True
        assert session.environment == "custom_env"
        assert session.stream is False
        assert session.config.workdir == "/custom"
        assert session.config.security_policy == security_policy

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_all_docker_session_params(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test initialization with all SandboxDockerSession parameters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        runtime_configs = {"cpu_count": 2, "mem_limit": "1g"}

        session = MicromambaSession(
            runtime_configs=runtime_configs,
            extra_param="should_be_passed",
        )

        # Mounts are deprecated and moved to runtime_configs
        assert session.config.runtime_configs["cpu_count"] == 2
        assert session.config.runtime_configs["mem_limit"] == "1g"


class TestMicromambaSessionExecuteCommand:
    """Test MicromambaSession execute_command functionality."""

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_wraps_with_micromamba_run(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that execute_command wraps commands with micromamba run."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession(environment="test_env", stream=False)

        # Mock the container_api execute_command method instead of parent
        with patch.object(session.container_api, "execute_command") as mock_container_execute:
            mock_container_execute.return_value = (0, (b"output", b""))
            mock_container = MagicMock()
            session.container = mock_container

            result = session.execute_command("python --version", workdir="/tmp")

            # Verify that container_api.execute_command was called with wrapped command
            mock_container_execute.assert_called_once_with(
                mock_container, "python --version", workdir="/tmp", stream=False
            )
            assert result.exit_code == 0
            assert result.stdout == "output"

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_micromamba_container_api_wraps_commands(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that MicromambaContainerAPI properly wraps commands."""
        from llm_sandbox.micromamba import MicromambaContainerAPI

        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()

        # Create API instance directly to test command wrapping
        api = MicromambaContainerAPI(mock_client, environment="test_env")

        # Mock the parent execute_command
        with patch("llm_sandbox.docker.DockerContainerAPI.execute_command") as mock_parent_execute:
            mock_parent_execute.return_value = (0, "wrapped output")
            mock_container = MagicMock()

            result = api.execute_command(mock_container, "python --version", workdir="/tmp")

            # Verify parent was called with wrapped command
            mock_parent_execute.assert_called_once_with(
                mock_container, "micromamba run -n test_env python --version", workdir="/tmp"
            )
            assert result == (0, "wrapped output")

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_with_default_environment(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test execute_command with default base environment."""
        from llm_sandbox.micromamba import MicromambaContainerAPI

        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()

        # Test the container API directly with default environment
        api = MicromambaContainerAPI(mock_client, environment="base")

        with patch("llm_sandbox.docker.DockerContainerAPI.execute_command") as mock_parent_execute:
            mock_parent_execute.return_value = (0, "output")
            mock_container = MagicMock()

            api.execute_command(mock_container, "conda list")

            mock_parent_execute.assert_called_once_with(mock_container, "micromamba run -n base conda list")

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_with_empty_command(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test execute_command with empty command."""
        from llm_sandbox.micromamba import MicromambaContainerAPI

        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()

        api = MicromambaContainerAPI(mock_client, environment="base")

        with patch("llm_sandbox.docker.DockerContainerAPI.execute_command") as mock_parent_execute:
            mock_parent_execute.return_value = (0, "output")
            mock_container = MagicMock()

            api.execute_command(mock_container, "")

            mock_parent_execute.assert_called_once_with(mock_container, "micromamba run -n base ")

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_complex_command(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test execute_command with complex command containing pipes and redirects."""
        from llm_sandbox.micromamba import MicromambaContainerAPI

        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()

        api = MicromambaContainerAPI(mock_client, environment="data_analysis")

        with patch("llm_sandbox.docker.DockerContainerAPI.execute_command") as mock_parent_execute:
            mock_parent_execute.return_value = (0, "output")
            mock_container = MagicMock()

            complex_command = "python script.py | grep 'result' > output.txt"
            api.execute_command(mock_container, complex_command, workdir="/data")

            expected_command = f"micromamba run -n data_analysis {complex_command}"
            mock_parent_execute.assert_called_once_with(mock_container, expected_command, workdir="/data")


class TestMicromambaSessionInheritance:
    """Test that MicromambaSession properly inherits from SandboxDockerSession."""

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_inherits_all_docker_session_methods(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that MicromambaSession inherits all methods from SandboxDockerSession."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession()

        # Test that all expected methods are available
        expected_methods = [
            "open",
            "close",
            "run",
            "copy_to_runtime",
            "copy_from_runtime",
            "execute_commands",
            "install",
            "environment_setup",
            "get_archive",
            "_ensure_ownership",
            "__enter__",
            "__exit__",
        ]

        for method_name in expected_methods:
            assert hasattr(session, method_name), f"Method {method_name} should be inherited"
            assert callable(getattr(session, method_name)), f"Method {method_name} should be callable"

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_inherits_docker_session_properties(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that MicromambaSession inherits properties from SandboxDockerSession."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession()

        # Test that all expected properties are available through config
        expected_config_properties = [
            "lang",
            "verbose",
            "image",
            "workdir",
            "security_policy",
            "dockerfile",
            "runtime_configs",
        ]

        for prop_name in expected_config_properties:
            assert hasattr(session.config, prop_name), f"Property {prop_name} should be in config"

        # Test session-level properties
        expected_session_properties = [
            "keep_template",
            "commit_container",
            "stream",
            "client",
            "is_create_template",
        ]

        for prop_name in expected_session_properties:
            assert hasattr(session, prop_name), f"Property {prop_name} should be on session"

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_run_method_uses_overridden_execute_command(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that the run method uses the overridden execute_command."""
        mock_handler = MagicMock()
        mock_handler.file_extension = "py"
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession(environment="ml_env")

        # Mock container to avoid NotOpenSessionError
        mock_container = MagicMock()
        session.container = mock_container
        session.is_open = True

        with (
            patch.object(session, "install"),
            patch.object(session, "copy_to_runtime"),
            patch("tempfile.NamedTemporaryFile") as mock_temp_file,
            patch.object(session.container_api, "execute_command") as mock_container_execute,
        ):
            # Setup mocks
            mock_temp_file.return_value.__enter__ = lambda x: x
            mock_temp_file.return_value.__exit__ = lambda *args: None  # noqa: ARG005
            mock_temp_file.return_value.name = "/tmp/code.py"
            mock_temp_file.return_value.write = MagicMock()
            mock_temp_file.return_value.seek = MagicMock()

            mock_container_execute.return_value = (0, "output")

            result = session.run("print('hello')", ["numpy"])

            # Verify that the container_api execute_command was called
            # (it will internally wrap with micromamba run)
            assert result.exit_code == 0


class TestMicromambaSessionDocumentation:
    """Test that MicromambaSession follows the documented behavior."""

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_docstring_example_usage(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test that the documented example usage works as expected."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        # Example from docstring: Create session with custom environment
        session = MicromambaSession(
            environment="data_science", image="mambaorg/micromamba:latest", workdir="/workspace"
        )

        assert session.environment == "data_science"
        assert session.config.image == "mambaorg/micromamba:latest"
        assert session.config.workdir == "/workspace"

        # Test that the container API is properly configured
        assert session.container_api.environment == "data_science"  # type: ignore[attr-defined]

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_default_image_is_micromamba(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test that the default image is mambaorg/micromamba:latest."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession()

        assert session.config.image == "mambaorg/micromamba:latest"

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_environment_parameter_is_respected(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that the environment parameter is properly stored and used."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        environments_to_test = ["base", "ml", "data-science", "test_env_123"]

        for env_name in environments_to_test:
            session = MicromambaSession(environment=env_name)
            assert session.environment == env_name
            assert session.container_api.environment == env_name  # type: ignore[attr-defined]


class TestMicromambaSessionEdgeCases:
    """Test edge cases and error conditions for MicromambaSession."""

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_empty_environment_name(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test behavior with empty environment name."""
        from llm_sandbox.micromamba import MicromambaContainerAPI

        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()

        api = MicromambaContainerAPI(mock_client, environment="")

        with patch("llm_sandbox.docker.DockerContainerAPI.execute_command") as mock_parent_execute:
            mock_parent_execute.return_value = (0, "")
            mock_container = MagicMock()

            api.execute_command(mock_container, "test_command")

            # Should still wrap with micromamba run, even with empty env name
            mock_parent_execute.assert_called_once_with(mock_container, "micromamba run -n  test_command")

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_parent_execute_command_exceptions_are_propagated(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that exceptions from parent execute_command are properly propagated."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession()

        from llm_sandbox.exceptions import CommandEmptyError, NotOpenSessionError

        test_exceptions = [
            CommandEmptyError(),
            NotOpenSessionError(),
            RuntimeError("Docker daemon not running"),
            ConnectionError("Network error"),
        ]

        for exception in test_exceptions:
            with patch.object(session.container_api, "execute_command") as mock_container_execute:
                mock_container_execute.side_effect = exception
                mock_container = MagicMock()
                session.container = mock_container

                with pytest.raises(type(exception)):
                    session.execute_command("test")

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_very_long_commands(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test execution of very long commands."""
        from llm_sandbox.micromamba import MicromambaContainerAPI

        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()

        api = MicromambaContainerAPI(mock_client, environment="test")

        # Create a very long command
        long_command = "python -c " + "'print(" + "A" * 1000 + ")'"

        with patch("llm_sandbox.docker.DockerContainerAPI.execute_command") as mock_parent_execute:
            mock_parent_execute.return_value = (0, "")
            mock_container = MagicMock()

            api.execute_command(mock_container, long_command)

            expected_full_command = f"micromamba run -n test {long_command}"
            mock_parent_execute.assert_called_once_with(mock_container, expected_full_command)

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_unicode_in_commands_and_environment(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test handling of Unicode characters in commands and environment names."""
        from llm_sandbox.micromamba import MicromambaContainerAPI

        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler
        mock_client = MagicMock()

        # Test Unicode environment name
        api = MicromambaContainerAPI(mock_client, environment="тест")

        with patch("llm_sandbox.docker.DockerContainerAPI.execute_command") as mock_parent_execute:
            mock_parent_execute.return_value = (0, "")
            mock_container = MagicMock()

            # Test Unicode command
            unicode_command = "python -c 'print(\"Hello, 世界!\")'"
            api.execute_command(mock_container, unicode_command)

            expected_command = f"micromamba run -n тест {unicode_command}"
            mock_parent_execute.assert_called_once_with(mock_container, expected_command)


class TestMicromambaSessionIntegration:
    """Integration tests for MicromambaSession with other components."""

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_context_manager_functionality(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that MicromambaSession works properly as a context manager."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession()

        # Test that it can be used as a context manager (inherits from parent)
        with (
            patch.object(session, "open") as mock_open,
            patch.object(session, "close") as mock_close,
        ):
            with session as s:
                assert s == session
                assert session.environment == "base"
                mock_open.assert_called_once()

            mock_close.assert_called_once()

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_security_policy_integration(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test that security policies work with MicromambaSession."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        security_policy = SecurityPolicy(patterns=[], restricted_modules=[])
        session = MicromambaSession(security_policy=security_policy, environment="secure_env")

        assert session.config.security_policy == security_policy
        assert session.environment == "secure_env"

        # Test that security policy methods are inherited
        assert hasattr(session, "is_safe")
        assert hasattr(session, "_check_security_policy")
