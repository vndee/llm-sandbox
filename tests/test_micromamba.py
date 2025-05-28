# ruff: noqa: SLF001, PLR2004, ARG002

"""Tests for Micromamba backend implementation."""

from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.data import ConsoleOutput
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

        assert session.lang == SupportedLanguage.PYTHON
        assert session.verbose is False
        assert session.image == "mambaorg/micromamba:latest"
        assert session.keep_template is False
        assert session.commit_container is False
        assert session.stream is True
        assert session.workdir == "/sandbox"
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

        assert session.image == "custom-micromamba:latest"
        assert session.lang == "java"
        assert session.keep_template is True
        assert session.commit_container is True
        assert session.verbose is True
        assert session.environment == "custom_env"
        assert session.stream is False
        assert session.workdir == "/custom"
        assert session.security_policy == security_policy

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

        from docker.types import Mount

        test_mounts = [Mount("/host/path", "/container/path", type="bind")]
        runtime_configs = {"cpu_count": 2, "mem_limit": "1g"}

        session = MicromambaSession(
            mounts=test_mounts,
            runtime_configs=runtime_configs,
            extra_param="should_be_passed",  # type: ignore[arg-type] # Test **kwargs
        )

        assert session.mounts == test_mounts
        assert session.runtime_configs == runtime_configs

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_init_with_dockerfile(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test initialization with dockerfile."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession(dockerfile="/path/to/Dockerfile", environment="data_science")

        assert session.dockerfile == "/path/to/Dockerfile"
        assert session.image is None
        assert session.environment == "data_science"


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

        session = MicromambaSession(environment="test_env")

        # Mock the parent execute_command method
        with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
            expected_result = ConsoleOutput(exit_code=0, stdout="output")
            mock_parent_execute.return_value = expected_result

            result = session.execute_command("python --version", workdir="/tmp")

            assert result == expected_result
            mock_parent_execute.assert_called_once_with("micromamba run -n test_env python --version", "/tmp")

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_with_default_environment(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test execute_command with default base environment."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession()  # Default environment is "base"

        with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
            expected_result = ConsoleOutput(exit_code=0, stdout="output")
            mock_parent_execute.return_value = expected_result

            result = session.execute_command("conda list")

            assert result == expected_result
            mock_parent_execute.assert_called_once_with("micromamba run -n base conda list", None)

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_with_none_command(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test execute_command with None command."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession()

        with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
            expected_result = ConsoleOutput(exit_code=0, stdout="output")
            mock_parent_execute.return_value = expected_result

            result = session.execute_command(None)

            assert result == expected_result
            mock_parent_execute.assert_called_once_with("micromamba run -n base ", None)

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_with_empty_command(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test execute_command with empty command."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession(environment="ml_env")

        with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
            expected_result = ConsoleOutput(exit_code=0, stdout="output")
            mock_parent_execute.return_value = expected_result

            result = session.execute_command("", workdir="/workspace")

            assert result == expected_result
            mock_parent_execute.assert_called_once_with("micromamba run -n ml_env ", "/workspace")

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_complex_command(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test execute_command with complex command containing pipes and redirects."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession(environment="data_analysis")

        with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
            expected_result = ConsoleOutput(exit_code=0, stdout="output")
            mock_parent_execute.return_value = expected_result

            complex_command = "python script.py | grep 'result' > output.txt"
            result = session.execute_command(complex_command, workdir="/data")

            assert result == expected_result
            mock_parent_execute.assert_called_once_with(f"micromamba run -n data_analysis {complex_command}", "/data")

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_with_quotes_and_special_chars(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test execute_command with commands containing quotes and special characters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession(environment="test")

        with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
            expected_result = ConsoleOutput(exit_code=0, stdout="output")
            mock_parent_execute.return_value = expected_result

            special_command = '''python -c "print('Hello, World!')" && echo "Done"'''
            result = session.execute_command(special_command)

            assert result == expected_result
            mock_parent_execute.assert_called_once_with(f"micromamba run -n test {special_command}", None)

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_preserves_parent_functionality(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that execute_command preserves all parent class functionality."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession()

        # Test that parent's execute_command is called with correct parameters
        with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
            # Mock parent returning different exit codes and outputs
            test_cases = [
                ConsoleOutput(exit_code=0, stdout="success", stderr=""),
                ConsoleOutput(exit_code=1, stdout="", stderr="error occurred"),
                ConsoleOutput(exit_code=127, stdout="command not found", stderr="bash: command not found"),
            ]

            for expected_result in test_cases:
                mock_parent_execute.return_value = expected_result
                result = session.execute_command("test_command")
                assert result == expected_result

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_execute_command_environment_name_with_special_chars(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test execute_command with environment names containing special characters."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        # Test various environment name formats
        test_environments = [
            "my-env",
            "env_with_underscores",
            "env123",
            "ENV_UPPER",
            "env.with.dots",
        ]

        for env_name in test_environments:
            session = MicromambaSession(environment=env_name)

            with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
                expected_result = ConsoleOutput(exit_code=0, stdout="output")
                mock_parent_execute.return_value = expected_result

                result = session.execute_command("python --version")

                assert result == expected_result
                mock_parent_execute.assert_called_once_with(f"micromamba run -n {env_name} python --version", None)


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

        # Test that all expected properties are available
        expected_properties = [
            "lang",
            "verbose",
            "image",
            "keep_template",
            "commit_container",
            "stream",
            "workdir",
            "security_policy",
            "client",
            "dockerfile",
            "is_create_template",
            "mounts",
            "runtime_configs",
        ]

        for prop_name in expected_properties:
            assert hasattr(session, prop_name), f"Property {prop_name} should be inherited"

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_run_method_uses_overridden_execute_command(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test that the run method uses the overridden execute_command."""
        mock_handler = MagicMock()
        mock_handler.file_extension = "py"
        mock_handler.get_execution_commands.return_value = ["python /sandbox/code.py"]
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession(environment="ml_env")

        # Mock container to avoid NotOpenSessionError
        mock_container = MagicMock()
        session.container = mock_container

        with (
            patch.object(session, "install"),
            patch.object(session, "copy_to_runtime"),
            patch("tempfile.NamedTemporaryFile") as mock_temp_file,
            patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute,
        ):
            # Setup mocks
            mock_temp_file.return_value.__enter__ = lambda x: x
            mock_temp_file.return_value.__exit__ = lambda *args: None  # noqa: ARG005
            mock_temp_file.return_value.name = "/tmp/code.py"
            mock_temp_file.return_value.write = MagicMock()
            mock_temp_file.return_value.seek = MagicMock()

            expected_result = ConsoleOutput(exit_code=0, stdout="output")
            mock_parent_execute.return_value = expected_result

            result = session.run("print('hello')", ["numpy"])

            # Verify that the command was wrapped with micromamba run
            mock_parent_execute.assert_called_once_with("micromamba run -n ml_env python /sandbox/code.py", "/sandbox")
            assert result == expected_result


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
        assert session.image == "mambaorg/micromamba:latest"
        assert session.workdir == "/workspace"

        # Test that commands are properly wrapped
        with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
            expected_result = ConsoleOutput(exit_code=0, stdout="Python 3.9.0")
            mock_parent_execute.return_value = expected_result

            _ = session.execute_command("python --version")

            mock_parent_execute.assert_called_once_with("micromamba run -n data_science python --version", None)

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_default_image_is_micromamba(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test that the default image is mambaorg/micromamba:latest."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession()

        assert session.image == "mambaorg/micromamba:latest"

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

            # Verify it's used in command execution
            with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
                mock_parent_execute.return_value = ConsoleOutput(exit_code=0, stdout="")
                session.execute_command("test")
                mock_parent_execute.assert_called_once_with(f"micromamba run -n {env_name} test", None)


class TestMicromambaSessionEdgeCases:
    """Test edge cases and error conditions for MicromambaSession."""

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_empty_environment_name(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test behavior with empty environment name."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession(environment="")

        with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
            mock_parent_execute.return_value = ConsoleOutput(exit_code=0, stdout="")
            session.execute_command("test_command")

            # Should still wrap with micromamba run, even with empty env name
            mock_parent_execute.assert_called_once_with("micromamba run -n  test_command", None)

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
            with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
                mock_parent_execute.side_effect = exception

                with pytest.raises(type(exception)):
                    session.execute_command("test")

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_very_long_commands(self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock) -> None:
        """Test execution of very long commands."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        session = MicromambaSession(environment="test")

        # Create a very long command
        long_command = "python -c " + "'print(" + "A" * 1000 + ")'"

        with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
            mock_parent_execute.return_value = ConsoleOutput(exit_code=0, stdout="")

            session.execute_command(long_command)

            expected_full_command = f"micromamba run -n test {long_command}"
            mock_parent_execute.assert_called_once_with(expected_full_command, None)

    @patch("llm_sandbox.micromamba.docker.from_env")
    @patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler")
    def test_unicode_in_commands_and_environment(
        self, mock_create_handler: MagicMock, mock_docker_from_env: MagicMock
    ) -> None:
        """Test handling of Unicode characters in commands and environment names."""
        mock_handler = MagicMock()
        mock_create_handler.return_value = mock_handler

        # Test Unicode environment name
        session = MicromambaSession(environment="тест")

        with patch("llm_sandbox.docker.SandboxDockerSession.execute_command") as mock_parent_execute:
            mock_parent_execute.return_value = ConsoleOutput(exit_code=0, stdout="")

            # Test Unicode command
            unicode_command = "python -c 'print(\"Hello, 世界!\")'"
            session.execute_command(unicode_command)

            expected_command = f"micromamba run -n тест {unicode_command}"
            mock_parent_execute.assert_called_once_with(expected_command, None)


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

        assert session.security_policy == security_policy
        assert session.environment == "secure_env"

        # Test that security policy methods are inherited
        assert hasattr(session, "is_safe")
        assert hasattr(session, "_check_security_policy")
