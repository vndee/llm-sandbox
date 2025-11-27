"""Tests for library pre-installation in container pools."""

from unittest.mock import MagicMock, patch

from llm_sandbox.const import SandboxBackend, SupportedLanguage
from llm_sandbox.core.session_base import BaseSession
from llm_sandbox.pool import PoolConfig, create_pool_manager


class TestPoolManagerLibrariesParameter:
    """Test that libraries are passed and installed during container creation."""

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_pool_manager_passes_libraries_to_session(
        self, mock_session_class: MagicMock, mock_docker_env: MagicMock
    ) -> None:
        """Verify that libraries passed to create_pool_manager are passed to the session."""
        mock_client = MagicMock()
        mock_docker_env.return_value = mock_client

        # Create pool manager with libraries
        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=PoolConfig(min_pool_size=1, max_pool_size=1, enable_prewarming=False),
            lang=SupportedLanguage.PYTHON,
            client=mock_client,
            libraries=["test-lib-1", "test-lib-2"],
        )

        # Manually call _create_session_for_container to see if it passes the libraries
        pool._create_session_for_container()

        # Verify SandboxDockerSession was initialized with libraries
        call_args = mock_session_class.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert "libraries" in kwargs
        assert kwargs["libraries"] == ["test-lib-1", "test-lib-2"]

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_libraries_are_stored_in_base_session(
        self, mock_session_class: MagicMock, mock_docker_env: MagicMock
    ) -> None:
        """Verify that libraries are stored in BaseSession._initial_libraries."""
        mock_client = MagicMock()
        mock_docker_env.return_value = mock_client

        # Create a real session instance to test
        mock_session_instance = MagicMock(spec=BaseSession)
        mock_session_instance.container = MagicMock()
        mock_session_instance.container.id = "test-container-id"
        mock_session_instance.is_open = False
        mock_session_class.return_value = mock_session_instance

        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=PoolConfig(min_pool_size=1, max_pool_size=1, enable_prewarming=False),
            lang=SupportedLanguage.PYTHON,
            client=mock_client,
            libraries=["numpy", "pandas"],
        )

        # Create session and verify libraries are passed
        session = pool._create_session_for_container()
        call_kwargs = mock_session_class.call_args[1]
        assert call_kwargs["libraries"] == ["numpy", "pandas"]

        # Verify the session instance would have _initial_libraries set
        # (we can't directly check this with mocks, but we verify the kwargs are passed)

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_libraries_installed_during_environment_setup(
        self, mock_session_class: MagicMock, mock_docker_env: MagicMock
    ) -> None:
        """Verify that libraries are passed and will be installed during environment_setup()."""
        mock_client = MagicMock()
        mock_docker_env.return_value = mock_client

        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=PoolConfig(min_pool_size=1, max_pool_size=1, enable_prewarming=False),
            lang=SupportedLanguage.PYTHON,
            client=mock_client,
            libraries=["requests", "numpy"],
        )

        # Create session and verify libraries are passed in kwargs
        pool._create_session_for_container()

        call_kwargs = mock_session_class.call_args[1]
        assert "libraries" in call_kwargs
        assert call_kwargs["libraries"] == ["requests", "numpy"]

        # The actual installation happens during session.open() -> environment_setup()
        # which is tested in integration tests or when containers are actually created

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_no_libraries_when_none_provided(self, mock_session_class: MagicMock, mock_docker_env: MagicMock) -> None:
        """Verify that no libraries are passed when none are provided."""
        mock_client = MagicMock()
        mock_docker_env.return_value = mock_client

        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=PoolConfig(min_pool_size=1, max_pool_size=1, enable_prewarming=False),
            lang=SupportedLanguage.PYTHON,
            client=mock_client,
            # No libraries parameter
        )

        pool._create_session_for_container()

        call_kwargs = mock_session_class.call_args[1]
        # Libraries should not be in kwargs if not provided
        assert "libraries" not in call_kwargs or call_kwargs.get("libraries") is None

    @patch("llm_sandbox.pool.docker_pool.docker.from_env")
    @patch("llm_sandbox.docker.SandboxDockerSession")
    def test_libraries_with_skip_environment_setup(
        self, mock_session_class: MagicMock, mock_docker_env: MagicMock
    ) -> None:
        """Verify that libraries and skip_environment_setup are both passed to session."""
        mock_client = MagicMock()
        mock_docker_env.return_value = mock_client

        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=PoolConfig(min_pool_size=1, max_pool_size=1, enable_prewarming=False),
            lang=SupportedLanguage.PYTHON,
            client=mock_client,
            libraries=["numpy"],
            skip_environment_setup=True,
        )

        pool._create_session_for_container()

        # Verify both parameters are passed
        call_kwargs = mock_session_class.call_args[1]
        assert call_kwargs["skip_environment_setup"] is True
        assert call_kwargs["libraries"] == ["numpy"]
        # Note: The actual warning/logging behavior is tested in BaseSession tests
