import pytest
from unittest.mock import MagicMock, patch
from llm_sandbox.pool import create_pool_manager, PoolConfig
from llm_sandbox.const import SupportedLanguage, SandboxBackend
from llm_sandbox.docker import SandboxDockerSession

def test_pool_manager_passes_libraries_to_session():
    """Verify that libraries passed to create_pool_manager are installed in the session."""

    # Mock Docker client to avoid actual container creation
    mock_client = MagicMock()

    # Mock SandboxDockerSession to intercept initialization and open
    # Patch where the class is defined, not where it's imported inside a function
    with patch("llm_sandbox.docker.SandboxDockerSession") as MockSession:
        # Setup the mock session instance
        mock_session_instance = MockSession.return_value
        mock_session_instance.open = MagicMock()

        # Create pool manager with libraries
        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=PoolConfig(min_pool_size=1, max_pool_size=1, enable_prewarming=False),
            lang=SupportedLanguage.PYTHON,
            client=mock_client,
            libraries=["test-lib-1", "test-lib-2"]
        )

        # Trigger session creation (which happens during acquire if pool is empty/not prewarmed)
        # But wait, create_pool_manager with prewarming=False won't create sessions yet.
        # We need to call acquire() or manually trigger creation.

        # Actually, let's look at how DockerPoolManager creates sessions.
        # It calls _create_session_for_container.

        # Manually call _create_session_for_container to see if it passes the libraries
        pool._create_session_for_container()

        # Verify SandboxDockerSession was initialized with libraries
        call_args = MockSession.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert "libraries" in kwargs
        assert kwargs["libraries"] == ["test-lib-1", "test-lib-2"]
