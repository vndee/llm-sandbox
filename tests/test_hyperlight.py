# ruff: noqa: SLF001, PLR2004, ARG002, FBT003
"""Tests for Hyperlight backend implementation."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from llm_sandbox.const import SandboxBackend, SupportedLanguage
from llm_sandbox.exceptions import ContainerError, MissingDependencyError
from llm_sandbox.session import create_session


class TestHyperlightBackendIntegration:
    """Test Hyperlight backend integration with session factory."""

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.hyperlight.SandboxHyperlightSession")
    def test_create_hyperlight_session(
        self, mock_hyperlight_session: MagicMock, mock_find_spec: MagicMock
    ) -> None:
        """Test creating Hyperlight session via create_session."""
        mock_find_spec.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_hyperlight_session.return_value = mock_session_instance

        session = create_session(
            backend=SandboxBackend.HYPERLIGHT, lang="python", verbose=True, guest_binary_path="/tmp/guest"
        )

        assert session == mock_session_instance
        mock_hyperlight_session.assert_called_once_with(lang="python", verbose=True, guest_binary_path="/tmp/guest")

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.hyperlight.SandboxHyperlightSession")
    def test_hyperlight_session_with_default_params(
        self, mock_hyperlight_session: MagicMock, mock_find_spec: MagicMock
    ) -> None:
        """Test creating Hyperlight session with minimal parameters."""
        mock_find_spec.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_hyperlight_session.return_value = mock_session_instance

        session = create_session(backend=SandboxBackend.HYPERLIGHT)

        assert session == mock_session_instance
        mock_hyperlight_session.assert_called_once()


class TestHyperlightContainerAPI:
    """Test HyperlightContainerAPI implementation."""

    def test_container_api_creation(self) -> None:
        """Test creating HyperlightContainerAPI."""
        from llm_sandbox.hyperlight import HyperlightContainerAPI

        api = HyperlightContainerAPI(workdir="/sandbox", verbose=True)
        assert api.workdir == "/sandbox"
        assert api.verbose is True

    def test_create_container_with_valid_binary(self) -> None:
        """Test creating container with valid guest binary path."""
        from llm_sandbox.hyperlight import HyperlightContainerAPI

        api = HyperlightContainerAPI()

        with patch("pathlib.Path.exists", return_value=True):
            result = api.create_container({"guest_binary_path": "/tmp/guest"})
            assert result == "/tmp/guest"

    def test_create_container_without_binary_path(self) -> None:
        """Test creating container without guest_binary_path raises error."""
        from llm_sandbox.hyperlight import HyperlightContainerAPI

        api = HyperlightContainerAPI()

        with pytest.raises(ContainerError, match="requires guest_binary_path"):
            api.create_container({})

    def test_create_container_with_nonexistent_binary(self) -> None:
        """Test creating container with non-existent guest binary."""
        from llm_sandbox.hyperlight import HyperlightContainerAPI

        api = HyperlightContainerAPI()

        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(ContainerError, match="Guest binary not found"):
                api.create_container({"guest_binary_path": "/tmp/nonexistent"})

    def test_start_container_is_noop(self) -> None:
        """Test start_container does nothing (no-op)."""
        from llm_sandbox.hyperlight import HyperlightContainerAPI

        api = HyperlightContainerAPI()
        api.start_container("/tmp/guest")  # Should not raise

    def test_stop_container_is_noop(self) -> None:
        """Test stop_container does nothing (no-op)."""
        from llm_sandbox.hyperlight import HyperlightContainerAPI

        api = HyperlightContainerAPI()
        api.stop_container("/tmp/guest")  # Should not raise

    def test_copy_to_container_not_supported(self) -> None:
        """Test copy_to_container raises NotImplementedError."""
        from llm_sandbox.hyperlight import HyperlightContainerAPI

        api = HyperlightContainerAPI()

        with pytest.raises(NotImplementedError, match="not supported"):
            api.copy_to_container("/tmp/guest", "src.txt", "dest.txt")

    def test_copy_from_container_not_supported(self) -> None:
        """Test copy_from_container raises NotImplementedError."""
        from llm_sandbox.hyperlight import HyperlightContainerAPI

        api = HyperlightContainerAPI()

        with pytest.raises(NotImplementedError, match="not supported"):
            api.copy_from_container("/tmp/guest", "file.txt")


class TestHyperlightSession:
    """Test SandboxHyperlightSession implementation."""

    @patch("llm_sandbox.hyperlight.shutil.which")
    @patch("pathlib.Path.exists")
    def test_session_initialization(self, mock_exists: MagicMock, mock_which: MagicMock) -> None:
        """Test basic session initialization."""
        from llm_sandbox.hyperlight import SandboxHyperlightSession

        mock_which.return_value = "/usr/bin/cargo"
        mock_exists.return_value = True

        session = SandboxHyperlightSession(
            lang=SupportedLanguage.PYTHON, verbose=True, guest_binary_path="/tmp/guest"
        )

        assert session.config.lang == SupportedLanguage.PYTHON
        assert session.config.verbose is True
        assert session.guest_binary_path == "/tmp/guest"

    @patch("llm_sandbox.hyperlight.shutil.which")
    def test_check_dependencies_missing_rust(self, mock_which: MagicMock) -> None:
        """Test dependency check fails when Rust is not installed."""
        from llm_sandbox.hyperlight import SandboxHyperlightSession

        mock_which.return_value = None

        session = SandboxHyperlightSession(lang=SupportedLanguage.PYTHON)

        with pytest.raises(MissingDependencyError, match="requires Rust toolchain"):
            session._check_hyperlight_dependencies()

    @patch("llm_sandbox.hyperlight.shutil.which")
    def test_check_dependencies_with_rust(self, mock_which: MagicMock) -> None:
        """Test dependency check succeeds when Rust is installed."""
        from llm_sandbox.hyperlight import SandboxHyperlightSession

        mock_which.return_value = "/usr/bin/cargo"

        session = SandboxHyperlightSession(lang=SupportedLanguage.PYTHON, guest_binary_path="/tmp/guest")
        session._check_hyperlight_dependencies()  # Should not raise

    @patch("llm_sandbox.hyperlight.shutil.which")
    @patch("pathlib.Path.exists")
    @patch("llm_sandbox.hyperlight.SandboxHyperlightSession._compile_guest_binary")
    def test_open_session_with_existing_binary(
        self, mock_compile: MagicMock, mock_exists: MagicMock, mock_which: MagicMock
    ) -> None:
        """Test opening session with pre-compiled guest binary."""
        from llm_sandbox.hyperlight import SandboxHyperlightSession

        mock_which.return_value = "/usr/bin/cargo"
        mock_exists.return_value = True

        session = SandboxHyperlightSession(lang=SupportedLanguage.PYTHON, guest_binary_path="/tmp/guest")
        session.open()

        assert session.is_open is True
        mock_compile.assert_not_called()

    @patch("llm_sandbox.hyperlight.shutil.which")
    @patch("pathlib.Path.exists")
    def test_open_session_with_missing_binary(self, mock_exists: MagicMock, mock_which: MagicMock) -> None:
        """Test opening session with missing guest binary raises error."""
        from llm_sandbox.hyperlight import SandboxHyperlightSession

        mock_which.return_value = "/usr/bin/cargo"
        mock_exists.return_value = False

        session = SandboxHyperlightSession(lang=SupportedLanguage.PYTHON, guest_binary_path="/tmp/nonexistent")

        with pytest.raises(ContainerError, match="Guest binary not found"):
            session.open()

    @patch("llm_sandbox.hyperlight.shutil.which")
    @patch("pathlib.Path.exists")
    def test_close_session(self, mock_exists: MagicMock, mock_which: MagicMock) -> None:
        """Test closing session cleans up resources."""
        from llm_sandbox.hyperlight import SandboxHyperlightSession

        mock_which.return_value = "/usr/bin/cargo"
        mock_exists.return_value = True

        session = SandboxHyperlightSession(lang=SupportedLanguage.PYTHON, guest_binary_path="/tmp/guest")
        session.open()
        session.close()

        assert session.is_open is False

    @patch("llm_sandbox.hyperlight.shutil.which")
    @patch("pathlib.Path.exists")
    def test_context_manager(self, mock_exists: MagicMock, mock_which: MagicMock) -> None:
        """Test using session as context manager."""
        from llm_sandbox.hyperlight import SandboxHyperlightSession

        mock_which.return_value = "/usr/bin/cargo"
        mock_exists.return_value = True

        with SandboxHyperlightSession(lang=SupportedLanguage.PYTHON, guest_binary_path="/tmp/guest") as session:
            assert session.is_open is True

        assert session.is_open is False

    def test_install_not_supported(self) -> None:
        """Test that library installation returns error message."""
        from llm_sandbox.hyperlight import SandboxHyperlightSession

        session = SandboxHyperlightSession(lang=SupportedLanguage.PYTHON)

        result = session.install(["numpy", "pandas"])

        assert result.exit_code == 1
        assert "not supported" in result.output

    @patch("llm_sandbox.hyperlight.shutil.which")
    @patch("pathlib.Path.exists")
    def test_environment_setup_is_noop(self, mock_exists: MagicMock, mock_which: MagicMock) -> None:
        """Test environment setup does nothing for Hyperlight."""
        from llm_sandbox.hyperlight import SandboxHyperlightSession

        mock_which.return_value = "/usr/bin/cargo"
        mock_exists.return_value = True

        session = SandboxHyperlightSession(lang=SupportedLanguage.PYTHON, guest_binary_path="/tmp/guest")
        session.open()
        session.environment_setup()  # Should not raise


class TestHyperlightBackendConsistency:
    """Test that Hyperlight backend follows common patterns."""

    @patch("llm_sandbox.session.find_spec")
    def test_hyperlight_implements_required_methods(self, mock_find_spec: MagicMock) -> None:
        """Test that Hyperlight backend implements required methods."""
        mock_find_spec.return_value = MagicMock()

        required_methods = [
            "open",
            "close",
            "run",
            "execute_command",
            "execute_commands",
            "install",
            "environment_setup",
            "__enter__",
            "__exit__",
        ]

        with patch("llm_sandbox.hyperlight.SandboxHyperlightSession") as mock_session_class:
            mock_session_instance = MagicMock()
            mock_session_class.return_value = mock_session_instance

            session = create_session(backend=SandboxBackend.HYPERLIGHT)

            for method_name in required_methods:
                assert hasattr(session, method_name), f"Hyperlight should have {method_name} method"

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.hyperlight.SandboxHyperlightSession")
    def test_hyperlight_accepts_common_parameters(
        self, mock_hyperlight_session: MagicMock, mock_find_spec: MagicMock
    ) -> None:
        """Test that Hyperlight backend accepts common initialization parameters."""
        from llm_sandbox.security import SecurityPolicy

        mock_find_spec.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_hyperlight_session.return_value = mock_session_instance

        common_params = {
            "lang": "python",
            "verbose": True,
            "workdir": "/test",
            "security_policy": SecurityPolicy(patterns=[], restricted_modules=[]),
        }

        _ = create_session(backend=SandboxBackend.HYPERLIGHT, **common_params)

        mock_hyperlight_session.assert_called_once()
        call_kwargs = mock_hyperlight_session.call_args[1]

        assert call_kwargs["lang"] == "python"
        assert call_kwargs["verbose"] is True
        assert call_kwargs["workdir"] == "/test"


class TestHyperlightSpecificFeatures:
    """Test Hyperlight-specific features and parameters."""

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.hyperlight.SandboxHyperlightSession")
    def test_hyperlight_guest_binary_path(
        self, mock_hyperlight_session: MagicMock, mock_find_spec: MagicMock
    ) -> None:
        """Test Hyperlight-specific guest_binary_path parameter."""
        mock_find_spec.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_hyperlight_session.return_value = mock_session_instance

        _ = create_session(backend=SandboxBackend.HYPERLIGHT, guest_binary_path="/tmp/custom_guest", lang="python")

        mock_hyperlight_session.assert_called_once()
        call_kwargs = mock_hyperlight_session.call_args[1]
        assert call_kwargs["guest_binary_path"] == "/tmp/custom_guest"

    @patch("llm_sandbox.session.find_spec")
    @patch("llm_sandbox.hyperlight.SandboxHyperlightSession")
    def test_hyperlight_keep_template(self, mock_hyperlight_session: MagicMock, mock_find_spec: MagicMock) -> None:
        """Test keep_template parameter for Hyperlight."""
        mock_find_spec.return_value = MagicMock()
        mock_session_instance = MagicMock()
        mock_hyperlight_session.return_value = mock_session_instance

        _ = create_session(backend=SandboxBackend.HYPERLIGHT, keep_template=True, lang="python")

        mock_hyperlight_session.assert_called_once()
        call_kwargs = mock_hyperlight_session.call_args[1]
        assert call_kwargs["keep_template"] is True


class TestHyperlightGuestTemplate:
    """Test Hyperlight guest template generation."""

    def test_guest_template_exists(self) -> None:
        """Test that HYPERLIGHT_GUEST_TEMPLATE is defined."""
        from llm_sandbox.hyperlight import HYPERLIGHT_GUEST_TEMPLATE

        assert HYPERLIGHT_GUEST_TEMPLATE is not None
        assert len(HYPERLIGHT_GUEST_TEMPLATE) > 0
        assert "#![no_std]" in HYPERLIGHT_GUEST_TEMPLATE
        assert "hyperlight_guest" in HYPERLIGHT_GUEST_TEMPLATE


class TestHyperlightDocumentation:
    """Test that Hyperlight backend has proper documentation."""

    def test_module_has_docstring(self) -> None:
        """Test that hyperlight module has comprehensive docstring."""
        import llm_sandbox.hyperlight

        assert llm_sandbox.hyperlight.__doc__ is not None
        assert len(llm_sandbox.hyperlight.__doc__) > 100

    def test_session_class_has_docstring(self) -> None:
        """Test that SandboxHyperlightSession has docstring."""
        from llm_sandbox.hyperlight import SandboxHyperlightSession

        assert SandboxHyperlightSession.__doc__ is not None
        assert "Hyperlight" in SandboxHyperlightSession.__doc__

    def test_api_class_has_docstring(self) -> None:
        """Test that HyperlightContainerAPI has docstring."""
        from llm_sandbox.hyperlight import HyperlightContainerAPI

        assert HyperlightContainerAPI.__doc__ is not None
        assert "Hyperlight" in HyperlightContainerAPI.__doc__
