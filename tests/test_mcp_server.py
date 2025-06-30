"""Tests for MCP server module."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from mcp.types import ImageContent, TextContent

from llm_sandbox import SupportedLanguage
from llm_sandbox.const import SandboxBackend
from llm_sandbox.data import ExecutionResult, FileType, PlotOutput
from llm_sandbox.exceptions import MissingDependencyError
from llm_sandbox.mcp_server.const import LANGUAGE_RESOURCES
from llm_sandbox.mcp_server.server import (
    _get_backend,
    _supports_visualization,
    execute_code,
    get_language_details,
    get_supported_languages,
    language_details,
)


class TestGetBackend:
    """Test _get_backend function."""

    @patch("llm_sandbox.mcp_server.server._check_dependency")
    @patch.dict(os.environ, {"BACKEND": "docker"})
    def test_get_backend_docker_from_env(self, mock_check_dependency: MagicMock) -> None:
        """Test getting Docker backend from environment variable."""
        result = _get_backend()

        assert result == SandboxBackend.DOCKER
        mock_check_dependency.assert_called_once_with(SandboxBackend.DOCKER)

    @patch("llm_sandbox.mcp_server.server._check_dependency")
    @patch.dict(os.environ, {"BACKEND": "kubernetes"})
    def test_get_backend_kubernetes_from_env(self, mock_check_dependency: MagicMock) -> None:
        """Test getting Kubernetes backend from environment variable."""
        result = _get_backend()

        assert result == SandboxBackend.KUBERNETES
        mock_check_dependency.assert_called_once_with(SandboxBackend.KUBERNETES)

    @patch("llm_sandbox.mcp_server.server._check_dependency")
    @patch.dict(os.environ, {}, clear=True)
    def test_get_backend_default_docker(self, mock_check_dependency: MagicMock) -> None:
        """Test getting default Docker backend when no environment variable is set."""
        result = _get_backend()

        assert result == SandboxBackend.DOCKER
        mock_check_dependency.assert_called_once_with(SandboxBackend.DOCKER)

    @patch("llm_sandbox.mcp_server.server._check_dependency")
    @patch.dict(os.environ, {"BACKEND": "podman"})
    def test_get_backend_podman_from_env(self, mock_check_dependency: MagicMock) -> None:
        """Test getting Podman backend from environment variable."""
        result = _get_backend()

        assert result == SandboxBackend.PODMAN
        mock_check_dependency.assert_called_once_with(SandboxBackend.PODMAN)


class TestSupportsVisualization:
    """Test _supports_visualization function."""

    def test_supports_visualization_python(self) -> None:
        """Test that Python supports visualization."""
        result = _supports_visualization("python")
        assert result is True

    def test_supports_visualization_r(self) -> None:
        """Test that R supports visualization."""
        result = _supports_visualization("r")
        assert result is True

    def test_supports_visualization_javascript(self) -> None:
        """Test that JavaScript does not support visualization."""
        result = _supports_visualization("javascript")
        assert result is False

    def test_supports_visualization_java(self) -> None:
        """Test that Java does not support visualization."""
        result = _supports_visualization("java")
        assert result is False

    def test_supports_visualization_unknown_language(self) -> None:
        """Test that unknown language does not support visualization."""
        result = _supports_visualization("unknown")
        assert result is False

    def test_supports_visualization_none_language(self) -> None:
        """Test that None language does not support visualization."""
        result = _supports_visualization("")
        assert result is False


class TestExecuteCode:
    """Test execute_code tool function."""

    @patch("llm_sandbox.mcp_server.server._get_backend")
    @patch("llm_sandbox.mcp_server.server.ArtifactSandboxSession")
    def test_execute_code_basic_success(self, mock_session_cls: MagicMock, mock_get_backend: MagicMock) -> None:
        """Test basic successful code execution."""
        # Setup
        mock_backend = SandboxBackend.DOCKER
        mock_get_backend.return_value = mock_backend

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=None)

        mock_result = ExecutionResult(exit_code=0, stdout="Hello, World!", stderr="")
        mock_session.run.return_value = mock_result

        # Execute
        result = execute_code("print('Hello, World!')", "python")

        # Verify
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert result[0].type == "text"
        assert "Hello, World!" in result[0].text

        mock_session_cls.assert_called_once_with(
            lang="python",
            keep_template=True,
            verbose=False,
            backend=mock_backend,
            session_timeout=30,
        )
        mock_session.run.assert_called_once_with(code="print('Hello, World!')", libraries=[], timeout=30)

    @patch("llm_sandbox.mcp_server.server._get_backend")
    @patch("llm_sandbox.mcp_server.server.ArtifactSandboxSession")
    def test_execute_code_with_visualization(self, mock_session_cls: MagicMock, mock_get_backend: MagicMock) -> None:
        """Test code execution with visualization support."""
        # Setup
        mock_backend = SandboxBackend.DOCKER
        mock_get_backend.return_value = mock_backend

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=None)

        mock_plot = PlotOutput(
            content_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            format=FileType.PNG,
        )
        mock_result = ExecutionResult(exit_code=0, stdout="Plot created", stderr="", plots=[mock_plot])
        mock_session.run.return_value = mock_result

        # Execute
        result = execute_code("import matplotlib.pyplot as plt; plt.plot([1,2,3]); plt.show()", "python")

        # Verify
        assert len(result) == 2  # noqa: PLR2004

        # First result should be the image
        assert isinstance(result[0], ImageContent)
        assert result[0].type == "image"
        assert result[0].mimeType == "image/png"
        assert result[0].data == mock_plot.content_base64

        # Second result should be the execution result
        assert isinstance(result[1], TextContent)
        assert result[1].type == "text"
        assert "Plot created" in result[1].text

        mock_session_cls.assert_called_once_with(
            lang="python",
            keep_template=True,
            verbose=False,
            backend=mock_backend,
            session_timeout=30,
        )

    @patch("llm_sandbox.mcp_server.server._get_backend")
    @patch("llm_sandbox.mcp_server.server.ArtifactSandboxSession")
    def test_execute_code_with_libraries(self, mock_session_cls: MagicMock, mock_get_backend: MagicMock) -> None:
        """Test code execution with custom libraries."""
        # Setup
        mock_backend = SandboxBackend.DOCKER
        mock_get_backend.return_value = mock_backend

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=None)

        mock_result = ExecutionResult(exit_code=0, stdout="Success", stderr="")
        mock_session.run.return_value = mock_result

        # Execute
        _ = execute_code(
            "import requests; print(requests.__version__)",
            "python",
            libraries=["requests", "pandas"],
            timeout=60,
        )

        # Verify
        mock_session.run.assert_called_once_with(
            code="import requests; print(requests.__version__)",
            libraries=["requests", "pandas"],
            timeout=60,
        )
        mock_session_cls.assert_called_once_with(
            lang="python",
            keep_template=True,
            verbose=False,
            backend=mock_backend,
            session_timeout=60,
        )

    @patch("llm_sandbox.mcp_server.server._get_backend")
    @patch("llm_sandbox.mcp_server.server.SandboxSession")
    def test_execute_code_javascript(self, mock_session_cls: MagicMock, mock_get_backend: MagicMock) -> None:
        """Test JavaScript code execution (no visualization support)."""
        # Setup
        mock_backend = SandboxBackend.DOCKER
        mock_get_backend.return_value = mock_backend

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=None)

        mock_result = ExecutionResult(exit_code=0, stdout="Hello from JS", stderr="")
        mock_session.run.return_value = mock_result

        # Execute
        result = execute_code("console.log('Hello from JS')", "javascript")

        # Verify
        assert len(result) == 1
        assert isinstance(result[0], TextContent)

        # Should use SandboxSession (not ArtifactSandboxSession) for JS
        mock_session_cls.assert_called_once_with(
            lang="javascript",
            keep_template=True,
            verbose=False,
            backend=mock_backend,
            session_timeout=30,
        )

    @patch("llm_sandbox.mcp_server.server._get_backend")
    @patch("llm_sandbox.mcp_server.server.ArtifactSandboxSession")
    def test_execute_code_error_handling(self, mock_session_cls: MagicMock, mock_get_backend: MagicMock) -> None:
        """Test error handling during code execution."""
        # Setup
        mock_backend = SandboxBackend.DOCKER
        mock_get_backend.return_value = mock_backend

        # Simulate an exception during session creation
        mock_session_cls.side_effect = RuntimeError("Docker not available")

        # Execute
        result = execute_code("print('Hello')", "python")

        # Verify
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        result_data = json.loads(result[0].text)
        assert result_data["exit_code"] == 1
        assert "Docker not available" in result_data["stderr"]

    @patch("llm_sandbox.mcp_server.server._get_backend")
    @patch("llm_sandbox.mcp_server.server.ArtifactSandboxSession")
    def test_execute_code_session_error(self, mock_session_cls: MagicMock, mock_get_backend: MagicMock) -> None:
        """Test error handling when session.run fails."""
        # Setup
        mock_backend = SandboxBackend.DOCKER
        mock_get_backend.return_value = mock_backend

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=None)
        mock_session.run.side_effect = RuntimeError("Execution failed")

        # Execute
        result = execute_code("invalid code", "python")

        # Verify
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        result_data = json.loads(result[0].text)
        assert result_data["exit_code"] == 1
        assert "Execution failed" in result_data["stderr"]


class TestGetSupportedLanguages:
    """Test get_supported_languages tool function."""

    def test_get_supported_languages(self) -> None:
        """Test getting supported languages list."""
        result = get_supported_languages()

        assert isinstance(result, TextContent)
        assert result.type == "text"

        languages = json.loads(result.text)
        assert isinstance(languages, list)

        # Check that all supported languages are included
        expected_languages = [lang.value for lang in SupportedLanguage]
        assert set(languages) == set(expected_languages)

        # Verify specific languages are present
        assert "python" in languages
        assert "javascript" in languages
        assert "java" in languages
        assert "cpp" in languages
        assert "go" in languages
        assert "r" in languages
        assert "ruby" in languages


class TestGetLanguageDetails:
    """Test get_language_details tool function."""

    def test_get_language_details_python(self) -> None:
        """Test getting details for Python language."""
        result = get_language_details("python")

        assert isinstance(result, TextContent)
        assert result.type == "text"

        details = json.loads(result.text)
        assert isinstance(details, dict)

        # Check required fields
        assert "version" in details
        assert "package_manager" in details
        assert "preinstalled_libraries" in details
        assert "use_cases" in details
        assert "visualization_support" in details
        assert "examples" in details

        # Check Python-specific values
        assert details["version"] == "3.11"
        assert details["package_manager"] == "pip"
        assert details["visualization_support"] is True
        assert "numpy" in details["preinstalled_libraries"]
        assert "pandas" in details["preinstalled_libraries"]

    def test_get_language_details_javascript(self) -> None:
        """Test getting details for JavaScript language."""
        result = get_language_details("javascript")

        assert isinstance(result, TextContent)
        details = json.loads(result.text)

        assert details["version"] == "Node.js 22"
        assert details["package_manager"] == "npm"
        assert details["visualization_support"] is False

    def test_get_language_details_invalid_language(self) -> None:
        """Test getting details for invalid language."""
        result = get_language_details("invalid_lang")

        assert isinstance(result, TextContent)
        assert result.type == "text"

        details = json.loads(result.text)
        assert "error" in details
        assert "Unsupported language: invalid_lang" in details["error"]

    def test_get_language_details_all_supported_languages(self) -> None:
        """Test getting details for all supported languages."""
        for lang in SupportedLanguage:
            result = get_language_details(lang.value)

            assert isinstance(result, TextContent)
            details = json.loads(result.text)

            # Skip if details has an error (language not in LANGUAGE_RESOURCES)
            if "error" in details:
                continue

            # Should have all required fields
            required_fields = [
                "version",
                "package_manager",
                "preinstalled_libraries",
                "use_cases",
                "visualization_support",
                "examples",
            ]
            for field in required_fields:
                assert field in details


class TestLanguageDetailsResource:
    """Test language_details resource function."""

    def test_language_details_resource(self) -> None:
        """Test language details resource returns correct format."""
        result = language_details()

        assert isinstance(result, str)

        # Parse JSON
        details = json.loads(result)
        assert isinstance(details, dict)

        # Check that all supported languages have entries
        for lang in SupportedLanguage:
            assert lang.value in details

            lang_info = details[lang.value]
            assert isinstance(lang_info, dict)

            # Check required fields
            required_fields = [
                "version",
                "package_manager",
                "preinstalled_libraries",
                "use_cases",
                "visualization_support",
                "examples",
            ]
            for field in required_fields:
                assert field in lang_info

    def test_language_details_matches_const(self) -> None:
        """Test that language_details resource matches LANGUAGE_RESOURCES constant."""
        result = language_details()
        parsed_result = json.loads(result)

        # Should match the constant exactly
        assert parsed_result == LANGUAGE_RESOURCES


class TestIntegration:
    """Integration tests for MCP server functions."""

    @patch("llm_sandbox.mcp_server.server._get_backend")
    @patch("llm_sandbox.mcp_server.server._check_dependency")
    def test_backend_dependency_check_integration(
        self, mock_check_dependency: MagicMock, mock_get_backend: MagicMock
    ) -> None:
        """Test that backend dependency checking works correctly."""
        mock_check_dependency.side_effect = MissingDependencyError("Docker not found")
        mock_get_backend.side_effect = MissingDependencyError("Docker not found")

        # This should propagate the dependency error
        with pytest.raises(MissingDependencyError):
            _get_backend()

    def test_visualization_language_consistency(self) -> None:
        """Test that visualization support flags are consistent across functions."""
        for lang in SupportedLanguage:
            supports_viz = _supports_visualization(lang.value)

            # Get language details
            result = get_language_details(lang.value)
            details = json.loads(result.text)

            # Skip if details has an error (language not in LANGUAGE_RESOURCES)
            if "error" in details:
                continue

            # Should match the _supports_visualization result
            assert details["visualization_support"] == supports_viz

    def test_all_languages_have_examples(self) -> None:
        """Test that all supported languages have examples in their details."""
        for lang in SupportedLanguage:
            result = get_language_details(lang.value)
            details = json.loads(result.text)

            # Skip if details has an error (language not in LANGUAGE_RESOURCES)
            if "error" in details:
                continue

            assert "examples" in details
            assert isinstance(details["examples"], list)
            assert len(details["examples"]) > 0

            # Check example structure
            for example in details["examples"]:
                assert "title" in example
                assert "description" in example
                assert "code" in example
                assert isinstance(example["title"], str)
                assert isinstance(example["description"], str)
                assert isinstance(example["code"], str)
                assert len(example["title"]) > 0
                assert len(example["code"]) > 0


class TestErrorScenarios:
    """Test various error scenarios and edge cases."""

    @patch("llm_sandbox.mcp_server.server._get_backend")
    def test_execute_code_with_none_libraries(self, mock_get_backend: MagicMock) -> None:
        """Test execute_code with None libraries parameter."""
        mock_backend = SandboxBackend.DOCKER
        mock_get_backend.return_value = mock_backend

        with patch("llm_sandbox.mcp_server.server.ArtifactSandboxSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=None)
            mock_result = ExecutionResult(exit_code=0, stdout="OK", stderr="")
            mock_session.run.return_value = mock_result

            _ = execute_code("print('test')", "python", libraries=None)

            # Should convert None to empty list
            mock_session.run.assert_called_once_with(
                code="print('test')",
                libraries=[],
                timeout=30,
            )

    def test_supports_visualization_edge_cases(self) -> None:
        """Test _supports_visualization with edge case inputs."""
        # Test empty string
        assert _supports_visualization("") is False

        # Test language that exists but has no visualization_support key
        # (This shouldn't happen with current implementation, but test defensive code)
        with patch.dict(
            "llm_sandbox.mcp_server.server.LANGUAGE_RESOURCES", {"test_lang": {"version": "1.0"}}, clear=False
        ):
            assert _supports_visualization("test_lang") is False

    @patch("llm_sandbox.mcp_server.server._get_backend")
    @patch("llm_sandbox.mcp_server.server.ArtifactSandboxSession")
    def test_execute_code_no_plots_attribute(self, mock_session_cls: MagicMock, mock_get_backend: MagicMock) -> None:
        """Test execute_code when result has no plots attribute."""
        mock_backend = SandboxBackend.DOCKER
        mock_get_backend.return_value = mock_backend

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=None)

        # Create result without plots attribute
        mock_result = ExecutionResult(exit_code=0, stdout="No plots", stderr="")
        # Don't set plots attribute
        mock_session.run.return_value = mock_result

        result = execute_code("print('no plots')", "python")

        # Should only return text content (no image)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)

    @patch("llm_sandbox.mcp_server.server._get_backend")
    @patch("llm_sandbox.mcp_server.server.ArtifactSandboxSession")
    def test_execute_code_empty_plots(self, mock_session_cls: MagicMock, mock_get_backend: MagicMock) -> None:
        """Test execute_code when result has empty plots list."""
        mock_backend = SandboxBackend.DOCKER
        mock_get_backend.return_value = mock_backend

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_session_cls.return_value.__exit__ = MagicMock(return_value=None)

        mock_result = ExecutionResult(exit_code=0, stdout="Empty plots", stderr="", plots=[])  # Empty plots list
        mock_session.run.return_value = mock_result

        result = execute_code("print('empty plots')", "python")

        # Should only return text content (no image)
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
