"""Shared test fixtures and configuration for LLM Sandbox tests."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_docker_backend() -> Generator[dict[str, Any], None, None]:
    """Mock Docker backend to prevent real Docker connections in security tests.

    This fixture should only be used in security-related tests that don't need
    real language handlers or Docker connections.
    """
    with (
        patch("llm_sandbox.session.find_spec") as mock_find_spec,
        patch("llm_sandbox.docker.docker.from_env") as mock_docker_from_env,
        patch("llm_sandbox.language_handlers.factory.LanguageHandlerFactory.create_handler") as mock_create_handler,
    ):
        # Setup mocks
        mock_find_spec.return_value = MagicMock()

        # Create a proper mock for the language handler with realistic patterns
        mock_handler = MagicMock()

        def get_import_patterns(module: str) -> str:
            """Return realistic import patterns that handle aliases and submodules."""
            import re

            return (
                r"\s*(from\s+"
                + re.escape(module)
                + r"(?:\s|$|\.|import)|import\s+"
                + re.escape(module)
                + r"(?:\s|$|\.))"
            )

        def filter_comments(code: str) -> str:
            """Filter out Python comments from code.

            For security scanning, we want to be conservative and still scan
            string literals since they could contain malicious code that gets
            executed via exec() or eval().
            """
            import re

            # Only remove actual comment lines (lines starting with #)
            # Don't remove string literals as they could contain executable code
            lines = code.split("\n")
            filtered_lines = []

            for line in lines:
                # Remove inline comments, but be careful about # inside strings
                # This is a simplified approach - only remove # that appear to be comments
                stripped = line.strip()
                if stripped.startswith("#"):
                    # Skip comment-only lines
                    continue
                # For lines with potential inline comments, use a simple regex
                # that tries to avoid # inside string literals
                line_no_comment = re.sub(r'\s+#[^"\']*$', "", line)
                filtered_lines.append(line_no_comment)

            return "\n".join(filtered_lines)

        mock_handler.get_import_patterns.side_effect = get_import_patterns
        mock_handler.filter_comments.side_effect = filter_comments
        mock_handler.name = "python"
        mock_handler.file_extension = "py"
        mock_create_handler.return_value = mock_handler

        mock_client = MagicMock()
        mock_docker_from_env.return_value = mock_client

        yield {
            "find_spec": mock_find_spec,
            "docker_client": mock_docker_from_env,
            "language_handler": mock_create_handler,
        }
