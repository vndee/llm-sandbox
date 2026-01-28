"""Tests for encoding error handling in Docker sessions."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_sandbox.docker import SandboxDockerSession
from llm_sandbox.session import SandboxSession


@pytest.fixture
def mock_docker_client() -> MagicMock:
    """Create a mock Docker client."""
    return MagicMock()


class TestDefaultEncodingBehavior:
    """Test default encoding behavior without explicit encoding_errors parameter."""

    @patch("llm_sandbox.docker.docker.from_env")
    def test_sandbox_session_default_raises_on_invalid_utf8(self, mock_from_env: MagicMock) -> None:
        """SandboxSession without encoding_errors raises UnicodeDecodeError on invalid UTF-8.

        This documents the original/default behavior: non-UTF-8 output causes an error.
        """
        mock_client = MagicMock()
        mock_from_env.return_value = mock_client

        session = SandboxSession(lang="python")
        assert session.config.encoding_errors == "strict"

        # Simulate container output with invalid UTF-8
        invalid_utf8 = (b"output\xff\xfebytes", None)
        with pytest.raises(UnicodeDecodeError):
            session._process_non_stream_output(invalid_utf8)


class TestEncodingErrors:
    """Tests for encoding_errors parameter in Docker output processing."""

    def test_strict_raises_on_invalid_utf8(self, mock_docker_client: MagicMock) -> None:
        """Default strict mode raises UnicodeDecodeError on invalid UTF-8."""
        session = SandboxDockerSession(client=mock_docker_client, encoding_errors="strict")
        invalid_utf8 = (b"\xff\xfe", None)
        with pytest.raises(UnicodeDecodeError):
            session._process_non_stream_output(invalid_utf8)

    def test_replace_substitutes_invalid_bytes(self, mock_docker_client: MagicMock) -> None:
        """Replace mode uses replacement character for invalid bytes."""
        session = SandboxDockerSession(client=mock_docker_client, encoding_errors="replace")
        stdout, _ = session._process_non_stream_output((b"hello\xff\xfeworld", None))
        assert "hello" in stdout
        assert "world" in stdout
        assert "\ufffd" in stdout  # Replacement character

    def test_surrogateescape_roundtrips(self, mock_docker_client: MagicMock) -> None:
        """Surrogateescape mode allows round-tripping binary data."""
        session = SandboxDockerSession(client=mock_docker_client, encoding_errors="surrogateescape")
        original = b"hello\xff\xfeworld"
        stdout, _ = session._process_non_stream_output((original, None))
        assert stdout.encode("utf-8", errors="surrogateescape") == original

    def test_ignore_skips_invalid_bytes(self, mock_docker_client: MagicMock) -> None:
        """Ignore mode skips invalid bytes."""
        session = SandboxDockerSession(client=mock_docker_client, encoding_errors="ignore")
        stdout, _ = session._process_non_stream_output((b"hello\xff\xfeworld", None))
        assert stdout == "helloworld"

    def test_backslashreplace_escapes_invalid_bytes(self, mock_docker_client: MagicMock) -> None:
        """Backslashreplace mode escapes invalid bytes with backslash sequences."""
        session = SandboxDockerSession(client=mock_docker_client, encoding_errors="backslashreplace")
        stdout, _ = session._process_non_stream_output((b"hello\xffworld", None))
        assert "hello" in stdout
        assert "world" in stdout
        assert "\\xff" in stdout

    def test_default_encoding_is_strict(self, mock_docker_client: MagicMock) -> None:
        """Default encoding_errors is 'strict'."""
        session = SandboxDockerSession(client=mock_docker_client)
        assert session.config.encoding_errors == "strict"

    def test_stderr_also_uses_encoding_errors(self, mock_docker_client: MagicMock) -> None:
        """Stderr is also decoded using encoding_errors setting."""
        session = SandboxDockerSession(client=mock_docker_client, encoding_errors="replace")
        _, stderr = session._process_non_stream_output((None, b"error\xff\xfemsg"))
        assert "error" in stderr
        assert "msg" in stderr
        assert "\ufffd" in stderr

    def test_stream_output_respects_encoding_errors(self, mock_docker_client: MagicMock) -> None:
        """Stream processing also respects encoding_errors setting."""
        session = SandboxDockerSession(client=mock_docker_client, encoding_errors="replace", stream=True)

        def mock_stream() -> Generator[tuple[bytes | None, bytes | None], Any, None]:
            yield (b"hello\xff", None)
            yield (b"\xfeworld", None)

        stdout, _ = session._process_stream_output(mock_stream())
        assert "hello" in stdout
        assert "world" in stdout
        assert "\ufffd" in stdout

    def test_stream_strict_swallows_unicode_error(self, mock_docker_client: MagicMock) -> None:
        """Stream processing swallows UnicodeDecodeError for resilience."""
        session = SandboxDockerSession(client=mock_docker_client, encoding_errors="strict", stream=True)

        def mock_stream() -> Generator[tuple[bytes | None, bytes | None], Any, None]:
            yield (b"hello\xff\xfeworld", None)

        # Stream mode swallows exceptions (except SandboxTimeoutError) for resilience
        stdout, stderr = session._process_stream_output(mock_stream())
        # Returns partial output up to the error point
        assert stdout == ""
        assert stderr == ""

    def test_valid_utf8_works_with_all_modes(self, mock_docker_client: MagicMock) -> None:
        """Valid UTF-8 works correctly with all encoding modes."""
        valid_utf8 = "Hello, \u4e16\u754c!".encode()  # "Hello, World!" with Chinese characters
        modes = ["strict", "replace", "ignore", "surrogateescape", "backslashreplace"]

        for mode in modes:
            session = SandboxDockerSession(client=mock_docker_client, encoding_errors=mode)  # type: ignore[arg-type]
            stdout, _ = session._process_non_stream_output((valid_utf8, None))
            assert stdout == "Hello, \u4e16\u754c!", f"Failed for mode: {mode}"
