"""Tests for Kubernetes utility functions."""

from unittest.mock import MagicMock, patch

import pytest
from kubernetes.client.exceptions import ApiException

from llm_sandbox.k8s_utils import retry_k8s_api_call


class TestRetryK8sApiCall:
    """Test retry_k8s_api_call function."""

    def test_successful_call(self):
        """Test successful API call without retries."""
        mock_func = MagicMock(return_value="success")

        result = retry_k8s_api_call(mock_func, "arg1", kwarg1="value")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value")

    def test_retry_on_websocket_error(self):
        """Test retry on WebSocket handshake error."""
        # Mock ApiException for WebSocket error
        ws_error = ApiException(status=0, reason="Handshake status 404 Not Found")

        # Function raises error twice, then succeeds
        mock_func = MagicMock(side_effect=[ws_error, ws_error, "success"])

        # Mock time.sleep to speed up test
        with patch("time.sleep") as mock_sleep:
            result = retry_k8s_api_call(mock_func, max_retries=3)

        assert result == "success"
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2

    def test_fail_after_max_retries(self):
        """Test failure after exhausting retries."""
        ws_error = ApiException(status=0, reason="Handshake status 404 Not Found")
        mock_func = MagicMock(side_effect=ws_error)

        with patch("time.sleep"), pytest.raises(ApiException) as exc:
            retry_k8s_api_call(mock_func, max_retries=3)

        assert exc.value == ws_error
        assert mock_func.call_count == 3

    def test_no_retry_on_other_api_exception(self):
        """Test no retry on non-WebSocket ApiException."""
        # 404 Not Found (not handshake related)
        api_error = ApiException(status=404, reason="Not Found")
        mock_func = MagicMock(side_effect=api_error)

        with patch("time.sleep") as mock_sleep, pytest.raises(ApiException) as exc:
            retry_k8s_api_call(mock_func)

        assert exc.value == api_error
        assert mock_func.call_count == 1
        mock_sleep.assert_not_called()

    def test_no_retry_on_generic_exception(self):
        """Test no retry on generic Exception."""
        error = ValueError("Generic error")
        mock_func = MagicMock(side_effect=error)

        with patch("time.sleep") as mock_sleep, pytest.raises(ValueError, match="Generic error") as exc:
            retry_k8s_api_call(mock_func)

        assert exc.value == error
        assert mock_func.call_count == 1
        mock_sleep.assert_not_called()

    def test_custom_logger(self):
        """Test using a custom logger."""
        ws_error = ApiException(status=0, reason="Handshake status")
        mock_func = MagicMock(side_effect=[ws_error, "success"])
        mock_logger = MagicMock()

        with patch("time.sleep"):
            retry_k8s_api_call(mock_func, logger=mock_logger)

        assert mock_logger.warning.called

    def test_max_retries_zero(self):
        """Test max_retries=0 raises RuntimeError."""
        mock_func = MagicMock()

        with pytest.raises(RuntimeError, match="Unexpected state"):
            retry_k8s_api_call(mock_func, max_retries=0)

    def test_thread_safety(self):
        """Test thread safety using the global lock."""
        # This is a bit hard to deterministic test, but we can verify the lock is used
        mock_func = MagicMock(return_value="success")

        with patch("llm_sandbox.k8s_utils._k8s_api_lock") as mock_lock:
            retry_k8s_api_call(mock_func)

        mock_lock.__enter__.assert_called()
        mock_lock.__exit__.assert_called()
