"""Tests for language handlers."""

import logging

import pytest

from llm_sandbox.exceptions import LanguageNotSupportedError
from llm_sandbox.language_handlers.factory import LanguageHandlerFactory
from llm_sandbox.language_handlers.python_handler import PythonHandler


class TestLanguageHandlerFactory:
    """Test LanguageHandlerFactory."""

    def test_create_python_handler(self) -> None:
        """Test creating a Python handler."""
        handler = LanguageHandlerFactory.create_handler("python")
        assert isinstance(handler, PythonHandler)

    def test_create_java_handler(self) -> None:
        """Test creating a Java handler."""
        handler = LanguageHandlerFactory.create_handler("java")
        assert handler.__class__.__name__ == "JavaHandler"

    def test_create_javascript_handler(self) -> None:
        """Test creating a JavaScript handler."""
        handler = LanguageHandlerFactory.create_handler("javascript")
        assert handler.__class__.__name__ == "JavaScriptHandler"

    def test_create_cpp_handler(self) -> None:
        """Test creating a C++ handler."""
        handler = LanguageHandlerFactory.create_handler("cpp")
        assert handler.__class__.__name__ == "CppHandler"

    def test_create_go_handler(self) -> None:
        """Test creating a Go handler."""
        handler = LanguageHandlerFactory.create_handler("go")
        assert handler.__class__.__name__ == "GoHandler"

    def test_create_ruby_handler(self) -> None:
        """Test creating a Ruby handler."""
        handler = LanguageHandlerFactory.create_handler("ruby")
        assert handler.__class__.__name__ == "RubyHandler"

    def test_create_r_handler(self) -> None:
        """Test creating a R handler."""
        handler = LanguageHandlerFactory.create_handler("r")
        assert handler.__class__.__name__ == "RHandler"

    def test_create_handler_with_logger(self) -> None:
        """Test creating a handler with custom logger."""
        custom_logger = logging.getLogger("custom")
        handler = LanguageHandlerFactory.create_handler("python", custom_logger)
        assert isinstance(handler, PythonHandler)
        assert handler.logger == custom_logger

    def test_create_unsupported_handler(self) -> None:
        """Test creating an unsupported handler."""
        with pytest.raises(LanguageNotSupportedError, match="Language unsupported is not supported"):
            LanguageHandlerFactory.create_handler("unsupported")


class TestAbstractLanguageHandler:
    """Test AbstractLanguageHandler base functionality."""

    def test_is_support_plot_detection_true(self) -> None:
        """Test is_support_plot_detection when plot detection is configured."""
        handler = PythonHandler()  # Has plot detection configured
        assert handler.is_support_plot_detection is True

    def test_filter_comments_removes_multiline(self) -> None:
        """Test filter_comments removes multiline comments."""
        handler = PythonHandler()
        code = "print('hello')\n'''This is a comment'''\nprint('world')"

        filtered = handler.filter_comments(code)

        assert "print('hello')" in filtered
        assert "print('world')" in filtered
        assert "This is a comment" not in filtered

    def test_filter_comments_removes_inline(self) -> None:
        """Test filter_comments removes inline comments."""
        handler = PythonHandler()
        code = "print('hello')  # This is a comment\nprint('world')"

        filtered = handler.filter_comments(code)

        assert "print('hello')" in filtered
        assert "print('world')" in filtered
        assert "This is a comment" not in filtered

    def test_filter_comments_removes_both(self) -> None:
        """Test filter_comments removes both multiline and inline comments."""
        handler = PythonHandler()
        code = """
        print('start')  # Inline comment
        '''
        Multiline comment
        spanning multiple lines
        '''
        print('end')
        """

        filtered = handler.filter_comments(code)

        assert "print('start')" in filtered
        assert "print('end')" in filtered
        assert "Inline comment" not in filtered
        assert "Multiline comment" not in filtered
        assert "spanning multiple lines" not in filtered
