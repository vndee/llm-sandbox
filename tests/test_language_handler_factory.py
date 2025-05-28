# ruff: noqa: SLF001, PLR2004

import logging
from typing import TYPE_CHECKING

import pytest

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.exceptions import LanguageNotSupportedError
from llm_sandbox.language_handlers.base import AbstractLanguageHandler, PlotOutput
from llm_sandbox.language_handlers.cpp_handler import CppHandler
from llm_sandbox.language_handlers.factory import LanguageHandlerFactory
from llm_sandbox.language_handlers.go_handler import GoHandler
from llm_sandbox.language_handlers.java_handler import JavaHandler
from llm_sandbox.language_handlers.javascript_handler import JavaScriptHandler
from llm_sandbox.language_handlers.python_handler import PythonHandler
from llm_sandbox.language_handlers.ruby_handler import RubyHandler

if TYPE_CHECKING:
    from llm_sandbox.language_handlers.base import ContainerProtocol


class TestLanguageHandlerFactory:
    """Test LanguageHandlerFactory functionality."""

    def test_create_python_handler(self) -> None:
        """Test creating Python handler."""
        handler = LanguageHandlerFactory.create_handler("python")

        assert isinstance(handler, PythonHandler)
        assert handler.name == SupportedLanguage.PYTHON
        assert handler.file_extension == "py"

    def test_create_java_handler(self) -> None:
        """Test creating Java handler."""
        handler = LanguageHandlerFactory.create_handler("java")

        assert isinstance(handler, JavaHandler)
        assert handler.name == SupportedLanguage.JAVA
        assert handler.file_extension == "java"

    def test_create_javascript_handler(self) -> None:
        """Test creating JavaScript handler."""
        handler = LanguageHandlerFactory.create_handler("javascript")

        assert isinstance(handler, JavaScriptHandler)
        assert handler.name == SupportedLanguage.JAVASCRIPT
        assert handler.file_extension == "js"

    def test_create_cpp_handler(self) -> None:
        """Test creating C++ handler."""
        handler = LanguageHandlerFactory.create_handler("cpp")

        assert isinstance(handler, CppHandler)
        assert handler.name == SupportedLanguage.CPP
        assert handler.file_extension == "cpp"

    def test_create_go_handler(self) -> None:
        """Test creating Go handler."""
        handler = LanguageHandlerFactory.create_handler("go")

        assert isinstance(handler, GoHandler)
        assert handler.name == SupportedLanguage.GO
        assert handler.file_extension == "go"

    def test_create_ruby_handler(self) -> None:
        """Test creating Ruby handler (if implemented)."""
        # Note: Ruby handler might not be fully implemented yet
        try:
            handler = LanguageHandlerFactory.create_handler("ruby")
            assert isinstance(handler, RubyHandler)
            assert handler.name == SupportedLanguage.RUBY
            assert handler.file_extension == "rb"
        except LanguageNotSupportedError:
            # Ruby handler not implemented yet
            pass

    def test_create_handler_with_custom_logger(self) -> None:
        """Test creating handler with custom logger."""
        custom_logger = logging.getLogger("test_logger")
        handler = LanguageHandlerFactory.create_handler("python", logger=custom_logger)

        assert isinstance(handler, PythonHandler)
        assert handler.logger == custom_logger

    def test_create_handler_case_insensitive(self) -> None:
        """Test creating handler with different case."""
        handlers = [
            LanguageHandlerFactory.create_handler("PYTHON"),
            LanguageHandlerFactory.create_handler("Python"),
            LanguageHandlerFactory.create_handler("python"),
        ]

        for handler in handlers:
            assert isinstance(handler, PythonHandler)

    def test_create_handler_unsupported_language(self) -> None:
        """Test creating handler for unsupported language."""
        with pytest.raises(LanguageNotSupportedError):
            LanguageHandlerFactory.create_handler("unsupported_language")

    def test_get_supported_languages(self) -> None:
        """Test getting list of supported languages."""
        supported = LanguageHandlerFactory.get_supported_languages()

        assert isinstance(supported, list)
        assert "python" in supported
        assert "java" in supported
        assert "javascript" in supported
        assert "cpp" in supported
        assert "go" in supported

        # All supported languages should be valid SupportedLanguage values
        for lang in supported:
            assert lang in [member.value for member in SupportedLanguage]

    def test_register_custom_handler(self) -> None:
        """Test registering a custom language handler."""

        class CustomHandler(AbstractLanguageHandler):
            def __init__(self, logger: logging.Logger | None = None) -> None:
                super().__init__(logger)
                from llm_sandbox.language_handlers.base import LanguageConfig

                self.config = LanguageConfig(
                    name="custom",
                    file_extension="cust",
                    execution_commands=["custom {file}"],
                    package_manager="custom install",
                )

            def inject_plot_detection_code(self, code: str) -> str:
                return code

            def extract_plots(self, container: "ContainerProtocol", output_dir: str) -> list[PlotOutput]:  # noqa: ARG002
                return []

            def get_import_patterns(self, module: str) -> str:
                return f"import {module}"

            @staticmethod
            def get_multiline_comment_patterns() -> str:
                return r"/\*.*?\*/"

            @staticmethod
            def get_inline_comment_patterns() -> str:
                return r"//.*$"

        # Register the custom handler
        LanguageHandlerFactory.register_handler("custom", CustomHandler)

        # Test that it can be created
        handler = LanguageHandlerFactory.create_handler("custom")
        assert isinstance(handler, CustomHandler)
        assert handler.name == "custom"
        assert handler.file_extension == "cust"

        # Test that it appears in supported languages
        supported = LanguageHandlerFactory.get_supported_languages()
        assert "custom" in supported

    def test_register_invalid_handler_class(self) -> None:
        """Test registering an invalid handler class."""

        class InvalidHandler:
            """Not a subclass of AbstractLanguageHandler."""

        # Register the invalid handler
        LanguageHandlerFactory.register_handler("invalid", InvalidHandler)  # type: ignore[arg-type]

        # Creating it should raise TypeError
        with pytest.raises(TypeError, match="is not a subclass of AbstractLanguageHandler"):
            LanguageHandlerFactory.create_handler("invalid")

    def test_all_registered_handlers_are_abstract_subclasses(self) -> None:
        """Test that all registered handlers are proper subclasses."""
        supported_languages = LanguageHandlerFactory.get_supported_languages()

        for lang in supported_languages:
            if lang not in ["custom", "invalid"]:  # Skip test-specific handlers
                handler = LanguageHandlerFactory.create_handler(lang)
                assert isinstance(handler, AbstractLanguageHandler)

                # Test that required methods exist
                assert hasattr(handler, "inject_plot_detection_code")
                assert hasattr(handler, "extract_plots")
                assert hasattr(handler, "get_import_patterns")
                assert hasattr(handler, "get_multiline_comment_patterns")
                assert hasattr(handler, "get_inline_comment_patterns")

    def test_handler_consistency_across_creation(self) -> None:
        """Test that multiple creations of the same handler type are consistent."""
        handler1 = LanguageHandlerFactory.create_handler("python")
        handler2 = LanguageHandlerFactory.create_handler("python")

        # Should be different instances
        assert handler1 is not handler2

        # But should have same configuration
        assert type(handler1) is type(handler2)
        assert handler1.name == handler2.name
        assert handler1.file_extension == handler2.file_extension

    def test_factory_state_isolation(self) -> None:
        """Test that factory maintains proper state isolation."""
        # Get initial state
        initial_handlers = LanguageHandlerFactory._handlers.copy()
        initial_languages = set(LanguageHandlerFactory.get_supported_languages())

        # Register a temporary handler
        class TempHandler(AbstractLanguageHandler):
            def __init__(self, logger: logging.Logger | None = None) -> None:
                super().__init__(logger)
                from llm_sandbox.language_handlers.base import LanguageConfig

                self.config = LanguageConfig(
                    name="temp",
                    file_extension="tmp",
                    execution_commands=["temp {file}"],
                    package_manager="temp install",
                )

            def inject_plot_detection_code(self, code: str) -> str:
                return code

            def extract_plots(self, container: "ContainerProtocol", output_dir: str) -> list[PlotOutput]:  # noqa: ARG002
                return []

            def get_import_patterns(self, module: str) -> str:
                return f"import {module}"

            @staticmethod
            def get_multiline_comment_patterns() -> str:
                return r"/\*.*?\*/"

            @staticmethod
            def get_inline_comment_patterns() -> str:
                return r"//.*$"

        LanguageHandlerFactory.register_handler("temp", TempHandler)

        # Verify registration worked
        assert "temp" in LanguageHandlerFactory.get_supported_languages()
        handler = LanguageHandlerFactory.create_handler("temp")
        assert isinstance(handler, TempHandler)

        # Clean up by restoring initial state
        LanguageHandlerFactory._handlers = initial_handlers

        # Verify cleanup worked
        current_languages = set(LanguageHandlerFactory.get_supported_languages())
        assert current_languages == initial_languages

    def test_create_handler_with_none_logger(self) -> None:
        """Test creating handler with None logger (should use default)."""
        handler = LanguageHandlerFactory.create_handler("python", logger=None)

        assert isinstance(handler, PythonHandler)
        assert handler.logger is not None  # Should have default logger

    def test_language_enum_consistency(self) -> None:
        """Test that all handlers match their corresponding enum values."""
        enum_languages = {member.value: member for member in SupportedLanguage}
        supported_languages = LanguageHandlerFactory.get_supported_languages()

        for lang in supported_languages:
            if lang in enum_languages:  # Skip test-registered languages
                handler = LanguageHandlerFactory.create_handler(lang)
                expected_enum = enum_languages[lang]
                assert handler.name == expected_enum
