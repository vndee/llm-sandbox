from typing import ClassVar

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.exceptions import LanguageNotSupportedError

from .base import AbstractLanguageHandler
from .cpp_handler import CppHandler
from .go_handler import GoHandler
from .java_handler import JavaHandler
from .javascript_handler import JavaScriptHandler
from .python_handler import PythonHandler
from .ruby_handler import RubyHandler


class LanguageHandlerFactory:
    """Factory for creating language-specific handlers."""

    _handlers: ClassVar[dict[str, type[AbstractLanguageHandler]]] = {
        SupportedLanguage.PYTHON: PythonHandler,
        SupportedLanguage.JAVASCRIPT: JavaScriptHandler,
        SupportedLanguage.JAVA: JavaHandler,
        SupportedLanguage.CPP: CppHandler,
        SupportedLanguage.GO: GoHandler,
        SupportedLanguage.RUBY: RubyHandler,
    }

    @classmethod
    def create_handler(cls, language: str) -> AbstractLanguageHandler:
        """Create handler for specified language."""
        if language.lower() not in cls._handlers:
            raise LanguageNotSupportedError(language)

        handler_class = cls._handlers[language.lower()]
        return handler_class()

    @classmethod
    def get_supported_languages(cls) -> list[str]:
        """Get list of supported languages."""
        return list(cls._handlers.keys())

    @classmethod
    def register_handler(
        cls, language: str, handler_class: type[AbstractLanguageHandler]
    ) -> None:
        """Register new language handler."""
        cls._handlers[language.lower()] = handler_class
