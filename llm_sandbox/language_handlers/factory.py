import logging
from typing import Any, ClassVar

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.exceptions import LanguageNotSupportedError

from .base import AbstractLanguageHandler
from .cpp_handler import CppHandler
from .go_handler import GoHandler
from .java_handler import JavaHandler
from .javascript_handler import JavaScriptHandler
from .python_handler import PythonHandler
from .r_handler import RHandler
from .ruby_handler import RubyHandler


class LanguageHandlerFactory:
    """Factory for creating language-specific handlers."""

    _handlers: ClassVar[dict[str, Any]] = {
        str(SupportedLanguage.PYTHON): PythonHandler,
        str(SupportedLanguage.JAVASCRIPT): JavaScriptHandler,
        str(SupportedLanguage.JAVA): JavaHandler,
        str(SupportedLanguage.CPP): CppHandler,
        str(SupportedLanguage.GO): GoHandler,
        str(SupportedLanguage.RUBY): RubyHandler,
        str(SupportedLanguage.R): RHandler,
    }

    @classmethod
    def create_handler(cls, language: str, logger: logging.Logger | None = None) -> AbstractLanguageHandler:
        """Create handler for specified language."""
        if language.lower() not in cls._handlers:
            raise LanguageNotSupportedError(language)

        handler_class = cls._handlers[language.lower()]
        if not issubclass(handler_class, AbstractLanguageHandler):
            msg = f"Handler class {handler_class} is not a subclass of AbstractLanguageHandler"
            raise TypeError(msg)

        return handler_class(logger=logger)  # type: ignore[no-any-return]

    @classmethod
    def get_supported_languages(cls) -> list[str]:
        """Get list of supported languages."""
        return list(cls._handlers.keys())

    @classmethod
    def register_handler(cls, language: str, handler_class: type[AbstractLanguageHandler]) -> None:
        """Register new language handler."""
        cls._handlers[language.lower()] = handler_class
