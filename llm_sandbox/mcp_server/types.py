from typing import TypedDict


class LanguageExample(TypedDict):
    """An example of a programming language."""

    title: str
    description: str
    code: str


class LanguageDetails(TypedDict):
    """Details about a programming language."""

    version: str
    package_manager: str
    preinstalled_libraries: list[str]
    use_cases: list[str]
    visualization_support: bool
    examples: list[LanguageExample]
