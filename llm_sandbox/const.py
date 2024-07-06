from dataclasses import dataclass


@dataclass
class SupportedLanguage:
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    CPP = "cpp"
    GO = "go"
    RUBY = "ruby"


SupportedLanguageValues = [
    v for k, v in SupportedLanguage.__dict__.items() if not k.startswith("__")
]
