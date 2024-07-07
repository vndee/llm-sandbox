from dataclasses import dataclass


@dataclass
class SupportedLanguage:
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    CPP = "cpp"
    GO = "go"
    RUBY = "ruby"


@dataclass
class DefaultImage:
    PYTHON = "python:3.9.19-bullseye"
    JAVA = "openjdk:11.0.12-jdk-bullseye"
    JAVASCRIPT = "node:16.6.1-bullseye"
    CPP = "gcc:11.2.0-bullseye"
    GO = "golang:1.17.0-bullseye"
    RUBY = "ruby:3.0.2-bullseye"


SupportedLanguageValues = [
    v for k, v in SupportedLanguage.__dict__.items() if not k.startswith("__")
]
