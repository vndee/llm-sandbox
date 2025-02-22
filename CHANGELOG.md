# Changelog

All notable changes to LLM Sandbox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2025-02-22
### Fixed
- Add error handling for command execution in SandboxDockerSession (https://github.com/vndee/llm-sandbox/pull/28)

## [0.2.2] - 2025-02-12
### Fixed
- Refactor SandboxBackend to inherit from str and Enum for improved python 3.10 and earlier compatibility

## [0.2.1] - 2025-02-10
### Added
- Add support for additional kwargs in SandboxSession (https://github.com/vndee/llm-sandbox/pull/25)

## [0.2.0] - 2025-02-02

### Added
- CHANGELOG.md
- SessionFactory for creating sessions with context manager support, change the init method from `use_kubernetes`, `use_podman` to `backend` for more flexibility. See https://github.com/vndee/llm-sandbox/blob/main/llm_sandbox/session.py
- Separated library dependencies as extras (https://github.com/vndee/llm-sandbox/issues/18). Now you can install only the dependencies you need, for example: `pip install llm-sandbox[kubernetes]`, `pip install llm-sandbox[podman]`, `pip install llm-sandbox[docker]`.
- `ConsoleOutput` for docker backend now support `exit_code` (https://github.com/vndee/llm-sandbox/pull/15)
- `pod_manifest` settings for k8s backend (https://github.com/vndee/llm-sandbox/pull/16)
