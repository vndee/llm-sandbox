# Changelog

All notable changes to LLM Sandbox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-03-21

### Added
- Security scanning functionality with configurable severity levels
  - Code analysis for potential security threats
  - Pattern-based vulnerability detection
  - Configurable strict and non-strict modes
  - Support for detecting system calls, code execution, file operations, and network access

- Resource monitoring and limits
  - CPU usage tracking and limits
  - Memory usage monitoring
  - Network traffic monitoring
  - Execution time limits
  - Detailed resource usage statistics

- Factory pattern for session creation
  - Unified factory for creating different session types
  - Support for Docker, Kubernetes, and Podman backends
  - Better configuration management
  - Improved session lifecycle management

- Improved error handling
  - Custom exception hierarchy
  - More detailed error messages
  - Better error classification
  - Proper error propagation

- Better logging and monitoring
  - Structured logging
  - Resource usage tracking
  - Performance metrics
  - Debug information

### Changed
- Refactored session management
  - Better separation of concerns
  - More modular code structure
  - Improved session lifecycle
  - Better resource cleanup

- Improved test suite
  - Converted to pytest from unittest
  - Added parameterized tests
  - Better test coverage
  - More realistic test scenarios

- Enhanced container management
  - Better resource limits enforcement
  - Improved container lifecycle
  - Better cleanup handling
  - Support for container state persistence

### Fixed
- Resource leaks in container management
- Inconsistent error handling
- Security vulnerabilities in code execution
- Memory management issues
