"""Runtime context for language handlers.

This module provides a RuntimeContext dataclass that contains runtime paths
and configuration needed by language handlers, avoiding circular dependencies
between session and handler classes.
"""

from dataclasses import dataclass


@dataclass
class RuntimeContext:
    """Runtime execution context for language handlers.

    Contains paths and configuration that language handlers need
    without requiring a reference to the entire session object.
    """

    workdir: str
    """Working directory for code execution."""

    python_executable_path: str | None = None
    """Path to Python executable in the virtual environment."""

    pip_executable_path: str | None = None
    """Path to pip executable in the virtual environment."""

    pip_cache_dir: str | None = None
    """Path to pip cache directory."""
