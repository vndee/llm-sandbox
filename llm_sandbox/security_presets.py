"""Security presets combining SecurityPolicy with backend-specific runtime configurations.

This module provides predefined security configurations that combine code-level security
policies with container runtime restrictions for different use cases and security requirements.
"""

from dataclasses import dataclass
from typing import Any

from llm_sandbox.security import RestrictedModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy


@dataclass
class SecurityConfiguration:
    """Complete security configuration combining policy and runtime settings.

    Attributes:
        security_policy: Code-level security policy for pre-execution analysis.
        runtime_config: Backend-specific container runtime configuration.
            For Docker/Podman: parameters passed to containers.create()
            For Kubernetes: use pod_manifest parameter instead (not runtime_config).
        pod_manifest: Kubernetes pod manifest (alternative to runtime_config).

    """

    security_policy: SecurityPolicy
    runtime_config: dict[str, Any] | None = None
    pod_manifest: dict[str, Any] | None = None


def _create_development_policy_python() -> SecurityPolicy:
    """Create development security policy for Python."""
    patterns = [
        SecurityPattern(
            pattern=r"\bos\.system\s*\(",
            description="Direct system command execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bsubprocess\.(run|call|Popen)\s*\(",
            description="Process execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ]

    restricted_modules = [
        RestrictedModule(
            name="ctypes",
            description="Foreign function library",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.HIGH,
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


def _create_production_policy_python() -> SecurityPolicy:
    """Create production security policy for Python."""
    patterns = [
        SecurityPattern(
            pattern=r"\bos\.system\s*\(",
            description="System command execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bsubprocess\.(run|call|Popen)\s*\(",
            description="Subprocess execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\beval\s*\(",
            description="Code evaluation",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        SecurityPattern(
            pattern=r"\bexec\s*\(",
            description="Code execution",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        SecurityPattern(
            pattern=r"\b__import__\s*\(",
            description="Dynamic imports",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
    ]

    restricted_modules = [
        RestrictedModule(
            name="os",
            description="Operating system interface",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="subprocess",
            description="Process execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="ctypes",
            description="Foreign functions",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="importlib",
            description="Dynamic imports",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.MEDIUM,
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


def _create_strict_policy_python() -> SecurityPolicy:
    """Create strict security policy for Python."""
    patterns = [
        SecurityPattern(
            pattern=r"\bopen\s*\([^)]*['\"][wa]['\"][^)]*\)",
            description="File write operations",
            severity=SecurityIssueSeverity.LOW,
        ),
        SecurityPattern(
            pattern=r"\bos\.(remove|unlink|rmdir)\s*\(",
            description="File deletion",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        SecurityPattern(
            pattern=r"\brequests\.(get|post|put|delete)\s*\(",
            description="HTTP requests",
            severity=SecurityIssueSeverity.LOW,
        ),
        SecurityPattern(
            pattern=r"\bsocket\.socket\s*\(",
            description="Socket creation",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        SecurityPattern(
            pattern=r"\bos\.system\s*\(",
            description="System commands",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bsubprocess\.",
            description="Process operations",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ]

    restricted_modules = [
        RestrictedModule(
            name="os",
            description="Operating system interface",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="subprocess",
            description="Process execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="ctypes",
            description="Foreign functions",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="socket",
            description="Network operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        RestrictedModule(
            name="requests",
            description="HTTP library",
            severity=SecurityIssueSeverity.LOW,
        ),
        RestrictedModule(
            name="urllib",
            description="URL library",
            severity=SecurityIssueSeverity.LOW,
        ),
        RestrictedModule(
            name="shutil",
            description="File operations",
            severity=SecurityIssueSeverity.LOW,
        ),
        RestrictedModule(
            name="importlib",
            description="Dynamic imports",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        RestrictedModule(
            name="pickle",
            description="Serialization",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.LOW,
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


def _create_educational_policy_python() -> SecurityPolicy:
    """Create educational security policy for Python."""
    patterns = [
        SecurityPattern(
            pattern=r"\bos\.system\s*\(",
            description="System command execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bsubprocess\.(run|call|Popen)\s*\(",
            description="Subprocess execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bos\.(remove|unlink|rmdir)\s*\(",
            description="File deletion operations",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ]

    restricted_modules = [
        RestrictedModule(
            name="os",
            description="Operating system interface",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="subprocess",
            description="Subprocess management",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="ctypes",
            description="Foreign function library",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.MEDIUM,
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


def _create_development_runtime_docker() -> dict[str, Any]:
    """Create development runtime config for Docker/Podman."""
    return {
        "mem_limit": "1g",
        "cpu_period": 100000,
        "cpu_quota": 200000,  # 2 CPUs worth of quota
        "network_mode": "bridge",  # Internet access allowed
        "read_only": False,
        # user defaults to "root" when not specified
    }


def _create_production_runtime_docker() -> dict[str, Any]:
    """Create production runtime config for Docker/Podman."""
    return {
        "mem_limit": "512m",
        "cpu_period": 100000,
        "cpu_quota": 100000,  # 1 CPU worth of quota
        "network_mode": "none",  # No network access
        "read_only": True,
        "tmpfs": {"/tmp": "size=100m,noexec"},
        "user": "nobody:nogroup",
    }


def _create_strict_runtime_docker() -> dict[str, Any]:
    """Create strict runtime config for Docker/Podman."""
    return {
        "mem_limit": "128m",
        "cpu_period": 100000,
        "cpu_quota": 50000,  # 0.5 CPU worth of quota
        "network_mode": "none",
        "read_only": True,
        "tmpfs": {"/tmp": "size=50m,noexec,nosuid,nodev"},
        "user": "nobody:nogroup",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
    }


def _create_educational_runtime_docker() -> dict[str, Any]:
    """Create educational runtime config for Docker/Podman."""
    return {
        "mem_limit": "256m",
        "cpu_period": 100000,
        "cpu_quota": 100000,  # 1 CPU worth of quota
        "network_mode": "bridge",  # Allow access to educational resources
        "read_only": True,
        "tmpfs": {"/tmp": "size=100m"},
        "user": "1000:1000",  # Run as non-root student user
    }


def get_security_preset(
    preset_name: str,
    language: str = "python",
    backend: str = "docker",
) -> SecurityConfiguration:
    """Get predefined security configuration for specific language and backend.

    This function returns a complete security configuration combining SecurityPolicy
    (code-level analysis) with backend-specific runtime configurations.

    Args:
        preset_name: Security preset name. One of:
            - "development": Permissive, for local development and testing
            - "production": Strict, for production applications
            - "strict": Very strict, for untrusted code execution
            - "educational": Balanced, for educational platforms
        language: Programming language. Currently supports "python".
        backend: Container backend. One of:
            - "docker": Docker containers
            - "podman": Podman containers
            - "kubernetes": Kubernetes pods (returns None for runtime_config,
              use pod_manifest parameter instead)

    Returns:
        SecurityConfiguration with security_policy and runtime_config.

    Raises:
        ValueError: If preset_name, language, or backend is not supported.

    Example:
        >>> config = get_security_preset("production", "python", "docker")
        >>> session = SandboxSession(
        ...     lang="python",
        ...     security_policy=config.security_policy,
        ...     runtime_configs=config.runtime_config
        ... )

    """
    # Normalize inputs
    preset_name = preset_name.lower()
    language = language.lower()
    backend = backend.lower()

    # Validate inputs
    valid_presets = {"development", "production", "strict", "educational"}
    if preset_name not in valid_presets:
        msg = f"Unknown preset '{preset_name}'. Available: {', '.join(sorted(valid_presets))}"
        raise ValueError(msg)

    valid_languages = {"python"}
    if language not in valid_languages:
        msg = f"Unsupported language '{language}'. Available: {', '.join(sorted(valid_languages))}"
        raise ValueError(msg)

    valid_backends = {"docker", "podman", "kubernetes"}
    if backend not in valid_backends:
        msg = f"Unsupported backend '{backend}'. Available: {', '.join(sorted(valid_backends))}"
        raise ValueError(msg)

    # Get security policy based on language
    policy_map = {
        "python": {
            "development": _create_development_policy_python,
            "production": _create_production_policy_python,
            "strict": _create_strict_policy_python,
            "educational": _create_educational_policy_python,
        },
    }

    security_policy = policy_map[language][preset_name]()

    # Get runtime config based on backend
    if backend == "kubernetes":
        # Kubernetes doesn't use runtime_config, users should provide pod_manifest
        return SecurityConfiguration(
            security_policy=security_policy,
            runtime_config=None,
            pod_manifest=None,
        )

    # Docker and Podman use the same runtime config structure
    runtime_map = {
        "development": _create_development_runtime_docker,
        "production": _create_production_runtime_docker,
        "strict": _create_strict_runtime_docker,
        "educational": _create_educational_runtime_docker,
    }

    runtime_config = runtime_map[preset_name]()

    return SecurityConfiguration(
        security_policy=security_policy,
        runtime_config=runtime_config,
    )


def list_available_presets() -> list[str]:
    """List all available security preset names.

    Returns:
        List of available preset names.

    """
    return ["development", "production", "strict", "educational"]


def list_supported_languages() -> list[str]:
    """List all supported languages for security presets.

    Returns:
        List of supported language names.

    """
    return ["python"]


def list_supported_backends() -> list[str]:
    """List all supported backends for security presets.

    Returns:
        List of supported backend names.

    """
    return ["docker", "podman", "kubernetes"]
