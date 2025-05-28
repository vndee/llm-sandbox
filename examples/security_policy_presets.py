"""Pre-configured security policy presets for common use cases.

This module provides ready-to-use security policies for different environments
and risk levels, making it easy to get started with LLM Sandbox security.
"""

import logging

from llm_sandbox.security import RestrictedModule, SecurityIssueSeverity, SecurityPattern, SecurityPolicy


def create_development_policy() -> SecurityPolicy:
    """Create a security policy suitable for development environments.

    This policy is relatively permissive, allowing most operations while
    blocking only the most dangerous ones.

    Returns:
        SecurityPolicy: A development-friendly security policy

    """
    patterns = [
        # Only block the most dangerous operations
        SecurityPattern(
            pattern=r"\bos\.system\s*\(",
            description="Direct system command execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\brm\s+-rf\s+/",
            description="Dangerous recursive file deletion",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bformat\s+[A-Za-z]:[/\\]",
            description="Disk formatting commands",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ]

    restricted_modules = [
        # Only block modules that can cause immediate system damage
        RestrictedModule(
            name="ctypes",
            description="Direct system call access",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.HIGH,  # Only block HIGH severity
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


def create_production_policy() -> SecurityPolicy:
    """Create a security policy suitable for production environments.

    This policy is strict and blocks many potentially dangerous operations.

    Returns:
        SecurityPolicy: A production-ready security policy

    """
    patterns = [
        # System command execution
        SecurityPattern(
            pattern=r"\bos\.system\s*\(",
            description="System command execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bsubprocess\.(run|call|Popen|check_output)\s*\(",
            description="Subprocess execution",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # Dynamic code execution
        SecurityPattern(
            pattern=r"\beval\s*\(",
            description="Dynamic code evaluation",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        SecurityPattern(
            pattern=r"\bexec\s*\(",
            description="Dynamic code execution",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # File system operations
        SecurityPattern(
            pattern=r"\bopen\s*\([^)]*['\"][wa]['\"][^)]*\)",
            description="File write operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        SecurityPattern(
            pattern=r"\bos\.(remove|unlink|rmdir)\s*\(",
            description="File deletion operations",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # Environment access
        SecurityPattern(
            pattern=r"\bos\.environ",
            description="Environment variable access",
            severity=SecurityIssueSeverity.LOW,
        ),
        # Network operations
        SecurityPattern(
            pattern=r"\bsocket\.socket\s*\(",
            description="Raw socket creation",
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
            description="Subprocess management",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="ctypes",
            description="Foreign function library",
            severity=SecurityIssueSeverity.HIGH,
        ),
        RestrictedModule(
            name="socket",
            description="Network socket operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        RestrictedModule(
            name="multiprocessing",
            description="Process-based parallelism",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.LOW,  # Block LOW and above
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


def create_educational_policy() -> SecurityPolicy:
    """Create a security policy suitable for educational environments.

    This policy allows learning-oriented operations while preventing
    system damage and network abuse.

    Returns:
        SecurityPolicy: An education-focused security policy

    """
    patterns = [
        # System operations
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
        # File system damage
        SecurityPattern(
            pattern=r"\bos\.(remove|unlink|rmdir)\s*\(",
            description="File deletion operations",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bshutil\.rmtree\s*\(",
            description="Directory tree deletion",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # Network abuse prevention
        SecurityPattern(
            pattern=r"\bsocket\.socket\s*\([^)]*SOCK_RAW",
            description="Raw socket creation",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # Allow eval/exec but warn about it
        SecurityPattern(
            pattern=r"\beval\s*\(",
            description="Dynamic code evaluation (educational warning)",
            severity=SecurityIssueSeverity.LOW,
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
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # Allow networking libraries for learning
        RestrictedModule(
            name="socket",
            description="Network socket operations",
            severity=SecurityIssueSeverity.LOW,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.MEDIUM,  # Block MEDIUM and above
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


def create_data_science_policy() -> SecurityPolicy:
    """Create a security policy suitable for data science environments.

    This policy allows data processing and analysis while preventing
    system access and network abuse.

    Returns:
        SecurityPolicy: A data science-focused security policy

    """
    patterns = [
        # System operations
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
        # File system operations (allow reading, restrict writing)
        SecurityPattern(
            pattern=r"\bopen\s*\([^)]*['\"]w['\"][^)]*\)",
            description="File write operations",
            severity=SecurityIssueSeverity.LOW,  # Allow but log
        ),
        SecurityPattern(
            pattern=r"\bos\.(remove|unlink|rmdir)\s*\(",
            description="File deletion operations",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # Network operations (allow HTTP requests, block raw sockets)
        SecurityPattern(
            pattern=r"\bsocket\.socket\s*\(",
            description="Raw socket creation",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # Pickle security (common in data science but risky)
        SecurityPattern(
            pattern=r"\bpickle\.loads?\s*\(",
            description="Pickle deserialization (potential security risk)",
            severity=SecurityIssueSeverity.LOW,
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
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # Allow common data science networking
        RestrictedModule(
            name="urllib",
            description="URL handling library",
            severity=SecurityIssueSeverity.LOW,
        ),
        RestrictedModule(
            name="requests",
            description="HTTP requests library",
            severity=SecurityIssueSeverity.LOW,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.MEDIUM,  # Block MEDIUM and above
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


def create_web_scraping_policy() -> SecurityPolicy:
    """Create a security policy suitable for web scraping environments.

    This policy allows HTTP requests and web-related operations while
    preventing system access and file system abuse.

    Returns:
        SecurityPolicy: A web scraping-focused security policy

    """
    patterns = [
        # System operations
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
        # File system operations
        SecurityPattern(
            pattern=r"\bos\.(remove|unlink|rmdir)\s*\(",
            description="File deletion operations",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # Allow file writing but monitor it
        SecurityPattern(
            pattern=r"\bopen\s*\([^)]*['\"]w['\"][^)]*\)",
            description="File write operations",
            severity=SecurityIssueSeverity.LOW,
        ),
        # Network abuse prevention
        SecurityPattern(
            pattern=r"\btime\.sleep\s*\(\s*[0-9]*\.?[0-9]+\s*\)",
            description="Sleep operations (potential rate limiting bypass)",
            severity=SecurityIssueSeverity.LOW,
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
        # Allow networking libraries (essential for web scraping)
        RestrictedModule(
            name="urllib",
            description="URL handling library",
            severity=SecurityIssueSeverity.LOW,
        ),
        RestrictedModule(
            name="requests",
            description="HTTP requests library",
            severity=SecurityIssueSeverity.LOW,
        ),
        RestrictedModule(
            name="socket",
            description="Network socket operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.MEDIUM,  # Block MEDIUM and above
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


def create_minimal_policy() -> SecurityPolicy:
    """Create a minimal security policy with only essential protections.

    This policy provides the bare minimum security while allowing
    maximum flexibility.

    Returns:
        SecurityPolicy: A minimal security policy

    """
    patterns = [
        # Only block the most dangerous operations
        SecurityPattern(
            pattern=r"\bos\.system\s*\([^)]*(?:rm|del|format)[^)]*\)",
            description="Dangerous file deletion commands",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bsubprocess\.(run|call)\s*\([^)]*(?:rm|del|format)[^)]*\)",
            description="Dangerous subprocess commands",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ]

    restricted_modules = [
        # Only block modules that can cause immediate system damage
        RestrictedModule(
            name="ctypes",
            description="Direct system call access",
            severity=SecurityIssueSeverity.HIGH,
        ),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.HIGH,  # Only block HIGH severity
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


def create_strict_policy() -> SecurityPolicy:
    """Create a strict security policy for high-security environments.

    This policy blocks almost all potentially dangerous operations.

    Returns:
        SecurityPolicy: A strict security policy

    """
    patterns = [
        # System operations
        SecurityPattern(
            pattern=r"\bos\.\w+\s*\(",
            description="Any os module function call",
            severity=SecurityIssueSeverity.HIGH,
        ),
        SecurityPattern(
            pattern=r"\bsubprocess\.[\w]+\s*\(",
            description="Any subprocess module function call",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # Dynamic code execution
        SecurityPattern(
            pattern=r"\b(eval|exec|compile)\s*\(",
            description="Dynamic code execution functions",
            severity=SecurityIssueSeverity.HIGH,
        ),
        # File operations
        SecurityPattern(
            pattern=r"\bopen\s*\(",
            description="File operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # Imports
        SecurityPattern(
            pattern=r"\b__import__\s*\(",
            description="Dynamic imports",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
        # Attribute access
        SecurityPattern(
            pattern=r"\b(getattr|setattr|delattr)\s*\(",
            description="Dynamic attribute access",
            severity=SecurityIssueSeverity.LOW,
        ),
        # Network operations
        SecurityPattern(
            pattern=r"\bsocket\.[\w]+\s*\(",
            description="Socket operations",
            severity=SecurityIssueSeverity.MEDIUM,
        ),
    ]

    restricted_modules = [
        # Block all potentially dangerous modules
        RestrictedModule(name="os", description="Operating system interface", severity=SecurityIssueSeverity.HIGH),
        RestrictedModule(name="sys", description="System-specific parameters", severity=SecurityIssueSeverity.HIGH),
        RestrictedModule(name="subprocess", description="Subprocess management", severity=SecurityIssueSeverity.HIGH),
        RestrictedModule(name="ctypes", description="Foreign function library", severity=SecurityIssueSeverity.HIGH),
        RestrictedModule(name="socket", description="Network socket operations", severity=SecurityIssueSeverity.MEDIUM),
        RestrictedModule(name="urllib", description="URL handling library", severity=SecurityIssueSeverity.MEDIUM),
        RestrictedModule(name="requests", description="HTTP requests library", severity=SecurityIssueSeverity.MEDIUM),
        RestrictedModule(
            name="multiprocessing", description="Process-based parallelism", severity=SecurityIssueSeverity.MEDIUM
        ),
        RestrictedModule(name="threading", description="Thread-based parallelism", severity=SecurityIssueSeverity.LOW),
        RestrictedModule(name="pickle", description="Python object serialization", severity=SecurityIssueSeverity.LOW),
        RestrictedModule(
            name="marshal", description="Internal Python object serialization", severity=SecurityIssueSeverity.LOW
        ),
        RestrictedModule(name="shelve", description="Python object persistence", severity=SecurityIssueSeverity.LOW),
    ]

    return SecurityPolicy(
        severity_threshold=SecurityIssueSeverity.LOW,  # Block LOW and above (very strict)
        patterns=patterns,
        restricted_modules=restricted_modules,
    )


# Convenience dictionary for easy access to presets
SECURITY_PRESETS = {
    "development": create_development_policy,
    "production": create_production_policy,
    "educational": create_educational_policy,
    "data_science": create_data_science_policy,
    "web_scraping": create_web_scraping_policy,
    "minimal": create_minimal_policy,
    "strict": create_strict_policy,
}


def get_security_policy(preset_name: str) -> SecurityPolicy:
    """Get a pre-configured security policy by name.

    Args:
        preset_name: Name of the preset policy

    Returns:
        SecurityPolicy: The requested security policy

    Raises:
        ValueError: If preset_name is not recognized

    """
    if preset_name not in SECURITY_PRESETS:
        available = ", ".join(SECURITY_PRESETS.keys())
        msg = f"Unknown preset '{preset_name}'. Available presets: {available}"
        raise ValueError(msg)

    return SECURITY_PRESETS[preset_name]()


def list_available_presets() -> list[str]:
    """List all available security policy presets.

    Returns:
        List[str]: Names of available presets

    """
    return list(SECURITY_PRESETS.keys())


if __name__ == "__main__":
    """Demonstrate security policy presets."""
    import logging

    from llm_sandbox import SandboxSession

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Test code samples
    test_codes = [
        "print('Hello, World!')",  # Safe
        "import requests\nrequests.get('http://example.com')",  # HTTP request
        "import os\nos.system('ls')",  # System command
        "eval('2 + 2')",  # Dynamic evaluation
    ]

    # Test each preset
    for preset_name in list_available_presets():
        logger.info("\nTesting %s policy:", preset_name.upper())
        policy = get_security_policy(preset_name)

        with SandboxSession(lang="python", security_policy=policy, verbose=False) as session:
            for i, code in enumerate(test_codes, 1):
                is_safe, violations = session.is_safe(code)
                status = "✅ SAFE" if is_safe else "❌ BLOCKED"
                logger.info("  Test %s: %s - %s...", i, status, code[:30])

                if violations:
                    for violation in violations[:2]:  # Show first 2 violations
                        logger.info("    → %s", violation.description)

    logger.info("\nAvailable presets: %s", ", ".join(list_available_presets()))
