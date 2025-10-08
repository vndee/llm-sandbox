"""Example demonstrating security presets with runtime configurations.

This example shows how to use predefined security presets that combine
SecurityPolicy (code-level analysis) with backend-specific runtime configurations.
"""

from llm_sandbox import SandboxSession, get_security_preset, list_available_presets


def demo_basic_usage() -> None:
    """Demonstrate basic usage of security presets."""
    print("=" * 80)
    print("Basic Usage of Security Presets")
    print("=" * 80)

    # Get a security configuration preset
    config = get_security_preset("production", "python", "docker")

    print("\nProduction preset configuration:")
    print(f"  Security Policy Threshold: {config.security_policy.severity_threshold}")
    print(f"  Number of Pattern Rules: {len(config.security_policy.patterns or [])}")
    print(f"  Number of Restricted Modules: {len(config.security_policy.restricted_modules or [])}")
    print(f"\n  Runtime Config:")
    for key, value in (config.runtime_config or {}).items():
        print(f"    {key}: {value}")

    # Use the preset in a session
    print("\n  Creating session with preset...")
    print("  (This is a dry run - not actually creating container)")
    print(f"  SandboxSession would be created with:")
    print(f"    - security_policy={config.security_policy}")
    print(f"    - runtime_configs={config.runtime_config}")


def demo_all_presets() -> None:
    """Demonstrate all available presets."""
    print("\n" + "=" * 80)
    print("All Available Security Presets")
    print("=" * 80)

    for preset_name in list_available_presets():
        print(f"\n{preset_name.upper()} Preset:")
        print("-" * 40)

        config = get_security_preset(preset_name, "python", "docker")

        # Security policy details
        policy = config.security_policy
        print(f"  Severity Threshold: {policy.severity_threshold.name}")
        print(f"  Pattern Rules: {len(policy.patterns or [])}")
        print(f"  Restricted Modules: {len(policy.restricted_modules or [])}")

        # Runtime config highlights
        runtime = config.runtime_config or {}
        print(f"  Memory Limit: {runtime.get('mem_limit', 'N/A')}")
        print(f"  CPU Quota: {runtime.get('cpu_quota', 'N/A')} (period: {runtime.get('cpu_period', 'N/A')})")
        print(f"  Network Mode: {runtime.get('network_mode', 'N/A')}")
        print(f"  Read Only: {runtime.get('read_only', False)}")
        print(f"  User: {runtime.get('user', 'root (default)')}")

        if "cap_drop" in runtime:
            print(f"  Capabilities Dropped: {runtime['cap_drop']}")
        if "security_opt" in runtime:
            print(f"  Security Options: {runtime['security_opt']}")


def demo_preset_comparison() -> None:
    """Compare different presets for the same code."""
    print("\n" + "=" * 80)
    print("Preset Comparison")
    print("=" * 80)

    test_codes = [
        ("print('Hello, World!')", "Safe code"),
        ("import requests\\nrequests.get('http://example.com')", "HTTP request"),
        ("import os\\nos.system('ls')", "System command"),
        ("eval('2 + 2')", "Dynamic evaluation"),
    ]

    print("\nNote: This shows what would be blocked at different security levels")
    print("(Not actually executing code)")

    for code, description in test_codes:
        print(f"\n{description}: {code[:50]}...")
        print("  " + "-" * 70)

        for preset_name in ["development", "educational", "production", "strict"]:
            config = get_security_preset(preset_name, "python", "docker")

            # Would be blocked if there are any violations
            # (This is a simplified check - actual check would use session.is_safe())
            print(f"  {preset_name:12s}: Would need full session to check")


def demo_custom_modifications() -> None:
    """Demonstrate customizing a preset."""
    print("\n" + "=" * 80)
    print("Customizing Security Presets")
    print("=" * 80)

    # Start with a preset
    config = get_security_preset("production", "python", "docker")

    print("\nOriginal production preset:")
    print(f"  Memory Limit: {config.runtime_config['mem_limit']}")
    print(f"  User: {config.runtime_config['user']}")

    # Modify runtime config
    config.runtime_config["mem_limit"] = "1g"  # Increase memory
    config.runtime_config["user"] = "1000:1000"  # Change user

    # Add custom security pattern
    from llm_sandbox import SecurityPattern, SecurityIssueSeverity

    custom_pattern = SecurityPattern(
        pattern=r"\bpandas\.",
        description="Pandas library usage",
        severity=SecurityIssueSeverity.LOW,
    )
    config.security_policy.add_pattern(custom_pattern)

    print("\nCustomized preset:")
    print(f"  Memory Limit: {config.runtime_config['mem_limit']}")
    print(f"  User: {config.runtime_config['user']}")
    print(f"  Extra Patterns: +1 (pandas restriction)")


def demo_backend_differences() -> None:
    """Demonstrate differences between backends."""
    print("\n" + "=" * 80)
    print("Backend-Specific Configurations")
    print("=" * 80)

    preset_name = "production"

    for backend in ["docker", "podman", "kubernetes"]:
        print(f"\n{backend.upper()} Backend:")
        print("-" * 40)

        config = get_security_preset(preset_name, "python", backend)

        print(f"  Security Policy: {config.security_policy.severity_threshold.name}")

        if backend == "kubernetes":
            print("  Runtime Config: Not used (use pod_manifest instead)")
            print("  Pod Manifest: User must provide custom manifest")
        else:
            print(f"  Runtime Config: {len(config.runtime_config or {})} parameters")
            print(f"    Memory: {config.runtime_config.get('mem_limit', 'N/A')}")
            print(f"    Network: {config.runtime_config.get('network_mode', 'N/A')}")


if __name__ == "__main__":
    """Run all demonstrations."""
    print("\nSecurity Presets Demonstration")
    print("=" * 80)
    print("\nThis demo shows how to use predefined security presets that combine")
    print("SecurityPolicy with backend-specific runtime configurations.")

    demo_basic_usage()
    demo_all_presets()
    demo_preset_comparison()
    demo_custom_modifications()
    demo_backend_differences()

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nTo use in your code:")
    print("  from llm_sandbox import SandboxSession, get_security_preset")
    print("  config = get_security_preset('production', 'python', 'docker')")
    print("  session = SandboxSession(")
    print("      lang='python',")
    print("      security_policy=config.security_policy,")
    print("      runtime_configs=config.runtime_config")
    print("  )")
