"""Integration test to validate security presets work with actual parameters.

This test validates that the runtime_config parameters from security presets
are compatible with docker-py's containers.create() method.
"""

import pytest

from llm_sandbox.security_presets import get_security_preset, list_available_presets


def test_runtime_config_parameter_names() -> None:
    """Verify all runtime config parameters use correct docker-py parameter names."""
    # Known correct parameter names from docker-py documentation
    valid_params = {
        # Resource limits
        "mem_limit",
        "mem_reservation",
        "memswap_limit",
        "mem_swappiness",
        "cpu_period",
        "cpu_quota",
        "cpu_shares",
        "cpuset_cpus",
        "cpuset_mems",
        "nano_cpus",
        # Security
        "user",
        "privileged",
        "cap_add",
        "cap_drop",
        "security_opt",
        "read_only",
        # Networking
        "network_mode",
        "network_disabled",
        "dns",
        "dns_search",
        "ports",
        "hostname",
        # Storage
        "tmpfs",
        "volumes",
        "volumes_from",
        "mounts",
        "working_dir",
        # Other
        "environment",
        "labels",
        "tty",
        "detach",
        "stdin_open",
        "auto_remove",
        "shm_size",
    }

    # Check all presets for Docker/Podman
    for preset_name in list_available_presets():
        config = get_security_preset(preset_name, "python", "docker")

        if config.runtime_config:
            for param_name in config.runtime_config.keys():
                assert param_name in valid_params, (
                    f"Preset '{preset_name}' uses invalid parameter '{param_name}'. "
                    f"This parameter is not supported by docker-py containers.create(). "
                    f"Valid parameters include: {', '.join(sorted(valid_params))}"
                )


def test_no_cpu_count_parameter() -> None:
    """Verify that cpu_count parameter is not used (it's Windows-only)."""
    for preset_name in list_available_presets():
        config = get_security_preset(preset_name, "python", "docker")

        if config.runtime_config:
            assert "cpu_count" not in config.runtime_config, (
                f"Preset '{preset_name}' uses 'cpu_count' parameter which is Windows-only. "
                "Use 'cpu_period' and 'cpu_quota' instead for cross-platform CPU limits."
            )


def test_no_memory_parameter() -> None:
    """Verify that 'memory' parameter is not used (should be 'mem_limit')."""
    for preset_name in list_available_presets():
        config = get_security_preset(preset_name, "python", "docker")

        if config.runtime_config:
            assert "memory" not in config.runtime_config, (
                f"Preset '{preset_name}' uses 'memory' parameter. "
                "The correct parameter name is 'mem_limit' for docker-py."
            )


def test_mem_limit_format() -> None:
    """Verify mem_limit values use correct format."""
    for preset_name in list_available_presets():
        config = get_security_preset(preset_name, "python", "docker")

        if config.runtime_config and "mem_limit" in config.runtime_config:
            mem_limit = config.runtime_config["mem_limit"]
            assert isinstance(mem_limit, (str, int)), (
                f"Preset '{preset_name}' has invalid mem_limit type: {type(mem_limit)}. "
                "Should be str (e.g., '512m') or int (bytes)."
            )

            if isinstance(mem_limit, str):
                # Check format: should end with unit (m, g, k, etc.)
                assert mem_limit[-1] in "bkmg", (
                    f"Preset '{preset_name}' has invalid mem_limit format: {mem_limit}. "
                    "Should end with unit: b, k, m, or g (e.g., '512m', '1g')."
                )


def test_cpu_quota_and_period_together() -> None:
    """Verify that if cpu_quota is set, cpu_period is also set."""
    for preset_name in list_available_presets():
        config = get_security_preset(preset_name, "python", "docker")

        if config.runtime_config:
            has_quota = "cpu_quota" in config.runtime_config
            has_period = "cpu_period" in config.runtime_config

            # If one is set, both should be set
            if has_quota or has_period:
                assert has_quota and has_period, (
                    f"Preset '{preset_name}' has incomplete CPU configuration. "
                    "Both 'cpu_quota' and 'cpu_period' should be set together."
                )


def test_network_mode_values() -> None:
    """Verify network_mode uses valid values."""
    valid_modes = {"bridge", "none", "host", "container"}

    for preset_name in list_available_presets():
        config = get_security_preset(preset_name, "python", "docker")

        if config.runtime_config and "network_mode" in config.runtime_config:
            network_mode = config.runtime_config["network_mode"]

            # Check if it's a container reference or a valid mode
            is_container_ref = network_mode.startswith("container:")
            is_valid_mode = network_mode in valid_modes

            assert is_container_ref or is_valid_mode, (
                f"Preset '{preset_name}' has invalid network_mode: {network_mode}. "
                f"Valid values are: {', '.join(valid_modes)} or 'container:<name|id>'."
            )


def test_tmpfs_format() -> None:
    """Verify tmpfs uses correct format."""
    for preset_name in list_available_presets():
        config = get_security_preset(preset_name, "python", "docker")

        if config.runtime_config and "tmpfs" in config.runtime_config:
            tmpfs = config.runtime_config["tmpfs"]
            assert isinstance(tmpfs, dict), (
                f"Preset '{preset_name}' has invalid tmpfs type: {type(tmpfs)}. "
                "Should be dict mapping paths to options."
            )

            for path, options in tmpfs.items():
                assert path.startswith("/"), (
                    f"Preset '{preset_name}' has invalid tmpfs path: {path}. "
                    "Paths should be absolute."
                )


def test_user_format() -> None:
    """Verify user parameter uses correct format."""
    for preset_name in list_available_presets():
        config = get_security_preset(preset_name, "python", "docker")

        if config.runtime_config and "user" in config.runtime_config:
            user = config.runtime_config["user"]
            assert isinstance(user, (str, int)), (
                f"Preset '{preset_name}' has invalid user type: {type(user)}. "
                "Should be str (e.g., 'nobody:nogroup', '1000:1000') or int (UID)."
            )


def test_cap_drop_format() -> None:
    """Verify cap_drop uses correct format."""
    for preset_name in list_available_presets():
        config = get_security_preset(preset_name, "python", "docker")

        if config.runtime_config and "cap_drop" in config.runtime_config:
            cap_drop = config.runtime_config["cap_drop"]
            assert isinstance(cap_drop, list), (
                f"Preset '{preset_name}' has invalid cap_drop type: {type(cap_drop)}. "
                "Should be list of capability names."
            )

            for cap in cap_drop:
                assert isinstance(cap, str), (
                    f"Preset '{preset_name}' has invalid capability in cap_drop: {cap}. "
                    "Each capability should be a string."
                )


def test_security_opt_format() -> None:
    """Verify security_opt uses correct format."""
    for preset_name in list_available_presets():
        config = get_security_preset(preset_name, "python", "docker")

        if config.runtime_config and "security_opt" in config.runtime_config:
            security_opt = config.runtime_config["security_opt"]
            assert isinstance(security_opt, list), (
                f"Preset '{preset_name}' has invalid security_opt type: {type(security_opt)}. "
                "Should be list of security options."
            )

            for opt in security_opt:
                assert isinstance(opt, str), (
                    f"Preset '{preset_name}' has invalid option in security_opt: {opt}. "
                    "Each option should be a string."
                )


if __name__ == "__main__":
    """Run validation tests."""
    pytest.main([__file__, "-v"])
