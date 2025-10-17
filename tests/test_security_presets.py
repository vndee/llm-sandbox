"""Tests for security presets functionality."""

import pytest

from llm_sandbox.security import SecurityIssueSeverity, SecurityPolicy
from llm_sandbox.security_presets import (
    SecurityConfiguration,
    get_security_preset,
    list_available_presets,
    list_supported_backends,
    list_supported_languages,
)


class TestSecurityConfiguration:
    """Test SecurityConfiguration dataclass."""

    def test_security_configuration_with_runtime_config(self) -> None:
        """Test SecurityConfiguration with runtime config."""
        policy = SecurityPolicy(severity_threshold=SecurityIssueSeverity.HIGH)
        runtime_config = {"mem_limit": "512m"}

        config = SecurityConfiguration(
            security_policy=policy,
            runtime_config=runtime_config,
        )

        assert config.security_policy == policy
        assert config.runtime_config == runtime_config
        assert config.pod_manifest is None

    def test_security_configuration_with_pod_manifest(self) -> None:
        """Test SecurityConfiguration with pod manifest."""
        policy = SecurityPolicy(severity_threshold=SecurityIssueSeverity.HIGH)
        pod_manifest = {"apiVersion": "v1", "kind": "Pod"}

        config = SecurityConfiguration(
            security_policy=policy,
            pod_manifest=pod_manifest,
        )

        assert config.security_policy == policy
        assert config.runtime_config is None
        assert config.pod_manifest == pod_manifest


class TestGetSecurityPreset:
    """Test get_security_preset function."""

    def test_development_preset_docker(self) -> None:
        """Test development preset for Docker."""
        config = get_security_preset("development", "python", "docker")

        assert isinstance(config, SecurityConfiguration)
        assert isinstance(config.security_policy, SecurityPolicy)
        assert config.runtime_config is not None
        assert config.pod_manifest is None

        # Check runtime config has expected keys
        assert "mem_limit" in config.runtime_config
        assert "cpu_period" in config.runtime_config
        assert "cpu_quota" in config.runtime_config
        assert "network_mode" in config.runtime_config
        assert config.runtime_config["network_mode"] == "bridge"  # Development allows network

    def test_production_preset_docker(self) -> None:
        """Test production preset for Docker."""
        config = get_security_preset("production", "python", "docker")

        assert isinstance(config, SecurityConfiguration)
        assert config.runtime_config is not None

        # Check production settings
        assert config.runtime_config["network_mode"] == "none"  # No network in production
        assert config.runtime_config["read_only"] is True
        assert "user" in config.runtime_config
        assert config.runtime_config["user"] == "nobody:nogroup"

    def test_strict_preset_docker(self) -> None:
        """Test strict preset for Docker."""
        config = get_security_preset("strict", "python", "docker")

        assert isinstance(config, SecurityConfiguration)
        assert config.runtime_config is not None

        # Check strict settings
        assert config.runtime_config["network_mode"] == "none"
        assert config.runtime_config["read_only"] is True
        assert "cap_drop" in config.runtime_config
        assert config.runtime_config["cap_drop"] == ["ALL"]
        assert "security_opt" in config.runtime_config
        assert "no-new-privileges:true" in config.runtime_config["security_opt"]

    def test_educational_preset_docker(self) -> None:
        """Test educational preset for Docker."""
        config = get_security_preset("educational", "python", "docker")

        assert isinstance(config, SecurityConfiguration)
        assert config.runtime_config is not None

        # Check educational settings
        assert config.runtime_config["network_mode"] == "bridge"  # Allow network for learning
        assert config.runtime_config["read_only"] is True
        assert config.runtime_config["user"] == "1000:1000"

    def test_preset_podman_backend(self) -> None:
        """Test preset with Podman backend."""
        config = get_security_preset("production", "python", "podman")

        assert isinstance(config, SecurityConfiguration)
        assert config.runtime_config is not None
        # Podman uses same runtime config structure as Docker

    def test_preset_kubernetes_backend(self) -> None:
        """Test preset with Kubernetes backend."""
        config = get_security_preset("production", "python", "kubernetes")

        assert isinstance(config, SecurityConfiguration)
        assert isinstance(config.security_policy, SecurityPolicy)
        # Kubernetes doesn't use runtime_config
        assert config.runtime_config is None
        assert config.pod_manifest is None

    def test_invalid_preset_name(self) -> None:
        """Test with invalid preset name."""
        with pytest.raises(ValueError, match="Unknown preset 'invalid'"):
            get_security_preset("invalid", "python", "docker")

    def test_invalid_language(self) -> None:
        """Test with unsupported language."""
        with pytest.raises(ValueError, match="Unsupported language 'java'"):
            get_security_preset("production", "java", "docker")

    def test_invalid_backend(self) -> None:
        """Test with unsupported backend."""
        with pytest.raises(ValueError, match="Unsupported backend 'invalid'"):
            get_security_preset("production", "python", "invalid")

    def test_case_insensitive_inputs(self) -> None:
        """Test that inputs are case-insensitive."""
        config1 = get_security_preset("Production", "Python", "Docker")
        config2 = get_security_preset("PRODUCTION", "PYTHON", "DOCKER")
        config3 = get_security_preset("production", "python", "docker")

        # All should succeed and return equivalent configs
        assert config1.security_policy.severity_threshold == config2.security_policy.severity_threshold
        assert config2.security_policy.severity_threshold == config3.security_policy.severity_threshold


class TestSecurityPolicyContent:
    """Test the actual security policy content of presets."""

    def test_development_policy_is_permissive(self) -> None:
        """Test development policy is permissive."""
        config = get_security_preset("development", "python", "docker")
        policy = config.security_policy

        # Development should only block HIGH severity
        assert policy.severity_threshold == SecurityIssueSeverity.HIGH

        # Should have minimal restrictions
        assert policy.patterns is not None
        assert policy.restricted_modules is not None
        assert len(policy.patterns) <= 3
        assert len(policy.restricted_modules) <= 2

    def test_production_policy_is_strict(self) -> None:
        """Test production policy is stricter."""
        config = get_security_preset("production", "python", "docker")
        policy = config.security_policy

        # Production should block MEDIUM and above
        assert policy.severity_threshold == SecurityIssueSeverity.MEDIUM

        # Should have more restrictions than development
        assert policy.patterns is not None
        assert policy.restricted_modules is not None
        assert len(policy.patterns) >= 4
        assert len(policy.restricted_modules) >= 3

    def test_strict_policy_is_most_restrictive(self) -> None:
        """Test strict policy is most restrictive."""
        config = get_security_preset("strict", "python", "docker")
        policy = config.security_policy

        # Strict should block LOW and above (everything above SAFE)
        assert policy.severity_threshold == SecurityIssueSeverity.LOW

        # Should have most restrictions
        assert policy.patterns is not None
        assert policy.restricted_modules is not None
        assert len(policy.patterns) >= 5
        assert len(policy.restricted_modules) >= 7

    def test_educational_policy_balanced(self) -> None:
        """Test educational policy is balanced."""
        config = get_security_preset("educational", "python", "docker")
        policy = config.security_policy

        # Educational should block MEDIUM and above
        assert policy.severity_threshold == SecurityIssueSeverity.MEDIUM

        # Should be between development and strict
        assert policy.patterns is not None
        assert policy.restricted_modules is not None


class TestRuntimeConfigContent:
    """Test the runtime configuration content."""

    def test_all_docker_configs_have_memory_limit(self) -> None:
        """Test all Docker configs specify memory limit."""
        for preset in list_available_presets():
            config = get_security_preset(preset, "python", "docker")
            assert "mem_limit" in config.runtime_config

    def test_all_docker_configs_have_cpu_limits(self) -> None:
        """Test all Docker configs specify CPU limits."""
        for preset in list_available_presets():
            config = get_security_preset(preset, "python", "docker")
            assert "cpu_period" in config.runtime_config
            assert "cpu_quota" in config.runtime_config

    def test_all_docker_configs_have_network_mode(self) -> None:
        """Test all Docker configs specify network mode."""
        for preset in list_available_presets():
            config = get_security_preset(preset, "python", "docker")
            assert "network_mode" in config.runtime_config

    def test_production_and_strict_are_read_only(self) -> None:
        """Test production and strict configs are read-only."""
        for preset in ["production", "strict", "educational"]:
            config = get_security_preset(preset, "python", "docker")
            assert config.runtime_config["read_only"] is True

    def test_production_and_strict_run_as_non_root(self) -> None:
        """Test production and strict run as non-root user."""
        for preset in ["production", "strict", "educational"]:
            config = get_security_preset(preset, "python", "docker")
            assert "user" in config.runtime_config
            assert config.runtime_config["user"] != "root"

    def test_strict_has_capability_drops(self) -> None:
        """Test strict config drops all capabilities."""
        config = get_security_preset("strict", "python", "docker")
        assert "cap_drop" in config.runtime_config
        assert "ALL" in config.runtime_config["cap_drop"]

    def test_strict_has_security_options(self) -> None:
        """Test strict config has security options."""
        config = get_security_preset("strict", "python", "docker")
        assert "security_opt" in config.runtime_config
        assert any("no-new-privileges" in opt for opt in config.runtime_config["security_opt"])

    def test_development_allows_network(self) -> None:
        """Test development allows network access."""
        config = get_security_preset("development", "python", "docker")
        assert config.runtime_config["network_mode"] == "bridge"

    def test_production_and_strict_disable_network(self) -> None:
        """Test production and strict disable network."""
        for preset in ["production", "strict"]:
            config = get_security_preset(preset, "python", "docker")
            assert config.runtime_config["network_mode"] == "none"


class TestListFunctions:
    """Test list utility functions."""

    def test_list_available_presets(self) -> None:
        """Test listing available presets."""
        presets = list_available_presets()

        assert isinstance(presets, list)
        assert len(presets) > 0
        assert "development" in presets
        assert "production" in presets
        assert "strict" in presets
        assert "educational" in presets

    def test_list_supported_languages(self) -> None:
        """Test listing supported languages."""
        languages = list_supported_languages()

        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "python" in languages

    def test_list_supported_backends(self) -> None:
        """Test listing supported backends."""
        backends = list_supported_backends()

        assert isinstance(backends, list)
        assert len(backends) > 0
        assert "docker" in backends
        assert "podman" in backends
        assert "kubernetes" in backends
