# ruff: noqa: PLR2004
"""Tests for llm_sandbox.const module."""

import pytest

from llm_sandbox.const import DefaultImage, SandboxBackend, StrEnum, SupportedLanguage


class TestStrEnum:
    """Test StrEnum functionality."""

    def test_str_enum_creation(self) -> None:
        """Test basic StrEnum creation."""
        # Test normal string values
        assert SandboxBackend.DOCKER == "docker"
        assert SandboxBackend.KUBERNETES == "kubernetes"
        assert SandboxBackend.PODMAN == "podman"
        assert SandboxBackend.MICROMAMBA == "micromamba"

    def test_str_enum_invalid_type(self) -> None:
        """Test StrEnum with invalid type."""
        # Test line 25-26: TypeError for non-string values
        with pytest.raises(TypeError, match="StrEnum values must be strings"):

            class InvalidEnum(StrEnum):  # NOSONAR
                NUMBER = 123

    def test_str_enum_string_methods(self) -> None:
        """Test StrEnum string methods."""
        backend = SandboxBackend.DOCKER

        # Test __str__ method (line 39)
        assert str(backend) == "docker"

        # Test __repr__ method
        repr_str = repr(backend)
        assert "SandboxBackend.DOCKER" in repr_str
        assert "'docker'" in repr_str

    def test_str_enum_missing_case_insensitive(self) -> None:
        """Test case-insensitive lookup for StrEnum."""
        # Test _missing_ method (lines 45-51)
        # This should work for case-insensitive lookup
        assert SandboxBackend("DOCKER") == SandboxBackend.DOCKER
        assert SandboxBackend("Docker") == SandboxBackend.DOCKER
        assert SandboxBackend("docker") == SandboxBackend.DOCKER

    def test_str_enum_missing_invalid_value(self) -> None:
        """Test StrEnum with invalid value."""
        # Test ValueError for invalid values (lines 45-51)
        with pytest.raises(ValueError, match="'invalid' is not a valid SandboxBackend"):
            SandboxBackend("invalid")

        # Test that non-string values also raise ValueError via _missing_
        with pytest.raises(ValueError, match="123 is not a valid SandboxBackend"):
            SandboxBackend._missing_(123)

    def test_str_enum_comparison(self) -> None:
        """Test StrEnum comparison with strings."""
        assert SandboxBackend.DOCKER == "docker"
        assert SandboxBackend.KUBERNETES == "kubernetes"
        assert SandboxBackend.PODMAN == "podman"
        assert SandboxBackend.MICROMAMBA == "micromamba"

    def test_str_enum_in_collections(self) -> None:
        """Test StrEnum usage in collections."""
        backends = [SandboxBackend.DOCKER, SandboxBackend.KUBERNETES]
        assert "docker" in backends
        assert "kubernetes" in backends
        assert "invalid" not in backends


class TestSandboxBackend:
    """Test SandboxBackend enum."""

    def test_all_backends(self) -> None:
        """Test all backend values."""
        assert SandboxBackend.DOCKER == "docker"
        assert SandboxBackend.KUBERNETES == "kubernetes"
        assert SandboxBackend.PODMAN == "podman"
        assert SandboxBackend.MICROMAMBA == "micromamba"

    def test_backend_iteration(self) -> None:
        """Test iterating over backends."""
        backends = list(SandboxBackend)
        assert len(backends) == 4
        assert SandboxBackend.DOCKER in backends
        assert SandboxBackend.KUBERNETES in backends
        assert SandboxBackend.PODMAN in backends
        assert SandboxBackend.MICROMAMBA in backends

    def test_backend_case_insensitive(self) -> None:
        """Test case-insensitive backend lookup."""
        assert SandboxBackend("DOCKER") == SandboxBackend.DOCKER
        assert SandboxBackend("Kubernetes") == SandboxBackend.KUBERNETES
        assert SandboxBackend("PODMAN") == SandboxBackend.PODMAN
        assert SandboxBackend("MicroMamba") == SandboxBackend.MICROMAMBA


class TestSupportedLanguage:
    """Test SupportedLanguage enum."""

    def test_all_languages(self) -> None:
        """Test all language values."""
        assert SupportedLanguage.PYTHON == "python"
        assert SupportedLanguage.JAVA == "java"
        assert SupportedLanguage.JAVASCRIPT == "javascript"
        assert SupportedLanguage.CPP == "cpp"
        assert SupportedLanguage.GO == "go"
        assert SupportedLanguage.RUBY == "ruby"

    def test_language_iteration(self) -> None:
        """Test iterating over languages."""
        languages = list(SupportedLanguage)
        assert len(languages) == 7
        assert SupportedLanguage.PYTHON in languages
        assert SupportedLanguage.JAVA in languages
        assert SupportedLanguage.JAVASCRIPT in languages
        assert SupportedLanguage.CPP in languages
        assert SupportedLanguage.GO in languages
        assert SupportedLanguage.RUBY in languages
        assert SupportedLanguage.R in languages

    def test_language_case_insensitive(self) -> None:
        """Test case-insensitive language lookup."""
        assert SupportedLanguage("PYTHON") == SupportedLanguage.PYTHON
        assert SupportedLanguage("Java") == SupportedLanguage.JAVA
        assert SupportedLanguage("JavaScript") == SupportedLanguage.JAVASCRIPT
        assert SupportedLanguage("CPP") == SupportedLanguage.CPP
        assert SupportedLanguage("Go") == SupportedLanguage.GO
        assert SupportedLanguage("Ruby") == SupportedLanguage.RUBY


class TestDefaultImage:
    """Test DefaultImage dataclass."""

    def test_all_default_images(self) -> None:
        """Test all default image values."""
        assert DefaultImage.PYTHON == "ghcr.io/vndee/sandbox-python-311-bullseye"
        assert DefaultImage.JAVA == "ghcr.io/vndee/sandbox-java-11-bullseye"
        assert DefaultImage.JAVASCRIPT == "ghcr.io/vndee/sandbox-node-22-bullseye"
        assert DefaultImage.CPP == "ghcr.io/vndee/sandbox-cpp-11-bullseye"
        assert DefaultImage.GO == "ghcr.io/vndee/sandbox-go-123-bullseye"
        assert DefaultImage.RUBY == "ghcr.io/vndee/sandbox-ruby-302-bullseye"

    def test_default_image_access(self) -> None:
        """Test accessing default images via attribute."""
        image_instance = DefaultImage()

        # Test accessing via instance
        assert hasattr(image_instance, "PYTHON")
        assert hasattr(image_instance, "JAVA")
        assert hasattr(image_instance, "JAVASCRIPT")
        assert hasattr(image_instance, "CPP")
        assert hasattr(image_instance, "GO")
        assert hasattr(image_instance, "RUBY")

    def test_default_image_class_attributes(self) -> None:
        """Test accessing default images as class attributes."""
        # Test accessing via class
        assert hasattr(DefaultImage, "PYTHON")
        assert hasattr(DefaultImage, "JAVA")
        assert hasattr(DefaultImage, "JAVASCRIPT")
        assert hasattr(DefaultImage, "CPP")
        assert hasattr(DefaultImage, "GO")
        assert hasattr(DefaultImage, "RUBY")

    def test_image_names_format(self) -> None:
        """Test that all image names follow expected format."""
        images = [
            DefaultImage.PYTHON,
            DefaultImage.JAVA,
            DefaultImage.JAVASCRIPT,
            DefaultImage.CPP,
            DefaultImage.GO,
            DefaultImage.RUBY,
        ]

        for image in images:
            assert image.startswith("ghcr.io/vndee/sandbox-")
            assert image.endswith("-bullseye")


class TestEnumIntegration:
    """Test integration between different enums."""

    def test_enum_string_compatibility(self) -> None:
        """Test that enums work well with string operations."""
        # Test concatenation
        backend_str = f"Using {SandboxBackend.DOCKER} backend"
        assert backend_str == "Using docker backend"

        language_str = f"Running {SupportedLanguage.PYTHON} code"
        assert language_str == "Running python code"

    def test_enum_dict_keys(self) -> None:
        """Test using enums as dictionary keys."""
        backend_config = {
            SandboxBackend.DOCKER: {"socket": "/var/run/docker.sock"},
            SandboxBackend.KUBERNETES: {"namespace": "default"},
        }

        assert backend_config[SandboxBackend.DOCKER]["socket"] == "/var/run/docker.sock"
        assert backend_config[SandboxBackend.KUBERNETES]["namespace"] == "default"

    def test_enum_set_operations(self) -> None:
        """Test using enums in set operations."""
        supported_backends = {SandboxBackend.DOCKER, SandboxBackend.KUBERNETES}
        assert SandboxBackend.DOCKER in supported_backends
        assert SandboxBackend.PODMAN not in supported_backends

    def test_language_image_mapping(self) -> None:
        """Test mapping between languages and default images."""
        # Create a mapping using both enums
        language_to_image = {
            SupportedLanguage.PYTHON: DefaultImage.PYTHON,
            SupportedLanguage.JAVA: DefaultImage.JAVA,
            SupportedLanguage.JAVASCRIPT: DefaultImage.JAVASCRIPT,
            SupportedLanguage.CPP: DefaultImage.CPP,
            SupportedLanguage.GO: DefaultImage.GO,
            SupportedLanguage.RUBY: DefaultImage.RUBY,
        }

        assert language_to_image[SupportedLanguage.PYTHON] == DefaultImage.PYTHON
        assert language_to_image[SupportedLanguage.JAVA] == DefaultImage.JAVA
        assert language_to_image[SupportedLanguage.JAVASCRIPT] == DefaultImage.JAVASCRIPT


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_enum_membership(self) -> None:
        """Test membership testing."""
        assert "docker" in [member.value for member in SandboxBackend]
        assert "invalid" not in [member.value for member in SandboxBackend]

    def test_enum_equality(self) -> None:
        """Test equality comparisons."""
        assert SandboxBackend.DOCKER != SandboxBackend.KUBERNETES
        assert SandboxBackend.DOCKER == "docker"
        assert SandboxBackend.DOCKER != "kubernetes"
