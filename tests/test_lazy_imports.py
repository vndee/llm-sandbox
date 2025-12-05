"""Test that llm_sandbox can be imported without optional backend dependencies."""

import subprocess
import sys


def test_import_without_kubernetes_dependency() -> None:
    """Test that llm_sandbox can be imported when kubernetes package is not installed.

    This test verifies the fix for the issue where installing llm-sandbox[docker]
    would fail to import due to missing kubernetes dependency.
    """
    code = """
import sys

# Block kubernetes import by adding a fake finder
class BlockKubernetesImport:
    def find_module(self, name, path=None):
        if name == 'kubernetes' or name.startswith('kubernetes.'):
            return self
        return None
    def load_module(self, name):
        raise ImportError(f"No module named '{name}' (blocked for testing)")

sys.meta_path.insert(0, BlockKubernetesImport())

# Now try to import llm_sandbox - it should work
try:
    import llm_sandbox
    print("SUCCESS: llm_sandbox imported successfully without kubernetes")
    print(f"KernelType: {llm_sandbox.KernelType.IPYTHON}")
except ImportError as e:
    print(f"FAILURE: {e}")
    sys.exit(1)
"""
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)  # noqa: S603
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "SUCCESS" in result.stdout


def test_import_without_podman_dependency() -> None:
    """Test that llm_sandbox can be imported when podman package is not installed.

    This test verifies the fix for the issue where installing llm-sandbox[docker]
    would fail to import due to missing podman dependency.
    """
    code = """
import sys

# Block podman import by adding a fake finder
class BlockPodmanImport:
    def find_module(self, name, path=None):
        if name == 'podman' or name.startswith('podman.'):
            return self
        return None
    def load_module(self, name):
        raise ImportError(f"No module named '{name}' (blocked for testing)")

sys.meta_path.insert(0, BlockPodmanImport())

# Now try to import llm_sandbox - it should work
try:
    import llm_sandbox
    print("SUCCESS: llm_sandbox imported successfully without podman")
    print(f"KernelType: {llm_sandbox.KernelType.IPYTHON}")
except ImportError as e:
    print(f"FAILURE: {e}")
    sys.exit(1)
"""
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)  # noqa: S603
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "SUCCESS" in result.stdout


def test_import_without_both_kubernetes_and_podman() -> None:
    """Test that llm_sandbox can be imported when both kubernetes and podman are not installed."""
    code = """
import sys

# Block both kubernetes and podman imports
class BlockOptionalImports:
    def find_module(self, name, path=None):
        blocked = ['kubernetes', 'podman']
        for b in blocked:
            if name == b or name.startswith(f'{b}.'):
                return self
        return None
    def load_module(self, name):
        raise ImportError(f"No module named '{name}' (blocked for testing)")

sys.meta_path.insert(0, BlockOptionalImports())

# Now try to import llm_sandbox - it should work
try:
    import llm_sandbox
    print("SUCCESS: llm_sandbox imported successfully without kubernetes and podman")
    print(f"KernelType: {llm_sandbox.KernelType.IPYTHON}")
except ImportError as e:
    print(f"FAILURE: {e}")
    sys.exit(1)
"""
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)  # noqa: S603
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "SUCCESS" in result.stdout
