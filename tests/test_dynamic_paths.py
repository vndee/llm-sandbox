"""Test dynamic path resolution for language handlers.

This module tests the new session-aware path resolution feature using
RuntimeContext - fixes issue #121 (read-only file system support).
"""

from llm_sandbox.language_handlers.cpp_handler import CppHandler
from llm_sandbox.language_handlers.python_handler import PythonHandler
from llm_sandbox.language_handlers.runtime_context import RuntimeContext


class TestPythonHandlerDynamicPaths:
    """Test Python handler with runtime context."""

    def test_execution_commands_with_context_default_workdir(self) -> None:
        """Test execution commands use runtime context paths with default workdir."""
        handler = PythonHandler()
        context = RuntimeContext(
            workdir="/sandbox",
            python_executable_path="/sandbox/.sandbox-venv/bin/python",
        )

        commands = handler.get_execution_commands("test.py", runtime_context=context)

        assert len(commands) == 1
        assert commands[0] == "/sandbox/.sandbox-venv/bin/python test.py"

    def test_execution_commands_with_context_custom_workdir(self) -> None:
        """Test execution commands use runtime context paths with custom workdir."""
        handler = PythonHandler()
        context = RuntimeContext(
            workdir="/workspace",
            python_executable_path="/workspace/.sandbox-venv/bin/python",
        )

        commands = handler.get_execution_commands("test.py", runtime_context=context)

        assert len(commands) == 1
        assert commands[0] == "/workspace/.sandbox-venv/bin/python test.py"

    def test_execution_commands_without_context(self) -> None:
        """Test execution commands fall back to system python without context."""
        handler = PythonHandler()

        commands = handler.get_execution_commands("test.py")

        assert len(commands) == 1
        assert commands[0] == "python test.py"

    def test_library_installation_with_context_default_workdir(self) -> None:
        """Test library installation uses runtime context paths with default workdir."""
        handler = PythonHandler()
        context = RuntimeContext(
            workdir="/sandbox",
            python_executable_path="/sandbox/.sandbox-venv/bin/python",
            pip_executable_path="/sandbox/.sandbox-venv/bin/pip",
            pip_cache_dir="/sandbox/.sandbox-pip-cache",
        )

        command = handler.get_library_installation_command("numpy", runtime_context=context)

        assert command == "/sandbox/.sandbox-venv/bin/pip install numpy --cache-dir /sandbox/.sandbox-pip-cache"

    def test_library_installation_with_context_custom_workdir(self) -> None:
        """Test library installation uses runtime context paths with custom workdir."""
        handler = PythonHandler()
        context = RuntimeContext(
            workdir="/workspace",
            python_executable_path="/workspace/.sandbox-venv/bin/python",
            pip_executable_path="/workspace/.sandbox-venv/bin/pip",
            pip_cache_dir="/workspace/.sandbox-pip-cache",
        )

        command = handler.get_library_installation_command("pandas", runtime_context=context)

        assert command == "/workspace/.sandbox-venv/bin/pip install pandas --cache-dir /workspace/.sandbox-pip-cache"

    def test_library_installation_without_context(self) -> None:
        """Test library installation falls back to system pip without context."""
        handler = PythonHandler()

        command = handler.get_library_installation_command("numpy")

        assert command == "pip install numpy"

    def test_multiple_workdir_configurations(self) -> None:
        """Test handler works with various workdir configurations."""
        handler = PythonHandler()
        workdirs = ["/sandbox", "/workspace", "/tmp/custom", "/opt/workdir"]

        for workdir in workdirs:
            context = RuntimeContext(
                workdir=workdir,
                python_executable_path=f"{workdir}/.sandbox-venv/bin/python",
            )
            commands = handler.get_execution_commands("test.py", runtime_context=context)

            assert len(commands) == 1
            assert commands[0] == f"{workdir}/.sandbox-venv/bin/python test.py"


class TestCppHandlerDynamicPaths:
    """Test C++ handler with runtime context."""

    def test_execution_commands_with_context_default_workdir(self) -> None:
        """Test execution commands use runtime context paths with default workdir."""
        handler = CppHandler()
        context = RuntimeContext(workdir="/sandbox")

        commands = handler.get_execution_commands("main.cpp", runtime_context=context)

        assert len(commands) == 2
        assert commands[0] == "g++ -std=c++17 -o /sandbox/a.out main.cpp"
        assert commands[1] == "/sandbox/a.out"

    def test_execution_commands_with_context_custom_workdir(self) -> None:
        """Test execution commands use runtime context paths with custom workdir."""
        handler = CppHandler()
        context = RuntimeContext(workdir="/workspace")

        commands = handler.get_execution_commands("main.cpp", runtime_context=context)

        assert len(commands) == 2
        assert commands[0] == "g++ -std=c++17 -o /workspace/a.out main.cpp"
        assert commands[1] == "/workspace/a.out"

    def test_execution_commands_without_context(self) -> None:
        """Test execution commands fall back to /tmp without context."""
        handler = CppHandler()

        commands = handler.get_execution_commands("main.cpp")

        assert len(commands) == 2
        assert commands[0] == "g++ -std=c++17 -o /tmp/a.out main.cpp"
        assert commands[1] == "/tmp/a.out"

    def test_multiple_workdir_configurations(self) -> None:
        """Test handler works with various workdir configurations."""
        handler = CppHandler()
        workdirs = ["/sandbox", "/workspace", "/tmp/custom", "/opt/workdir"]

        for workdir in workdirs:
            context = RuntimeContext(workdir=workdir)
            commands = handler.get_execution_commands("main.cpp", runtime_context=context)

            assert len(commands) == 2
            assert commands[0] == f"g++ -std=c++17 -o {workdir}/a.out main.cpp"
            assert commands[1] == f"{workdir}/a.out"


class TestBackwardsCompatibility:
    """Test backwards compatibility of dynamic path resolution."""

    def test_python_handler_none_context(self) -> None:
        """Test Python handler works when context is explicitly None."""
        handler = PythonHandler()

        commands = handler.get_execution_commands("test.py", runtime_context=None)
        install_cmd = handler.get_library_installation_command("numpy", runtime_context=None)

        assert commands[0] == "python test.py"
        assert install_cmd == "pip install numpy"

    def test_cpp_handler_none_context(self) -> None:
        """Test C++ handler works when context is explicitly None."""
        handler = CppHandler()

        commands = handler.get_execution_commands("main.cpp", runtime_context=None)

        assert commands[0] == "g++ -std=c++17 -o /tmp/a.out main.cpp"
        assert commands[1] == "/tmp/a.out"

    def test_python_handler_old_calling_convention(self) -> None:
        """Test Python handler works with old calling convention (no context arg)."""
        handler = PythonHandler()

        # Call without context parameter (old way)
        commands = handler.get_execution_commands("test.py")
        install_cmd = handler.get_library_installation_command("numpy")

        # Should use system python/pip for backwards compatibility with skip_environment_setup
        assert commands[0] == "python test.py"
        assert install_cmd == "pip install numpy"

    def test_cpp_handler_old_calling_convention(self) -> None:
        """Test C++ handler works with old calling convention (no context arg)."""
        handler = CppHandler()

        # Call without context parameter (old way)
        commands = handler.get_execution_commands("main.cpp")

        # Should use /tmp for backwards compatibility
        assert commands[0] == "g++ -std=c++17 -o /tmp/a.out main.cpp"


class TestReadOnlyFilesystemSupport:
    """Test that dynamic paths support read-only filesystem scenarios."""

    def test_python_writable_workspace(self) -> None:
        """Test Python handler uses writable workspace in read-only environment."""
        handler = PythonHandler()
        # Simulate Kubernetes with emptyDir at /workspace
        context = RuntimeContext(
            workdir="/workspace",
            python_executable_path="/workspace/.sandbox-venv/bin/python",
            pip_executable_path="/workspace/.sandbox-venv/bin/pip",
            pip_cache_dir="/workspace/.sandbox-pip-cache",
        )

        commands = handler.get_execution_commands("test.py", runtime_context=context)
        install_cmd = handler.get_library_installation_command("numpy", runtime_context=context)

        # All paths should be under /workspace (writable emptyDir)
        assert "/workspace/.sandbox-venv/bin/python" in commands[0]
        assert "/workspace/.sandbox-venv/bin/pip" in install_cmd
        assert "/workspace/.sandbox-pip-cache" in install_cmd

    def test_cpp_writable_workspace(self) -> None:
        """Test C++ handler uses writable workspace in read-only environment."""
        handler = CppHandler()
        # Simulate Kubernetes with emptyDir at /workspace
        context = RuntimeContext(workdir="/workspace")

        commands = handler.get_execution_commands("main.cpp", runtime_context=context)

        # Compilation output should be in writable workspace
        assert "/workspace/a.out" in commands[0]
        assert commands[1] == "/workspace/a.out"
