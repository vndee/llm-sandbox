"""Interactive sandbox session implementation."""

from __future__ import annotations

import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_sandbox.const import SandboxBackend, SupportedLanguage
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.docker import SandboxDockerSession
from llm_sandbox.exceptions import ContainerError, LanguageNotSupportedError, NotOpenSessionError, SandboxTimeoutError

from .const import StrEnum
from .exceptions import UnsupportedBackendError

PYTHON_EXECUTABLE = "/tmp/venv/bin/python"
PYTHON_PIP_EXECUTABLE = "/tmp/venv/bin/pip"
PIP_CACHE_DIR = "/tmp/pip_cache"


class KernelType(StrEnum):
    """Supported kernel types for interactive sessions."""

    STANDARD = "standard"
    IPYTHON = "ipython"


@dataclass(slots=True)
class InteractiveSettings:
    """Configuration values for interactive execution."""

    kernel_type: KernelType = KernelType.IPYTHON
    max_memory: str | None = None
    auto_cleanup: bool = True
    history_size: int = 1000
    enable_magic: bool = True
    timeout: float | None = 300.0

    def __post_init__(self) -> None:
        if self.history_size < 0:
            msg = "history_size must be non-negative"
            raise ValueError(msg)


class InteractiveSandboxSession(SandboxDockerSession):
    """Interactive sandbox session that preserves interpreter state across runs."""

    def __init__(
        self,
        *,
        backend: SandboxBackend = SandboxBackend.DOCKER,
        lang: str | SupportedLanguage = SupportedLanguage.PYTHON,
        kernel_type: KernelType | str = KernelType.IPYTHON,
        max_memory: str | None = "1GB",
        auto_cleanup: bool = True,
        history_size: int = 1000,
        enable_magic: bool = True,
        timeout: float | None = 300.0,
        runtime_configs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the interactive session."""
        if backend != SandboxBackend.DOCKER:
            raise UnsupportedBackendError(backend=backend)

        lang_value = str(lang)
        if lang_value.lower() != SupportedLanguage.PYTHON:
            raise LanguageNotSupportedError(lang_value)

        kernel_enum = KernelType(kernel_type)
        runtime_configs = runtime_configs.copy() if runtime_configs else {}
        if max_memory:
            runtime_configs.setdefault("mem_limit", max_memory)

        if timeout is not None and "execution_timeout" not in kwargs:
            kwargs["execution_timeout"] = timeout

        settings = InteractiveSettings(
            kernel_type=kernel_enum,
            max_memory=max_memory,
            auto_cleanup=auto_cleanup,
            history_size=history_size,
            enable_magic=enable_magic,
            timeout=timeout,
        )

        kwargs.setdefault("lang", SupportedLanguage.PYTHON)
        super().__init__(runtime_configs=runtime_configs, **kwargs)

        self.settings = settings
        self._context_path = f"{self.config.workdir.rstrip('/')}/.interactive_context.pkl"
        self._history_path = f"{self.config.workdir.rstrip('/')}/.interactive_history.jsonl"
        self._runtime_script_path = f"{self.config.workdir.rstrip('/')}/.interactive_runner.py"
        self._runtime_ready = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def open(self) -> None:
        """Open interactive session and prepare runtime assets."""
        super().open()
        self._ensure_interactive_dependencies()
        self._ensure_storage_paths()
        self._upload_runtime_script()

    def close(self) -> None:
        """Close the interactive session."""
        super().close()
        self._runtime_ready = False

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def run(self, code: str, libraries: list[str] | None = None, timeout: float | None = None) -> ConsoleOutput:
        """Execute code in the persistent interpreter context."""
        if not self.container or not self.is_open:
            raise NotOpenSessionError

        self._check_session_timeout()
        actual_timeout = timeout or self.settings.timeout or self.config.get_execution_timeout()

        def _execute() -> ConsoleOutput:
            """Inner execution block."""
            self.install(libraries)
            temp_code_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".py",
                    mode="w",
                    encoding="utf-8",
                ) as temp_code:
                    temp_code.write(code)
                    temp_code_path = temp_code.name

                code_destination = f"{self.config.workdir.rstrip('/')}/{Path(temp_code_path).name}"
                self.copy_to_runtime(temp_code_path, code_destination)

                command = self._build_execution_command(code_destination)
                result = self.execute_command(command, workdir=self.config.workdir)
                self.execute_command(f"rm -f {code_destination}", workdir=self.config.workdir)
                return result
            finally:
                if temp_code_path:
                    Path(temp_code_path).unlink(missing_ok=True)

        try:
            result = self._execute_with_timeout(_execute, timeout=actual_timeout)
            return result
        except SandboxTimeoutError:
            self._handle_timeout()
            raise

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _ensure_interactive_dependencies(self) -> None:
        """Install required Python packages for interactive execution."""
        packages: list[str] = ["cloudpickle"]
        if self.settings.kernel_type == KernelType.IPYTHON:
            packages.append("ipython")

        install_command = (
            f"{PYTHON_PIP_EXECUTABLE} install --quiet --disable-pip-version-check "
            f"--cache-dir {PIP_CACHE_DIR} {' '.join(packages)}"
        )

        result = self.execute_command(install_command)
        if result.exit_code:
            msg = f"Failed to install interactive dependencies: {result.stderr or result.stdout}"
            raise ContainerError(msg)

    def _ensure_storage_paths(self) -> None:
        """Create directories for context and history persistence."""
        paths = {Path(self._context_path).parent, Path(self._history_path).parent}
        commands = [(f"mkdir -p {path.as_posix()}", None) for path in paths]
        self.execute_commands(commands)

    def _upload_runtime_script(self) -> None:
        """Copy interactive runner script into the container."""
        if self._runtime_ready:
            return

        script_content = self._build_runtime_script()
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as script_file:
            script_file.write(script_content)
            temp_path = script_file.name

        try:
            self.copy_to_runtime(temp_path, self._runtime_script_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

        self._runtime_ready = True

    def _build_execution_command(self, code_path: str) -> str:
        """Construct the runtime command to execute user code."""
        arguments = [
            PYTHON_EXECUTABLE,
            self._runtime_script_path,
            "--code-file",
            code_path,
            "--context-file",
            self._context_path,
            "--history-file",
            self._history_path,
            "--history-size",
            str(self.settings.history_size),
            "--kernel-type",
            self.settings.kernel_type.value,
        ]

        if self.settings.auto_cleanup:
            arguments.append("--auto-cleanup")
        if self.settings.enable_magic:
            arguments.append("--enable-magic")

        return " ".join(arguments)

    def _build_runtime_script(self) -> str:
        """Return the Python script that maintains interactive state."""
        return textwrap.dedent(
            """
            import argparse
            import gc
            import json
            import sys
            import traceback
            from pathlib import Path

            try:
                import cloudpickle as pickle  # type: ignore[import-not-found]
            except Exception:  # pragma: no cover - fallback path
                import pickle  # type: ignore[no-redef]

            EXCLUDED_NAMES = {
                "__builtins__",
                "__name__",
                "__file__",
                "__package__",
                "__loader__",
                "__spec__",
            }

            def load_namespace(path: str) -> dict:
                ctx_path = Path(path)
                if not ctx_path.exists():
                    return {}
                with ctx_path.open("rb") as ctx_file:
                    return pickle.load(ctx_file)

            def is_picklable(name: str, value: object) -> bool:
                try:
                    pickle.dumps({name: value})
                except Exception:
                    return False
                return True

            def save_namespace(path: str, namespace: dict) -> None:
                ctx_path = Path(path)
                ctx_path.parent.mkdir(parents=True, exist_ok=True)
                filtered: dict = {}
                for key, value in namespace.items():
                    if key in EXCLUDED_NAMES:
                        continue
                    if is_picklable(key, value):
                        filtered[key] = value
                with ctx_path.open("wb") as ctx_file:
                    pickle.dump(filtered, ctx_file)

            def append_history(path: str, size: int, code: str, success: bool) -> None:
                if size <= 0:
                    return
                history_path = Path(path)
                history_path.parent.mkdir(parents=True, exist_ok=True)
                entries: list[dict] = []
                if history_path.exists():
                    with history_path.open("r", encoding="utf-8") as history_file:
                        entries = [json.loads(line) for line in history_file if line.strip()]
                entries.append({"code": code, "success": success})
                entries = entries[-size:]
                with history_path.open("w", encoding="utf-8") as history_file:
                    for entry in entries:
                        history_file.write(json.dumps(entry))
                        history_file.write("\\n")

            def validate_magic(code: str) -> bool:
                stripped = code.lstrip()
                if not stripped:
                    return True
                return not stripped.startswith("%")

            def run_code(args: argparse.Namespace) -> int:
                with open(args.code_file, "r", encoding="utf-8") as code_file:
                    code = code_file.read()

                if not args.enable_magic and not validate_magic(code):
                    print("Magic commands are disabled for this session.", file=sys.stderr)
                    return 1

                namespace = load_namespace(args.context_file)
                namespace["__builtins__"] = __builtins__
                namespace.setdefault("__name__", "__main__")
                namespace.setdefault("__package__", None)

                success = True

                try:
                    if args.kernel_type == "ipython":
                        from IPython.core.interactiveshell import InteractiveShell  # type: ignore[import-not-found]

                        shell = InteractiveShell.instance()
                        shell.cache_size = max(shell.cache_size, args.history_size or 0)
                        shell.user_ns = namespace
                        shell.user_global_ns = namespace
                        result = shell.run_cell(code, store_history=False, silent=False)
                        success = result.success
                    else:
                        exec(compile(code, args.code_file, "exec"), namespace)
                except SystemExit:
                    raise
                except Exception as exc:  # noqa: BLE001
                    success = False
                    traceback.print_exc()
                finally:
                    namespace.pop("__builtins__", None)

                save_namespace(args.context_file, namespace)
                append_history(args.history_file, args.history_size, code, success)

                if args.auto_cleanup:
                    gc.collect()

                return 0 if success else 1

            def main() -> None:
                parser = argparse.ArgumentParser(description="Interactive sandbox runner")
                parser.add_argument("--code-file", required=True)
                parser.add_argument("--context-file", required=True)
                parser.add_argument("--history-file", required=True)
                parser.add_argument("--history-size", type=int, default=0)
                parser.add_argument("--kernel-type", choices=("standard", "ipython"), default="standard")
                parser.add_argument("--auto-cleanup", action="store_true")
                parser.add_argument("--enable-magic", action="store_true")
                args = parser.parse_args()

                exit_code = run_code(args)
                sys.exit(exit_code)

            if __name__ == "__main__":
                main()
            """
        )
