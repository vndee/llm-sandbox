"""Main session module for LLM Sandbox."""

from importlib.util import find_spec
from types import TracebackType
from typing import Any, TypeVar, cast

from .base import ExecutionResult, Session
from .const import SandboxBackend, SupportedLanguage
from .exceptions import DependencyError
from .factory import SessionFactory

T = TypeVar("T", bound=Session)


def _check_dependency(backend: SandboxBackend) -> None:
    """Check if required dependency is installed for the given backend."""
    if backend in {SandboxBackend.DOCKER, SandboxBackend.MICROMAMBA} and not find_spec(
        "docker"
    ):
        msg = (
            "Docker backend requires 'docker' package. "
            "Install it with: pip install llm-sandbox[docker]"
        )
        raise DependencyError(msg)
    if backend == SandboxBackend.KUBERNETES and not find_spec("kubernetes"):
        msg = (
            "Kubernetes backend requires 'kubernetes' package. "
            "Install it with: pip install llm-sandbox[k8s]"
        )
        raise DependencyError(msg)
    if backend == SandboxBackend.PODMAN and not find_spec("podman"):
        msg = (
            "Podman backend requires 'podman' package. "
            "Install it with: pip install llm-sandbox[podman]"
        )
        raise DependencyError(msg)


class ContextManagerMixin:
    """Mixin to add context manager support to session instances."""

    def __enter__(self) -> "Session":
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager."""
        if hasattr(self, "close"):
            self.close()


class SandboxSession:
    """Factory class for creating sandbox sessions with context manager support."""

    def __new__(
        cls,
        backend: SandboxBackend = SandboxBackend.DOCKER,
        image: str | None = None,
        dockerfile: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        *,
        keep_template: bool = False,
        commit_container: bool = False,
        verbose: bool = False,
        runtime_configs: dict | None = None,
        workdir: str | None = "/sandbox",
        **kwargs: Any,  # noqa: ANN401
    ) -> "SandboxSession":
        """Create a new sandbox session.

        Args:
            backend: Docker, Kubernetes or Podman backend
            image: Container image to use
            dockerfile: Path to Dockerfile
            lang: Programming language
            keep_template: Whether to keep the container template
            commit_container: Whether to commit container changes
            verbose: Enable verbose logging
            runtime_configs: Additional runtime configurations, check the specific
                            backend for more details.
            workdir: Working directory inside the container. Defaults to "/sandbox".
                        Use "/tmp/sandbox" when running as non-root user.
            **kwargs: Additional keyword arguments.

        Returns:
            A new sandbox session of the appropriate type
            (Docker, Kubernetes, Podman, or Micromamba)

        Raises:
            DependencyError: If the required dependency
            for the chosen backend is not installed

        """
        # Check if required dependency is installed
        _check_dependency(backend)

        factory = SessionFactory()

        session = factory.create_session(
            backend=backend,
            image=image,
            dockerfile=dockerfile,
            lang=lang,
            keep_template=keep_template,
            commit_container=commit_container,
            verbose=verbose,
            runtime_configs=runtime_configs,
            workdir=workdir,
            **kwargs,
        )

        # Create a new class that inherits from both the session class and our mixin
        session.__class__ = type(
            "SandboxSession",
            (session.__class__, ContextManagerMixin, SandboxSession),
            {},
        )

        return cast("SandboxSession", session)


class ArtifactSandboxSession(Session):
    """Sandbox session for artifact generation."""

    def __init__(self, *args: Any, **kwargs: dict[str, Any]) -> None:  # noqa: ANN401
        """Initialize the artifact sandbox session."""
        super().__init__(*args, **kwargs)

        self.enable_plotting = kwargs.get("enable_plotting", True)
        self.enable_file_output = kwargs.get("enable_file_output", True)

    def run(self, code: str, libraries: list | None = None) -> ExecutionResult:
        """Run the code in the sandbox session."""
        if self.enable_plotting:
            injected_code = self.language_handler.inject_plot_detection_code(code)
        else:
            injected_code = code

        result = super().run(injected_code, libraries)

        plots, files = [], []

        if self.enable_plotting:
            plots = self.language_handler.extract_plots(
                self.container,
                "/tmp/sandbox_plots",
            )

        if self.enable_file_output:
            files = self.language_handler.extract_files(
                self.container,
                "/tmp/sandbox_output",
            )

        return ExecutionResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            plots=plots,
            files=files,
        )
