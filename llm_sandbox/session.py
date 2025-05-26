"""Main session module for LLM Sandbox."""

from importlib.util import find_spec
from types import TracebackType
from typing import Any

from .base import ExecutionResult, Session
from .const import SandboxBackend, SupportedLanguage
from .exceptions import DependencyError, UnsupportedBackendError


def _check_dependency(backend: SandboxBackend) -> None:
    """Check if required dependency is installed for the given backend."""
    if backend in {SandboxBackend.DOCKER, SandboxBackend.MICROMAMBA} and not find_spec("docker"):
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


def create_session(
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
    **kwargs: Any,
) -> Session:
    """Create a new sandbox session.

    Args:
    ----
        backend: Docker, Kubernetes or Podman backend
        image: Container image to use
        dockerfile: Path to Dockerfile
        lang: Programming language
        keep_template: Whether to keep the container template
        commit_container: Whether to commit container changes
        verbose: Enable verbose logging
        runtime_configs: Additional runtime configurations
        workdir: Working directory inside the container
        **kwargs: Additional keyword arguments

    Returns:
    -------
        A sandbox session instance

    Raises:
    ------
        DependencyError: If the required dependency for the chosen backend
                        is not installed
        UnsupportedBackendError: If the chosen backend is not supported

    """
    # Check if required dependency is installed
    _check_dependency(backend)

    # Create the appropriate session based on backend
    match backend:
        case SandboxBackend.DOCKER:
            from .docker import SandboxDockerSession

            return SandboxDockerSession(
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
        case SandboxBackend.KUBERNETES:
            from .kubernetes import SandboxKubernetesSession

            return SandboxKubernetesSession(
                image=image,
                lang=lang,
                keep_template=keep_template,
                verbose=verbose,
                runtime_configs=runtime_configs,
                workdir=workdir,
                **kwargs,
            )
        case SandboxBackend.PODMAN:
            from .podman import SandboxPodmanSession

            return SandboxPodmanSession(
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
        case SandboxBackend.MICROMAMBA:
            from .micromamba import MicromambaSession

            return MicromambaSession(
                image=image,
                lang=lang,
                keep_template=keep_template,
                verbose=verbose,
                runtime_configs=runtime_configs,
                workdir=workdir,
                **kwargs,
            )
        case _:
            raise UnsupportedBackendError(backend=backend)


class ArtifactSandboxSession:
    """Sandbox session with artifact extraction capabilities."""

    def __init__(
        self,
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
        enable_plotting: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a new artifact sandbox session.

        Args:
        ----
            backend: Docker, Kubernetes or Podman backend
            image: Container image to use
            dockerfile: Path to Dockerfile
            lang: Programming language
            keep_template: Whether to keep the container template
            commit_container: Whether to commit container changes
            verbose: Enable verbose logging
            runtime_configs: Additional runtime configurations
            workdir: Working directory inside the container
            enable_plotting: Whether to enable plot extraction
            **kwargs: Additional keyword arguments

        Raises:
        ------
            DependencyError: If the required dependency for the chosen backend
                            is not installed
            UnsupportedBackendError: If the chosen backend is not supported

        """
        # Create the base session
        self._session: Session = create_session(
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

        self.enable_plotting = enable_plotting

    def __enter__(self) -> "ArtifactSandboxSession":
        """Enter the context manager."""
        self._session.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager."""
        return self._session.__exit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name: str) -> Any:
        """Delegate any other attributes/methods to the underlying session."""
        return getattr(self._session, name)

    def run(self, code: str, libraries: list | None = None) -> ExecutionResult:
        """Run the code in the sandbox session with artifact extraction."""
        if self.enable_plotting:
            injected_code = self._session.language_handler.inject_plot_detection_code(code)
        else:
            injected_code = code

        result = self._session.run(injected_code, libraries)

        plots = []

        if self.enable_plotting:
            plots = self._session.language_handler.extract_plots(
                self._session,
                "/tmp/sandbox_plots",
            )

        return ExecutionResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            plots=plots,
        )


SandboxSession = create_session
