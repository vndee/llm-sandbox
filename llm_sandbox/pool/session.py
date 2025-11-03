"""Pool-aware session classes for using containers from a pool."""

from types import TracebackType
from typing import Any

from llm_sandbox.const import SandboxBackend, SupportedLanguage
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.core.session_base import BaseSession
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.pool.base import ContainerPoolManager, PooledContainer
from llm_sandbox.pool.config import PoolConfig
from llm_sandbox.pool.factory import create_pool_manager
from llm_sandbox.security import SecurityPolicy


class PooledSandboxSession(BaseSession):
    """Sandbox session that uses containers from a pool.

    This session class acquires containers from a pool manager instead
    of creating new ones, significantly improving performance for
    repeated code execution.

    The session automatically:
    - Acquires a container from the pool on open()
    - Uses the pooled container for execution
    - Returns the container to the pool on close()
    - Handles pool exhaustion based on configured strategy
    """

    def __init__(
        self,
        pool_manager: ContainerPoolManager | None = None,
        pool_config: PoolConfig | None = None,
        backend: SandboxBackend = SandboxBackend.DOCKER,
        lang: str = SupportedLanguage.PYTHON,
        image: str | None = None,
        verbose: bool = False,
        workdir: str = "/sandbox",
        security_policy: SecurityPolicy | None = None,
        default_timeout: float | None = None,
        execution_timeout: float | None = None,
        session_timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize pooled sandbox session.

        Args:
            pool_manager: Existing pool manager to use (creates new if None)
            pool_config: Pool configuration (only used if pool_manager is None)
            backend: Container backend (docker, kubernetes, podman)
            lang: Programming language
            image: Container image to use
            verbose: Enable verbose logging
            workdir: Working directory in container
            security_policy: Security policy to enforce
            default_timeout: Default timeout for operations
            execution_timeout: Timeout for code execution
            session_timeout: Maximum session lifetime
            **kwargs: Additional backend-specific arguments

        Examples:
            Using with auto-created pool:
            ```python
            from llm_sandbox.pool import PooledSandboxSession, PoolConfig

            pool_config = PoolConfig(max_pool_size=5, min_pool_size=2)

            with PooledSandboxSession(
                pool_config=pool_config,
                lang="python",
            ) as session:
                result = session.run("print('Hello from pool!')")
                print(result.stdout)
            ```

            Using with shared pool manager:
            ```python
            from llm_sandbox.pool import create_pool_manager, PooledSandboxSession

            # Create shared pool
            pool = create_pool_manager(
                backend="docker",
                config=PoolConfig(max_pool_size=10),
                lang="python",
            )

            # Multiple sessions can share the same pool
            with PooledSandboxSession(pool_manager=pool) as session1:
                result1 = session1.run("print('Session 1')")

            with PooledSandboxSession(pool_manager=pool) as session2:
                result2 = session2.run("print('Session 2')")

            # Clean up pool when done
            pool.close()
            ```

        """
        # Create session config
        config = SessionConfig(
            lang=SupportedLanguage(lang.upper()),
            image=image,
            verbose=verbose,
            workdir=workdir,
            security_policy=security_policy,
            default_timeout=default_timeout,
            execution_timeout=execution_timeout,
            session_timeout=session_timeout,
        )

        super().__init__(config=config, **kwargs)

        # Pool management
        self.backend = backend
        self._pool_manager = pool_manager
        self._pool_config = pool_config or PoolConfig()
        self._owns_pool = pool_manager is None
        self._pooled_container: PooledContainer | None = None
        self._session_kwargs = kwargs

        # Create pool manager if not provided
        if self._owns_pool:
            self._pool_manager = create_pool_manager(
                backend=backend,
                config=self._pool_config,
                lang=SupportedLanguage(lang.upper()),
                image=image,
                **kwargs,
            )

    def open(self) -> None:
        """Open session by acquiring a container from the pool.

        Raises:
            PoolExhaustedError: If pool is exhausted and strategy doesn't allow waiting
            PoolClosedError: If pool has been closed
            RuntimeError: If pool manager is not initialized

        """
        super().open()

        # Acquire container from pool
        if not self._pool_manager:
            msg = "Pool manager not initialized"
            raise RuntimeError(msg)

        self._pooled_container = self._pool_manager.acquire()
        self.container = self._pooled_container.container
        self.container_api = self._create_container_api()

        self._log(f"Acquired container {self._pooled_container.container_id} from pool")

    def close(self) -> None:
        """Close session and return container to pool.

        The container is returned to the pool for reuse unless:
        - It has been marked unhealthy
        - The pool has been closed
        - The session owns the pool (in which case pool is closed)
        """
        super().close()

        if self._pooled_container and self._pool_manager:
            self._log(f"Returning container {self._pooled_container.container_id} to pool")
            self._pool_manager.release(self._pooled_container)
            self._pooled_container = None

        self.container = None

        # Close pool if we own it
        if self._owns_pool and self._pool_manager:
            self._pool_manager.close()
            self._pool_manager = None

    def _create_container_api(self) -> Any:
        """Create appropriate container API for the backend.

        Returns:
            Container API instance

        Raises:
            RuntimeError: If pool manager is not initialized

        """
        if not self._pool_manager:
            msg = "Pool manager not initialized"
            raise RuntimeError(msg)

        match self.backend:
            case SandboxBackend.DOCKER:
                from llm_sandbox.docker import DockerContainerAPI
                from llm_sandbox.pool.docker_pool import DockerPoolManager

                if not isinstance(self._pool_manager, DockerPoolManager):
                    msg = f"Expected DockerPoolManager, got {type(self._pool_manager)}"
                    raise TypeError(msg)

                return DockerContainerAPI(
                    client=self._pool_manager.client,
                    stream=False,
                )

            case SandboxBackend.KUBERNETES:
                from llm_sandbox.kubernetes import KubernetesContainerAPI
                from llm_sandbox.pool.kubernetes_pool import KubernetesPoolManager

                if not isinstance(self._pool_manager, KubernetesPoolManager):
                    msg = f"Expected KubernetesPoolManager, got {type(self._pool_manager)}"
                    raise TypeError(msg)

                return KubernetesContainerAPI(
                    client=self._pool_manager.client,
                    namespace=self._pool_manager.namespace,
                )

            case SandboxBackend.PODMAN:
                from llm_sandbox.podman import PodmanContainerAPI
                from llm_sandbox.pool.podman_pool import PodmanPoolManager

                if not isinstance(self._pool_manager, PodmanPoolManager):
                    msg = f"Expected PodmanPoolManager, got {type(self._pool_manager)}"
                    raise TypeError(msg)

                return PodmanContainerAPI(
                    client=self._pool_manager.client,
                    stream=False,
                )

    def _handle_timeout(self) -> None:
        """Handle timeout cleanup for pooled session.

        For pooled sessions, we don't want to destroy the container,
        just return it to the pool.
        """
        # Just close the session, which will return container to pool
        try:
            self.close()
        except Exception as e:  # noqa: BLE001
            self._log(f"Error during timeout cleanup: {e}", "error")

    def _connect_to_existing_container(self, container_id: str) -> None:
        """Not supported for pooled sessions.

        Pooled sessions always acquire containers from the pool.

        Args:
            container_id: Container ID (ignored)

        Raises:
            NotImplementedError: Always raised

        """
        msg = "Pooled sessions do not support connecting to existing containers"
        raise NotImplementedError(msg)

    def environment_setup(self) -> None:
        """Skip environment setup for pooled containers.

        Pooled containers are pre-warmed during pool initialization,
        so we don't need to set up the environment again.
        """
        if self.config.skip_environment_setup or self._pooled_container:
            self._log("Skipping environment setup for pooled container", "info")
            return

        # Call parent if somehow not using pooled container
        super().environment_setup()

    def _ensure_directory_exists(self, path: str) -> None:
        """Ensure a directory exists in the container.

        Args:
            path: Directory path to create

        """
        if not self.container_api:
            msg = "Container API not initialized"
            raise RuntimeError(msg)
        self.container_api.execute_command(self.container, f"mkdir -p {path}")

    def _ensure_ownership(self, paths: list[str]) -> None:
        """Ensure proper ownership of files or directories.

        Args:
            paths: List of paths to set ownership for

        """
        # For pooled containers, ownership is already set during initialization

    def _process_stream_output(self, output: Any) -> tuple[str, str]:
        """Process streaming output from container.

        Args:
            output: Output stream from container

        Returns:
            Tuple of (stdout, stderr)

        """
        stdout_lines = []
        stderr_lines = []

        for line in output:
            if isinstance(line, dict):
                if "stream" in line:
                    stdout_lines.append(line["stream"])
                elif "error" in line:
                    stderr_lines.append(line["error"])
            elif isinstance(line, str):
                stdout_lines.append(line)

        return "".join(stdout_lines), "".join(stderr_lines)

    def _process_non_stream_output(self, output: Any) -> tuple[str, str]:
        """Process non-streaming output from container.

        Args:
            output: Result from container execution

        Returns:
            Tuple of (stdout, stderr)

        """
        if isinstance(output, bytes):
            return output.decode("utf-8"), ""
        return str(output), ""

    def get_archive(self, path: str) -> tuple[Any, ...]:
        """Get an archive of a file or directory from the container.

        Args:
            path: Path to file or directory in container

        Returns:
            Tuple containing archive data and metadata

        """
        if not self.container_api:
            msg = "Container API not initialized"
            raise RuntimeError(msg)

        # Check if container API has get_archive method
        if not hasattr(self.container_api, "get_archive"):
            msg = f"Container API {type(self.container_api)} does not support get_archive"
            raise NotImplementedError(msg)

        return self.container_api.get_archive(self.container, path)  # type: ignore[attr-defined,no-any-return]


class ArtifactPooledSandboxSession:
    """Pooled sandbox session with artifact extraction capabilities.

    This is the pooled version of ArtifactSandboxSession, combining
    artifact extraction with container pooling for maximum performance.
    """

    def __init__(
        self,
        pool_manager: ContainerPoolManager | None = None,
        pool_config: PoolConfig | None = None,
        backend: SandboxBackend = SandboxBackend.DOCKER,
        image: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        *,
        verbose: bool = False,
        workdir: str = "/sandbox",
        enable_plotting: bool = True,
        security_policy: SecurityPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new artifact pooled sandbox session.

        Args:
            pool_manager: Existing pool manager to use
            pool_config: Pool configuration
            backend: Container backend
            image: Container image to use
            lang: Programming language
            verbose: Enable verbose logging
            workdir: Working directory
            enable_plotting: Enable plot extraction
            security_policy: Security policy
            **kwargs: Additional arguments

        Examples:
            ```python
            from llm_sandbox.pool import ArtifactPooledSandboxSession, PoolConfig
            import base64
            from pathlib import Path

            pool_config = PoolConfig(max_pool_size=5, min_pool_size=2)

            with ArtifactPooledSandboxSession(
                pool_config=pool_config,
                lang="python",
                enable_plotting=True,
            ) as session:
                code = '''
                import matplotlib.pyplot as plt
                import numpy as np

                x = np.linspace(0, 10, 100)
                y = np.sin(x)

                plt.plot(x, y)
                plt.title('Sine Wave')
                plt.show()
                '''

                result = session.run(code, libraries=["matplotlib", "numpy"])

                # Save plots
                for i, plot in enumerate(result.plots):
                    with open(f"plot_{i}.{plot.format.value}", "wb") as f:
                        f.write(base64.b64decode(plot.content_base64))
            ```

        """
        # Create the base pooled session
        self._session = PooledSandboxSession(
            pool_manager=pool_manager,
            pool_config=pool_config,
            backend=backend,
            image=image,
            lang=lang,
            verbose=verbose,
            workdir=workdir,
            security_policy=security_policy,
            **kwargs,
        )

        self.enable_plotting = enable_plotting

    def __enter__(self) -> "ArtifactPooledSandboxSession":
        """Enter context manager."""
        self._session.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        return self._session.__exit__(exc_type, exc_val, exc_tb)

    def __getattr__(self, name: str) -> Any:
        """Delegate attributes to underlying session."""
        return getattr(self._session, name)

    def run(
        self,
        code: str,
        libraries: list | None = None,
        timeout: float | None = None,
        clear_plots: bool = False,
    ) -> ConsoleOutput:
        """Run code and extract artifacts.

        Args:
            code: Code to execute
            libraries: Libraries to install
            timeout: Execution timeout
            clear_plots: Clear existing plots before running

        Returns:
            ExecutionResult with plots

        """
        from llm_sandbox.data import ExecutionResult
        from llm_sandbox.exceptions import LanguageNotSupportPlotError

        # Check if plotting is supported
        if self.enable_plotting and not self._session.language_handler.is_support_plot_detection:
            raise LanguageNotSupportPlotError(self._session.language_handler.name)

        # Clear plots if requested
        if clear_plots and self.enable_plotting:
            self._clear_plots_in_container()

        # Use config default timeout if not specified
        if timeout is not None:
            effective_timeout = timeout
        else:
            config_timeout = self._session.config.get_execution_timeout()
            effective_timeout = config_timeout if config_timeout is not None else 60

        # Delegate to language handler for artifact extraction
        result, plots = self._session.language_handler.run_with_artifacts(
            container=self._session,
            code=code,
            libraries=libraries,
            enable_plotting=self.enable_plotting,
            output_dir="/tmp/sandbox_plots",
            timeout=int(effective_timeout),
        )

        return ExecutionResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            plots=plots,
        )

    def _clear_plots_in_container(self) -> None:
        """Clear plots in the container."""
        if not self.enable_plotting:
            return

        self._session.execute_command(
            'sh -c "mkdir -p /tmp/sandbox_plots && rm -rf /tmp/sandbox_plots/* && echo 0 > /tmp/sandbox_plots/.counter"'
        )

    def clear_plots(self) -> None:
        """Manually clear all plots."""
        if not self.enable_plotting:
            return

        self._clear_plots_in_container()
