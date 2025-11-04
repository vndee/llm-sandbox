"""Pool-aware session classes for using containers from a pool."""

import logging
from types import TracebackType
from typing import TYPE_CHECKING, Any

from llm_sandbox.const import SandboxBackend
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.pool.base import ContainerPoolManager, PooledContainer
from llm_sandbox.security import SecurityPolicy

if TYPE_CHECKING:
    from llm_sandbox.core.session_base import BaseSession


class DuplicateClientError(ValueError):
    """Error raised when a client is specified both in the pool manager and in the session."""

    def __init__(self) -> None:
        """Initialize the error."""
        message = (
            "Cannot specify 'client' parameter when using pooling mode. "
            "The client is managed by the pool manager. "
            "Please configure the client in the pool manager instead."
        )
        super().__init__(message)


class PooledSandboxSession:
    """Sandbox session that uses containers from a pool.

    This session class acquires containers from a pool manager instead
    of creating new ones, significantly improving performance for
    repeated code execution.

    The session automatically:
    - Acquires a container from the pool on open()
    - Creates a backend-specific session connected to the pooled container
    - Uses the backend session for all operations (leveraging existing implementations)
    - Returns the container to the pool on close()
    - Handles pool exhaustion based on configured strategy
    """

    def __init__(
        self,
        pool_manager: ContainerPoolManager,
        verbose: bool = False,
        stream: bool = False,
        workdir: str = "/sandbox",
        security_policy: SecurityPolicy | None = None,
        default_timeout: float | None = None,
        execution_timeout: float | None = None,
        session_timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize pooled sandbox session.

        Args:
            pool_manager: Pool manager to acquire containers from (required)
            verbose: Enable verbose logging
            stream: Enable streaming output for command execution
            workdir: Working directory in container
            security_policy: Security policy to enforce
            default_timeout: Default timeout for operations
            execution_timeout: Timeout for code execution
            session_timeout: Maximum session lifetime
            **kwargs: Additional backend-specific arguments

        Examples:
            ```python
            from llm_sandbox.pool import create_pool_manager, PooledSandboxSession, PoolConfig

            # Create a pool manager
            pool = create_pool_manager(
                backend="docker",
                config=PoolConfig(max_pool_size=10, min_pool_size=2),
                lang="python",
            )

            # Use the pool in a session
            with PooledSandboxSession(pool_manager=pool) as session:
                result = session.run("print('Hello from pool!')")
                print(result.stdout)

            # Multiple sessions can share the same pool
            with PooledSandboxSession(pool_manager=pool) as session2:
                result2 = session2.run("print('Session 2')")

            # Clean up pool when done
            pool.close()
            ```

        """
        # Pool management
        self._pool_manager = pool_manager
        self._pooled_container: PooledContainer | None = None

        # Infer backend from pool manager
        self.backend = self._infer_backend_from_pool()

        # Session parameters (stored for creating backend session later)
        self._verbose = verbose
        self._stream = stream
        self._workdir = workdir
        self._security_policy = security_policy
        self._default_timeout = default_timeout
        self._execution_timeout = execution_timeout
        self._session_timeout = session_timeout

        # Extract common parameters that the backend session expects, falling back to pool defaults
        self._lang = kwargs.pop("lang", self._pool_manager.lang)
        self._image = kwargs.pop("image", self._pool_manager.image)
        self._session_kwargs = kwargs

        # The actual backend session (created on open())
        self._backend_session: BaseSession | None = None

        # Logger for verbose output
        self._logger = logging.getLogger(__name__)
        if self._verbose and not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.DEBUG)

    def _infer_backend_from_pool(self) -> SandboxBackend:
        """Infer backend type from pool manager class.

        Returns:
            SandboxBackend enum value

        Raises:
            RuntimeError: If backend cannot be determined

        """
        pool_class_name = self._pool_manager.__class__.__name__

        if "Docker" in pool_class_name:
            return SandboxBackend.DOCKER
        if "Kubernetes" in pool_class_name:
            return SandboxBackend.KUBERNETES
        if "Podman" in pool_class_name:
            return SandboxBackend.PODMAN

        msg = f"Cannot infer backend from pool manager class: {pool_class_name}"
        raise RuntimeError(msg)

    def open(self) -> None:
        """Open session by acquiring a container from the pool.

        Raises:
            PoolExhaustedError: If pool is exhausted and strategy doesn't allow waiting
            PoolClosedError: If pool has been closed
            RuntimeError: If pool manager is not initialized

        """
        # Acquire container from pool
        if not self._pool_manager:
            msg = "Pool manager not initialized"
            raise RuntimeError(msg)

        self._pooled_container = self._pool_manager.acquire()

        # Create backend-specific session connected to the pooled container
        self._backend_session = self._create_backend_session(self._pooled_container.container_id)

        # Open the backend session (this connects to the existing container)
        if self._backend_session:
            self._backend_session.open()

        if self._verbose:
            self._logger.info("Acquired container %s from pool", self._pooled_container.container_id)

    def close(self) -> None:
        """Close session and return container to pool.

        The container is returned to the pool for reuse unless:
        - It has been marked unhealthy
        - The pool has been closed
        """
        # Close the backend session (but don't remove the container)
        if self._backend_session:
            self._backend_session.close()
            self._backend_session = None

        # Return container to pool
        if self._pooled_container and self._pool_manager:
            if self._verbose:
                self._logger.info("Returning container %s to pool", self._pooled_container.container_id)
            self._pool_manager.release(self._pooled_container)
            self._pooled_container = None

    def _create_backend_session(self, container_id: str) -> Any:
        """Create backend-specific session for the pooled container.

        Args:
            container_id: ID of the pooled container

        Returns:
            Backend-specific session instance

        Raises:
            ValueError: If 'client' parameter is passed (client is managed by pool)
            RuntimeError: If backend is not supported

        """
        match self.backend:
            case SandboxBackend.DOCKER:
                from llm_sandbox.docker import SandboxDockerSession

                # Validate that pool-managed parameters are not passed
                session_kwargs = self._session_kwargs.copy()
                if "client" in session_kwargs:
                    raise DuplicateClientError

                return SandboxDockerSession(
                    client=self._pool_manager.client,
                    image=self._image,
                    lang=self._lang,
                    verbose=self._verbose,
                    stream=self._stream,
                    workdir=self._workdir,
                    security_policy=self._security_policy,
                    default_timeout=self._default_timeout,
                    execution_timeout=self._execution_timeout,
                    session_timeout=self._session_timeout,
                    container_id=container_id,  # Connect to existing pooled container
                    skip_environment_setup=True,  # Pool already set up the environment
                    **session_kwargs,
                )

            case SandboxBackend.KUBERNETES:
                from llm_sandbox.kubernetes import SandboxKubernetesSession

                # Validate that pool-managed parameters are not passed
                session_kwargs = self._session_kwargs.copy()
                if "client" in session_kwargs:
                    raise DuplicateClientError

                namespace = session_kwargs.pop("namespace", None) or (
                    self._pool_manager.namespace if hasattr(self._pool_manager, "namespace") else "default"
                )

                return SandboxKubernetesSession(
                    client=self._pool_manager.client,
                    namespace=namespace,
                    image=self._image,
                    lang=self._lang,
                    verbose=self._verbose,
                    workdir=self._workdir,
                    security_policy=self._security_policy,
                    default_timeout=self._default_timeout,
                    execution_timeout=self._execution_timeout,
                    session_timeout=self._session_timeout,
                    pod_id=container_id,  # Connect to existing pooled pod
                    skip_environment_setup=True,
                    **session_kwargs,
                )

            case SandboxBackend.PODMAN:
                from llm_sandbox.podman import SandboxPodmanSession

                # Validate that pool-managed parameters are not passed
                session_kwargs = self._session_kwargs.copy()
                if "client" in session_kwargs:
                    raise DuplicateClientError

                return SandboxPodmanSession(
                    client=self._pool_manager.client,
                    image=self._image,
                    lang=self._lang,
                    verbose=self._verbose,
                    stream=self._stream,
                    workdir=self._workdir,
                    security_policy=self._security_policy,
                    default_timeout=self._default_timeout,
                    execution_timeout=self._execution_timeout,
                    session_timeout=self._session_timeout,
                    container_id=container_id,  # Connect to existing pooled container
                    skip_environment_setup=True,
                    **session_kwargs,
                )

            case _:
                msg = f"Unsupported backend: {self.backend}"
                raise RuntimeError(msg)

    def run(self, code: str, libraries: list | None = None, timeout: float | None = None) -> ConsoleOutput:
        """Run code in the pooled container.

        Args:
            code: Code to execute
            libraries: Libraries to install before execution
            timeout: Execution timeout

        Returns:
            ConsoleOutput with execution results

        Raises:
            NotOpenSessionError: If session is not open
            RuntimeError: If backend session is not initialized

        """
        if not self._backend_session:
            msg = "Session not open - call open() first or use context manager"
            raise RuntimeError(msg)

        return self._backend_session.run(code, libraries=libraries, timeout=timeout)

    def execute_command(self, command: str, workdir: str | None = None) -> ConsoleOutput:
        """Execute a command in the pooled container.

        Args:
            command: Command to execute
            workdir: Working directory for command execution

        Returns:
            ConsoleOutput with command results

        Raises:
            RuntimeError: If backend session is not initialized

        """
        if not self._backend_session:
            msg = "Session not open - call open() first or use context manager"
            raise RuntimeError(msg)

        return self._backend_session.execute_command(command, workdir=workdir)

    def copy_to_runtime(self, src: str, dest: str) -> None:
        """Copy file to the pooled container.

        Args:
            src: Source file path on host
            dest: Destination path in container

        Raises:
            RuntimeError: If backend session is not initialized

        """
        if not self._backend_session:
            msg = "Session not open - call open() first or use context manager"
            raise RuntimeError(msg)

        self._backend_session.copy_to_runtime(src, dest)

    def copy_from_runtime(self, src: str, dest: str) -> None:
        """Copy file from the pooled container.

        Args:
            src: Source path in container
            dest: Destination file path on host

        Raises:
            RuntimeError: If backend session is not initialized

        """
        if not self._backend_session:
            msg = "Session not open - call open() first or use context manager"
            raise RuntimeError(msg)

        self._backend_session.copy_from_runtime(src, dest)

    def __enter__(self) -> "PooledSandboxSession":
        """Enter context manager."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context manager."""
        self.close()

    @property
    def backend_session(self) -> Any:
        """Get the backend session instance.

        Returns:
            The underlying backend-specific session

        Raises:
            RuntimeError: If session is not open

        """
        if self._backend_session is None:
            msg = "Session not open - call open() first or use context manager"
            raise RuntimeError(msg)
        return self._backend_session

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to backend session.

        This allows pooled sessions to support all methods/attributes
        of the underlying backend session.
        """
        if self._backend_session is None:
            msg = f"Cannot access '{name}' - session not open"
            raise AttributeError(msg)
        return getattr(self._backend_session, name)


class ArtifactPooledSandboxSession:
    """Pooled sandbox session with artifact extraction capabilities.

    This is the pooled version of ArtifactSandboxSession, combining
    artifact extraction with container pooling for maximum performance.
    """

    def __init__(
        self,
        pool_manager: ContainerPoolManager,
        *,
        verbose: bool = False,
        stream: bool = False,
        workdir: str = "/sandbox",
        enable_plotting: bool = True,
        security_policy: SecurityPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new artifact pooled sandbox session.

        Args:
            pool_manager: Pool manager to acquire containers from (required)
            verbose: Enable verbose logging
            stream: Enable streaming output for command execution
            workdir: Working directory
            enable_plotting: Enable plot extraction
            security_policy: Security policy
            **kwargs: Additional arguments

        Examples:
            ```python
            from llm_sandbox.pool import create_pool_manager, ArtifactPooledSandboxSession, PoolConfig
            import base64

            # Create a pool manager
            pool = create_pool_manager(
                backend="docker",
                config=PoolConfig(max_pool_size=5, min_pool_size=2),
                lang="python",
                libraries=["matplotlib", "numpy"],
            )

            with ArtifactPooledSandboxSession(
                pool_manager=pool,
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

                result = session.run(code)

                # Save plots
                for i, plot in enumerate(result.plots):
                    with open(f"plot_{i}.{plot.format.value}", "wb") as f:
                        f.write(base64.b64decode(plot.content_base64))

            pool.close()
            ```

        """
        # Create the base pooled session
        self._session = PooledSandboxSession(
            pool_manager=pool_manager,
            verbose=verbose,
            stream=stream,
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

        # Get backend session
        backend_session = self._session.backend_session

        # Check if plotting is supported
        if self.enable_plotting and not backend_session.language_handler.is_support_plot_detection:
            raise LanguageNotSupportPlotError(backend_session.language_handler.name)

        # Clear plots if requested
        if clear_plots and self.enable_plotting:
            self._clear_plots_in_container()

        # Use config default timeout if not specified
        if timeout is not None:
            effective_timeout = timeout
        else:
            config_timeout = backend_session.config.get_execution_timeout()
            effective_timeout = config_timeout if config_timeout is not None else 60

        # Delegate to language handler for artifact extraction
        result, plots = backend_session.language_handler.run_with_artifacts(
            container=backend_session,
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
