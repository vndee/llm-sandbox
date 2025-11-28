"""Main session module for LLM Sandbox."""

from importlib.util import find_spec
from types import TracebackType
from typing import Any

from llm_sandbox.security import SecurityPolicy

from .const import SandboxBackend, SupportedLanguage
from .core.session_base import BaseSession
from .data import ExecutionResult
from .exceptions import LanguageNotSupportPlotError, MissingDependencyError, UnsupportedBackendError
from .interactive import InteractiveSandboxSession
from .pool.base import ContainerPoolManager
from .pool.session import ArtifactPooledSandboxSession, PooledSandboxSession

__all__ = [
    "ArtifactSandboxSession",
    "ContainerPoolManager",
    "InteractiveSandboxSession",
    "PooledSandboxSession",
    "SandboxSession",
    "create_session",
]


def _check_dependency(backend: SandboxBackend) -> None:
    """Check if required dependency is installed for the given backend."""
    if backend in {SandboxBackend.DOCKER, SandboxBackend.MICROMAMBA} and not find_spec("docker"):
        msg = "Docker backend requires 'docker' package. Install it with: pip install llm-sandbox[docker]"
        raise MissingDependencyError(msg)
    if backend == SandboxBackend.KUBERNETES and not find_spec("kubernetes"):
        msg = "Kubernetes backend requires 'kubernetes' package. Install it with: pip install llm-sandbox[k8s]"
        raise MissingDependencyError(msg)
    if backend == SandboxBackend.PODMAN and not find_spec("podman"):
        msg = "Podman backend requires 'podman' package. Install it with: pip install llm-sandbox[podman]"
        raise MissingDependencyError(msg)


def create_session(
    backend: SandboxBackend = SandboxBackend.DOCKER,
    pool: ContainerPoolManager | None = None,
    *args: Any,
    **kwargs: Any,
) -> BaseSession | PooledSandboxSession:
    r"""Create a new sandbox session for executing code in an isolated environment.

    This function creates a sandbox session that supports multiple programming languages
    and provides features like package installation, file operations, and secure code execution.
    For backward compatibility, we also keep a `SandboxSession` alias for this function.

    Args:
        backend (SandboxBackend): Container backend to use. Options:
            - SandboxBackend.DOCKER (default)
            - SandboxBackend.KUBERNETES
            - SandboxBackend.PODMAN
            - SandboxBackend.MICROMAMBA
        pool (ContainerPoolManager | None): Pool manager to use for container pooling.
            If provided, containers are acquired from the pool instead of being created new.
            Create a pool manager using `create_pool_manager()` from llm_sandbox.pool.
        *args: Additional positional arguments passed to the session constructor
        **kwargs: Additional keyword arguments passed to the session constructor.
                Common options include:
                    - lang (str): Programming language ("python", "java", "javascript", "cpp", "go")
                    - verbose (bool): Enable verbose logging
                    - keep_template (bool): Keep the container template
                    - image (str): Custom container image to use
                    - container_id (str): ID of existing container/pod to connect to

    Returns:
        Session: A sandbox session instance for the specified backend

    Raises:
        MissingDependencyError: If the required dependency for the chosen backend is not installed
        UnsupportedBackendError: If the chosen backend is not supported

    Examples:
        Using container pooling for improved performance:
        ```python
        from llm_sandbox import SandboxSession, SupportedLanguage, SandboxBackend
        from llm_sandbox.pool import create_pool_manager, PoolConfig

        # Create a pool manager
        pool = create_pool_manager(
            backend=SandboxBackend.DOCKER,
            config=PoolConfig(max_pool_size=10, min_pool_size=3),
            lang=SupportedLanguage.PYTHON,
        )

        # Use pooled session
        with SandboxSession(pool=pool, lang=SupportedLanguage.PYTHON) as session:
            result = session.run("print('Hello from pool!')")
            print(result.stdout)

        # Multiple sessions can share the same pool
        with SandboxSession(pool=pool, lang=SupportedLanguage.PYTHON) as session2:
            result2 = session2.run("print('Session 2')")

        # Clean up pool when done
        pool.close()
        ```

        Connect to existing Docker container:
        ```python
        # Assumes you have a running container with ID 'abc123...'
        with SandboxSession(container_id='abc123def456', lang=SupportedLanguage.PYTHON) as session:
            result = session.run("print('Hello from existing container!')")
            print(result.stdout)

            # Install libraries in existing container
            session.install(["numpy"])
            result = session.run("import numpy as np; print(np.random.rand())")

            # Execute commands
            result = session.execute_command("ls -la")

            # Copy files
            session.copy_to_runtime("local_file.py", "/container/path/file.py")
        ```

        Connect to existing Kubernetes pod:
        ```python
        # Assumes you have a running pod with name 'my-pod-abc123'
        with SandboxSession(
            backend=SandboxBackend.KUBERNETES,
            container_id='my-pod-abc123',  # pod name
            lang=SupportedLanguage.PYTHON
        ) as session:
            result = session.run("print('Hello from existing pod!')")
        ```

        Connect to existing Podman container:
        ```python
        from podman import PodmanClient

        client = PodmanClient()
        with SandboxSession(
            backend=SandboxBackend.PODMAN,
            client=client,
            container_id='podman-container-id',
            lang=SupportedLanguage.PYTHON
        ) as session:
            result = session.run("print('Hello from existing Podman container!')")
        ```

        Python session with package installation:
        ```python
        with SandboxSession(lang=SupportedLanguage.PYTHON, keep_template=True, verbose=True) as session:
            # Basic code execution
            result = session.run("print('Hello, World!')")
            print(result.stdout)  # Output: Hello, World!

            # Install and use packages
            result = session.run(
                "import numpy as np\nprint(np.random.rand())",
                libraries=["numpy"]
            )

            # Install additional packages during session
            session.install(["pandas"])
            result = session.run("import pandas as pd\nprint(pd.__version__)")

            # Copy files to runtime
            session.copy_to_runtime("README.md", "/sandbox/data.csv")
        ```

        Java session:
        ```python
        with SandboxSession(lang=SupportedLanguage.JAVA, keep_template=True, verbose=True) as session:
            result = session.run(\"\"\"
                public class Main {
                    public static void main(String[] args) {
                        System.out.println("Hello, World!");
                    }
                }
            \"\"\")
        ```

        JavaScript session with npm packages:
        ```python
        with SandboxSession(lang=SupportedLanguage.JAVASCRIPT, keep_template=True, verbose=True) as session:
            # Basic code execution
            result = session.run("console.log('Hello, World!')")

            # Using npm packages
            result = session.run(\"\"\"
                const axios = require('axios');
                axios.get('https://jsonplaceholder.typicode.com/posts/1')
                    .then(response => console.log(response.data));
            \"\"\", libraries=["axios"])
        ```

        C++ session:
        ```python
        with SandboxSession(lang=SupportedLanguage.CPP, keep_template=True, verbose=True) as session:
            result = session.run(\"\"\"
                #include <iostream>
                #include <vector>
                #include <algorithm>
                int main() {
                    std::vector<int> v = {1, 2, 3, 4, 5};
                    std::reverse(v.begin(), v.end());
                    for (int i : v) {
                        std::cout << i << " ";
                    }
                    std::cout << std::endl;
                    return 0;
                }
            \"\"\", libraries=["libstdc++"])
        ```

        Go session with external packages:
        ```python
        with SandboxSession(lang=SupportedLanguage.GO, keep_template=True, verbose=True) as session:
            result = session.run(\"\"\"
                package main
                import (
                    "fmt"
                    "github.com/spyzhov/ajson"
                )
                func main() {
                    json := []byte(`{"price": 100}`)
                    root, _ := ajson.Unmarshal(json)
                    nodes, _ := root.JSONPath("$..price")
                    for _, node := range nodes {
                        node.SetNumeric(node.MustNumeric() * 1.25)
                    }
                    result, _ := ajson.Marshal(root)
                    fmt.Printf("%s", result)
                }
            \"\"\", libraries=["github.com/spyzhov/ajson"])
        ```

    """
    # Use pooled session if pool manager is provided
    if pool is not None:
        from llm_sandbox.pool.session import PooledSandboxSession

        return PooledSandboxSession(
            pool_manager=pool,
            **kwargs,
        )

    # Check if required dependency is installed for non-pooled sessions
    _check_dependency(backend)

    # Create the appropriate session based on backend
    match backend:
        case SandboxBackend.DOCKER:
            from .docker import SandboxDockerSession

            return SandboxDockerSession(*args, **kwargs)
        case SandboxBackend.KUBERNETES:
            from .kubernetes import SandboxKubernetesSession

            return SandboxKubernetesSession(*args, **kwargs)
        case SandboxBackend.PODMAN:
            from .podman import SandboxPodmanSession

            return SandboxPodmanSession(*args, **kwargs)
        case SandboxBackend.MICROMAMBA:
            from .micromamba import MicromambaSession

            return MicromambaSession(*args, **kwargs)
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
        pool: "ContainerPoolManager | None" = None,
        keep_template: bool = False,
        commit_container: bool = False,
        verbose: bool = False,
        runtime_configs: dict | None = None,
        workdir: str | None = "/sandbox",
        enable_plotting: bool = True,
        security_policy: SecurityPolicy | None = None,
        container_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new artifact sandbox session.

        The ArtifactSandboxSession provides a secure environment for running code that generates artifacts
        like plots, images, or other files. It supports multiple container backends (Docker, Kubernetes, Podman)
        and can capture and extract artifacts from the execution.

        Args:
            backend (SandboxBackend): Container backend to use (Docker, Kubernetes or Podman)
            image (str): Container image to use (e.g., "vndee/sandbox-python-311-bullseye")
            dockerfile (str, optional): Path to Dockerfile
            lang (str): Programming language (e.g., "python")
            pool (ContainerPoolManager, optional): Pool manager for container pooling. When provided,
                the session will use pooled containers for better performance
            keep_template (bool, optional): Whether to keep the container template
            commit_container (bool, optional): Whether to commit container changes
            verbose (bool, optional): Enable verbose logging
            runtime_configs (dict, optional): Additional runtime configurations
            workdir (str, optional): Working directory inside the container
            enable_plotting (bool, optional): Whether to enable plot extraction
            security_policy (SecurityPolicy, optional): Security policy to enforce
            container_id (str, optional): ID of existing container/pod to connect to
            **kwargs: Additional keyword arguments for specific backends (e.g., client for Podman)

        Raises:
            MissingDependencyError: If the required dependency for the chosen backend is not installed
            UnsupportedBackendError: If the chosen backend is not supported

        Examples:
            Connect to existing container for artifact generation:
            ```python
            from llm_sandbox import ArtifactSandboxSession, SandboxBackend, SupportedLanguage
            from pathlib import Path
            import base64

            # Connect to existing container
            with ArtifactSandboxSession(
                container_id='existing-container-id',
                lang=SupportedLanguage.PYTHON,
                verbose=True,
                backend=SandboxBackend.DOCKER
            ) as session:
                # Code that generates plots in existing environment
                code = '''
                import matplotlib.pyplot as plt
                import numpy as np

                # Generate and plot data
                x = np.linspace(0, 10, 100)
                y = np.sin(x)

                plt.figure()
                plt.plot(x, y)
                plt.title('Plot from Existing Container')
                plt.show()
                '''

                result = session.run(code)
                print(f"Captured {len(result.plots)} plots")

                # Save captured plots
                for i, plot in enumerate(result.plots):
                    plot_path = Path("plots") / f"existing_{i + 1:06d}.{plot.format.value}"
                    with plot_path.open("wb") as f:
                        f.write(base64.b64decode(plot.content_base64))
            ```

            Basic usage with Docker backend:
            ```python
            from llm_sandbox import ArtifactSandboxSession, SandboxBackend
            from pathlib import Path
            import base64

            # Create plots directory
            Path("plots/docker").mkdir(parents=True, exist_ok=True)

            # Run code that generates plots
            with ArtifactSandboxSession(
                lang=SupportedLanguage.PYTHON,
                verbose=True,
                image="ghcr.io/vndee/sandbox-python-311-bullseye",
                backend=SandboxBackend.DOCKER
            ) as session:
                # Example code that generates matplotlib, seaborn, and plotly plots
                code = '''
                import matplotlib.pyplot as plt
                import numpy as np

                # Generate and plot data
                x = np.linspace(0, 10, 100)
                y = np.sin(x)

                plt.figure()
                plt.plot(x, y)
                plt.title('Simple Sine Wave')
                plt.show()
                '''

                result = session.run(code)
                print(f"Captured {len(result.plots)} plots")

                # Save captured plots
                for i, plot in enumerate(result.plots):
                    plot_path = Path("plots/docker") / f"{i + 1:06d}.{plot.format.value}"
                    with plot_path.open("wb") as f:
                        f.write(base64.b64decode(plot.content_base64))
            ```
            Using Podman backend:
            ```python
            from podman import PodmanClient

            # Initialize Podman client
            podman_client = PodmanClient(base_url="unix:///path/to/podman.sock")

            with ArtifactSandboxSession(
                client=podman_client,  # Podman specific
                lang=SupportedLanguage.PYTHON,
                verbose=True,
                image="ghcr.io/vndee/sandbox-python-311-bullseye",
                backend=SandboxBackend.PODMAN
            ) as session:
                result = session.run(code)
            ```

            Using Kubernetes backend:
            ```python
            with ArtifactSandboxSession(
                lang=SupportedLanguage.PYTHON,
                verbose=True,
                image="ghcr.io/vndee/sandbox-python-311-bullseye",
                backend=SandboxBackend.KUBERNETES
            ) as session:
                result = session.run(code)
            ```

            Using container pooling for better performance:
            ```python
            from llm_sandbox import ArtifactSandboxSession
            from llm_sandbox.pool import create_pool_manager, PoolConfig
            import base64
            from pathlib import Path

            # Create pool manager
            pool = create_pool_manager(
                backend="docker",
                config=PoolConfig(max_pool_size=5, min_pool_size=2),
                lang="python",
                libraries=["matplotlib", "numpy"],
            )

            try:
                # Use pool parameter for pooled execution
                with ArtifactSandboxSession(pool=pool, enable_plotting=True) as session:
                    result = session.run('''
                    import matplotlib.pyplot as plt
                    import numpy as np

                    x = np.linspace(0, 10, 100)
                    y = np.sin(x)
                    plt.plot(x, y)
                    plt.title('Pooled Container Plot')
                    plt.show()
                    ''')

                    # Save plots
                    for i, plot in enumerate(result.plots):
                        Path(f"plot_{i}.{plot.format.value}").write_bytes(
                            base64.b64decode(plot.content_base64)
                        )
            finally:
                pool.close()
            ```

        """
        # Initialize attributes with proper types
        self._pooled_impl: ArtifactPooledSandboxSession | None
        self._session: BaseSession | PooledSandboxSession | None

        # If pool is provided, delegate to ArtifactPooledSandboxSession
        if pool is not None:
            self._pooled_impl = ArtifactPooledSandboxSession(
                pool_manager=pool,
                verbose=verbose,
                workdir=workdir or "/sandbox",
                enable_plotting=enable_plotting,
                security_policy=security_policy,
                **kwargs,
            )
            self.enable_plotting = enable_plotting
            # Don't create _session when using pooled implementation
            self._session = None
        else:
            # Create the base session (non-pooled)
            self._pooled_impl = None
            self._session = create_session(
                backend=backend,
                image=image,
                dockerfile=dockerfile,
                lang=lang,
                keep_template=keep_template,
                commit_container=commit_container,
                verbose=verbose,
                runtime_configs=runtime_configs,
                workdir=workdir,
                security_policy=security_policy,
                container_id=container_id,
                **kwargs,
            )
            self.enable_plotting = enable_plotting

    def __enter__(self) -> "ArtifactSandboxSession":
        """Enter the context manager."""
        if self._pooled_impl is not None:
            self._pooled_impl.__enter__()
        else:
            self._session.__enter__()  # type: ignore[union-attr]
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager."""
        if self._pooled_impl is not None:
            self._pooled_impl.__exit__(exc_type, exc_val, exc_tb)
        else:
            self._session.__exit__(exc_type, exc_val, exc_tb)  # type: ignore[union-attr]

    def __getattr__(self, name: str) -> Any:
        """Delegate any other attributes/methods to the underlying session."""
        if self._pooled_impl is not None:
            return getattr(self._pooled_impl, name)
        return getattr(self._session, name)

    def run(
        self,
        code: str,
        libraries: list | None = None,
        timeout: float | None = None,
        clear_plots: bool = False,
    ) -> ExecutionResult:
        """Run code in the sandbox session and extract any generated artifacts.

        This method executes the provided code in an isolated environment and captures any
        generated artifacts (e.g., plots, figures). When plotting is enabled, it delegates
        to the language handler's run_with_artifacts method for language-specific artifact
        extraction.

        Args:
            code (str): The code to execute. Can include plotting commands from matplotlib,
                        seaborn, plotly, or other visualization libraries.
            libraries (list | None, optional): Additional libraries to install before running
                                                the code. Defaults to None.
            timeout (float | None, optional): Timeout in seconds for the code execution.
                                                Defaults to the configuration's execution_timeout
                                                (typically 60) or 60 if not configured.
            clear_plots (bool, optional): Whether to clear existing plots before running
                                            the code. Defaults to False.

        Returns:
            ExecutionResult: An object containing:
                - exit_code (int): The exit code of the execution
                - stdout (str): Standard output from the code execution
                - stderr (str): Standard error from the code execution
                - plots (list[Plot]): List of captured plots, each containing:
                    - content_base64 (str): Base64 encoded plot data
                    - format (PlotFormat): Format of the plot (e.g., 'png', 'svg')

        Raises:
            LanguageNotSupportPlotError: If the language does not support plot detection

        Examples:
            Basic plotting example:
            ```python
            with ArtifactSandboxSession(
                lang=SupportedLanguage.PYTHON,
                verbose=True,
                image="ghcr.io/vndee/sandbox-python-311-bullseye"
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
                print(f"Generated {len(result.plots)} plots")
            ```

            Multiple plot types and libraries:
            ```python
            from llm_sandbox import SupportedLanguage

            code = '''
            import matplotlib.pyplot as plt
            import seaborn as sns
            import plotly.express as px
            import pandas as pd
            import numpy as np

            # Matplotlib plot
            plt.figure(figsize=(10, 6))
            x = np.linspace(0, 10, 100)
            plt.plot(x, np.sin(x))
            plt.title('Matplotlib: Sine Wave')
            plt.show()

            # Seaborn plot
            data = pd.DataFrame({
                'x': np.random.randn(100),
                'y': np.random.randn(100)
            })
            sns.scatterplot(data=data, x='x', y='y')
            plt.title('Seaborn: Scatter Plot')
            plt.show()

            # Plotly plot
            fig = px.line(data, x='x', y='y', title='Plotly: Line Plot')
            fig.show()
            '''

            result = session.run(code, libraries=['plotly'])

            # Save the generated plots
            for i, plot in enumerate(result.plots):
                with open(f'plot_{i}.{plot.format.value}', 'wb') as f:
                    f.write(base64.b64decode(plot.content_base64))
            ```

            Installing additional libraries:
            ```python
            from llm_sandbox import SupportedLanguage

            code = '''
            import torch
            import torch.nn as nn
            print(f"PyTorch version: {torch.__version__}")
            '''

            result = session.run(code, libraries=['torch'])
            print(result.stdout)
            ```

            Clearing plots between runs:
            ```python
            from llm_sandbox import SupportedLanguage

            with ArtifactSandboxSession(lang=SupportedLanguage.PYTHON) as session:
                # First run with plots
                plot_code = '''
                import matplotlib.pyplot as plt
                plt.plot([1, 2, 3], [1, 4, 2])
                plt.show()
                '''
                result1 = session.run(plot_code)
                print(f"First run: {len(result1.plots)} plots")

                # Clear plots and run again
                result2 = session.run("print('hello world')", clear_plots=True)
                print(f"Second run: {len(result2.plots)} plots")

                # Manual plot clearing
                session.clear_plots()
                result3 = session.run(plot_code)
                print(f"Third run: {len(result3.plots)} plots")
            ```

        """
        # Delegate to pooled implementation if using pool
        if self._pooled_impl is not None:
            return self._pooled_impl.run(
                code=code,
                libraries=libraries,
                timeout=timeout,
                clear_plots=clear_plots,
            )

        # Non-pooled implementation
        # Check if plotting is enabled and language supports it
        if self.enable_plotting and not self._session.language_handler.is_support_plot_detection:  # type: ignore[union-attr]
            raise LanguageNotSupportPlotError(self._session.language_handler.name)  # type: ignore[union-attr]

        # Clear plots if requested
        if clear_plots and self.enable_plotting:
            self._clear_plots_in_container()

        # Use config default timeout if not specified
        if timeout is not None:
            effective_timeout = timeout
        else:
            config_timeout = self._session.config.get_execution_timeout()  # type: ignore[union-attr]
            effective_timeout = config_timeout if config_timeout is not None else 60

        # Delegate to language handler for language-specific artifact extraction
        result, plots = self._session.language_handler.run_with_artifacts(  # type: ignore[union-attr]
            container=self._session,  # type: ignore[arg-type]
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

        # Delegate to pooled implementation if using pool
        if self._pooled_impl is not None:
            self._pooled_impl._clear_plots_in_container()  # noqa: SLF001
            return

        # Use shell commands to create directory, clear plots, and reset counter
        self._session.execute_command(  # type: ignore[union-attr]
            'sh -c "mkdir -p /tmp/sandbox_plots && rm -rf /tmp/sandbox_plots/* && echo 0 > /tmp/sandbox_plots/.counter"'
        )

    def clear_plots(self) -> None:
        """Manually clear all plots and reset the plot counter.

        This method can be called between runs to clear existing plots
        without creating a new session.

        Raises:
            Exception: If plot clearing is not supported or fails

        """
        if not self.enable_plotting:
            return

        # Delegate to pooled implementation if using pool
        if self._pooled_impl is not None:
            self._pooled_impl.clear_plots()
            return

        self._clear_plots_in_container()


SandboxSession = create_session
