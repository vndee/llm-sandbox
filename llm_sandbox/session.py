"""Main session module for LLM Sandbox."""

from importlib.util import find_spec
from types import TracebackType
from typing import Any

from llm_sandbox.security import SecurityPolicy

from .base import ExecutionResult, Session
from .const import SandboxBackend, SupportedLanguage
from .exceptions import LanguageNotSupportPlotError, MissingDependencyError, UnsupportedBackendError


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
    *args: Any,
    **kwargs: Any,
) -> Session:
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
        *args: Additional positional arguments passed to the session constructor
        **kwargs: Additional keyword arguments passed to the session constructor.
                Common options include:
                    - lang (str): Programming language ("python", "java", "javascript", "cpp", "go")
                    - verbose (bool): Enable verbose logging
                    - keep_template (bool): Keep the container template
                    - image (str): Custom container image to use

    Returns:
        Session: A sandbox session instance for the specified backend

    Raises:
        MissingDependencyError: If the required dependency for the chosen backend is not installed
        UnsupportedBackendError: If the chosen backend is not supported

    Examples:
        Python session with package installation:
        ```python
        with SandboxSession(lang="python", keep_template=True, verbose=True) as session:
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
        with SandboxSession(lang="java", keep_template=True, verbose=True) as session:
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
        with SandboxSession(lang="javascript", keep_template=True, verbose=True) as session:
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
        with SandboxSession(lang="cpp", keep_template=True, verbose=True) as session:
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
        with SandboxSession(lang="go", keep_template=True, verbose=True) as session:
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
    # Check if required dependency is installed
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
        keep_template: bool = False,
        commit_container: bool = False,
        verbose: bool = False,
        runtime_configs: dict | None = None,
        workdir: str | None = "/sandbox",
        enable_plotting: bool = True,
        security_policy: SecurityPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes an artifact sandbox session for secure code execution with artifact extraction.
        
        Creates a sandboxed environment using the specified container backend and programming language, enabling the capture of artifacts such as plots or images generated during code execution. Supports Docker, Kubernetes, and Podman backends, and allows configuration of container image, working directory, security policy, and plot extraction.
        
        Raises:
            MissingDependencyError: If the required dependency for the chosen backend is not installed.
            UnsupportedBackendError: If the chosen backend is not supported.
        
        Examples:
            Create a session with Docker and extract plots:
                with ArtifactSandboxSession(
                    lang="python",
                    image="ghcr.io/vndee/sandbox-python-311-bullseye",
                    backend=SandboxBackend.DOCKER
                ) as session:
                    result = session.run(code)
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
            security_policy=security_policy,
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
        """
        Executes code in the sandbox and extracts generated artifacts such as plots.
        
        Runs the provided code in an isolated session, optionally installing additional libraries, and captures any artifacts produced during execution (e.g., plots from matplotlib, seaborn, or plotly). If plotting is enabled but not supported by the language, raises a LanguageNotSupportPlotError.
        
        Args:
            code: The code to execute, which may include plotting commands.
            libraries: Optional list of additional libraries to install before execution.
        
        Returns:
            An ExecutionResult containing the exit code, standard output, standard error, and a list of captured plots with their base64-encoded content and format.
        
        Raises:
            LanguageNotSupportPlotError: If plot extraction is requested for a language that does not support it.
        """
        # Check if plotting is enabled and language supports it
        if self.enable_plotting and not self._session.language_handler.is_support_plot_detection:
            raise LanguageNotSupportPlotError(self._session.language_handler.name)

        # Delegate to language handler for language-specific artifact extraction
        result, plots = self._session.language_handler.run_with_artifacts(
            container=self._session,
            code=code,
            libraries=libraries,
            enable_plotting=self.enable_plotting,
            output_dir="/tmp/sandbox_plots",
        )

        return ExecutionResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            plots=plots,
        )


SandboxSession = create_session
