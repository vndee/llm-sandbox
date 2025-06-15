# ruff: noqa: E501

from typing import TYPE_CHECKING, Any

from podman import PodmanClient

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.docker import DockerContainerAPI, SandboxDockerSession
from llm_sandbox.security import SecurityPolicy

if TYPE_CHECKING:
    from podman.domain.images import Image


class PodmanContainerAPI(DockerContainerAPI):
    """Podman implementation of the ContainerAPI protocol."""

    def __init__(self, client: PodmanClient, stream: bool = True) -> None:
        """Initialize Podman container API."""
        self.client = client
        self.stream = stream

    def execute_command(self, container: Any, command: str, **kwargs: Any) -> tuple[int, Any]:
        """Execute command in Podman container."""
        workdir = kwargs.get("workdir")
        exec_kwargs: dict[str, Any] = {
            "cmd": command,
            "stream": self.stream,
            "tty": False,
            "stderr": True,
            "stdout": True,
            "demux": True,
        }
        if workdir:
            exec_kwargs["workdir"] = workdir

        result = container.exec_run(**exec_kwargs)
        # Podman returns (exit_code, output) tuple, Docker returns object with .exit_code and .output
        exit_code, output = result
        return exit_code or 0, output


class SandboxPodmanSession(SandboxDockerSession):
    r"""Sandbox session implemented using Podman containers.

    This class provides a sandboxed environment for code execution by leveraging Podman.
    It inherits from SandboxDockerSession since Podman is designed to be Docker-compatible,
    only overriding the differences in client initialization and API behavior.
    """

    def __init__(
        self,  # NOSONAR
        client: PodmanClient | None = None,
        image: str | None = None,
        dockerfile: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        commit_container: bool = False,
        verbose: bool = False,
        mounts: list | None = None,
        stream: bool = False,
        runtime_configs: dict | None = None,
        workdir: str | None = "/sandbox",
        security_policy: SecurityPolicy | None = None,
        default_timeout: float | None = None,
        execution_timeout: float | None = None,
        session_timeout: float | None = None,
        container_id: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        r"""Initialize Podman session.

        Args:
            client (PodmanClient | None): The Podman client to use.
            image (str | None): The image to use.
            dockerfile (str | None): The Dockerfile to use.
            lang (str): The language to use.
            keep_template (bool): Whether to keep the template image.
            commit_container (bool): Whether to commit the container to a new image.
            verbose (bool): Whether to enable verbose output.
            mounts (list | None): The mounts to use.
            stream (bool): Whether to stream the output.
            runtime_configs (dict | None): The runtime configurations to use.
            workdir (str | None): The working directory to use.
            security_policy (SecurityPolicy | None): The security policy to use.
            default_timeout (float | None): The default timeout to use.
            execution_timeout (float | None): The execution timeout to use.
            session_timeout (float | None): The session timeout to use.
            container_id (str | None): ID of existing container to connect to.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        config = SessionConfig(
            image=image,
            dockerfile=dockerfile,
            lang=SupportedLanguage(lang.upper()),
            verbose=verbose,
            workdir=workdir or "/sandbox",
            runtime_configs=runtime_configs or {},
            security_policy=security_policy,
            default_timeout=default_timeout,
            execution_timeout=execution_timeout,
            session_timeout=session_timeout,
            container_id=container_id,
        )

        # Initialize BaseSession (skip Docker's __init__)
        from llm_sandbox.core.session_base import BaseSession

        BaseSession.__init__(self, config=config, **kwargs)

        if not client:
            self._log("Using local Podman context since client is not provided.")
            self.client = PodmanClient.from_env()
        else:
            self.client = client

        self.container_api = PodmanContainerAPI(self.client, stream)

        # Set other attributes
        self.docker_image: Image
        self.keep_template: bool = keep_template
        self.commit_container: bool = commit_container
        self.is_create_template: bool = False
        self.stream: bool = stream

        if mounts:
            import warnings

            warnings.warn(
                "The 'mounts' parameter is deprecated and will be removed in a future version. "
                "Put the mounts in 'runtime_configs' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.config.runtime_configs.setdefault("mounts", []).append(mounts)
