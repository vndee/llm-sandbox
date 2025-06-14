from typing import Any

import docker
from docker.types import Mount

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.docker import DockerContainerAPI, SandboxDockerSession
from llm_sandbox.security import SecurityPolicy


class MicromambaContainerAPI(DockerContainerAPI):
    """Micromamba implementation that wraps DockerContainerAPI."""

    def __init__(self, client: docker.DockerClient, environment: str = "base", stream: bool = True) -> None:
        """Initialize Micromamba container API."""
        super().__init__(client, stream)
        self.environment = environment

    def execute_command(self, container: Any, command: str, **kwargs: Any) -> tuple[int, Any]:
        """Execute command wrapped with micromamba run in Docker container."""
        # Wrap the command with micromamba run
        full_command = f"micromamba run -n {self.environment} {command}"
        return super().execute_command(container, full_command, **kwargs)


class MicromambaSession(SandboxDockerSession):
    r"""Extends `BaseSession` to execute commands within a Micromamba environment.

    This session leverages a Docker container (typically one with Micromamba pre-installed,
    like "mambaorg/micromamba:latest") and wraps executed commands with `micromamba run`
    to ensure they operate within a specified Micromamba environment.

    Reference: https://github.com/vndee/llm-sandbox/pull/3
    """

    def __init__(
        self,  # NOSONAR
        client: docker.DockerClient | None = None,
        image: str = "mambaorg/micromamba:latest",
        dockerfile: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        verbose: bool = False,
        mounts: list[Mount] | None = None,
        environment: str = "base",
        commit_container: bool = False,
        stream: bool = True,
        runtime_configs: dict | None = None,
        workdir: str = "/sandbox",
        security_policy: SecurityPolicy | None = None,
        default_timeout: float | None = None,
        execution_timeout: float | None = None,
        session_timeout: float | None = None,
        container_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        r"""Initialize a new Micromamba-enabled sandbox session.

        Args:
            client (docker.DockerClient | None, optional): An existing Docker client instance.
                If None, a new client is created from the local Docker environment. Defaults to None.
            image (str, optional): The Docker image to use, which should have Micromamba installed.
                Defaults to "mambaorg/micromamba:latest".
            dockerfile (str | None, optional): Path to a Dockerfile to build a custom image.
                The resulting image should have Micromamba. Defaults to None.
            lang (str, optional): The primary programming language. This mainly influences default file
                extensions for code execution. Defaults to SupportedLanguage.PYTHON.
            keep_template (bool, optional): If True, the Docker image will not be removed after the
                session ends. Defaults to False.
            verbose (bool, optional): If True, print detailed log messages. Defaults to False.
            mounts (list[Mount] | None, optional): A list of Docker `Mount` objects to be mounted
                into the container. Defaults to None.
            environment (str, optional): The name of the Micromamba environment to activate and run
                commands within (e.g., "base", "my_env"). Defaults to "base".
            commit_container (bool, optional): If True, the Docker container's state will be committed
                to a new image after the session ends. Defaults to False.
            stream (bool, optional): If True, the output from `execute_command` will be streamed.
                Defaults to True.
            runtime_configs (dict | None, optional): Additional configurations for the container runtime.
                Defaults to None.
            workdir (str, optional): The working directory inside the container.
                Defaults to "/sandbox".
            security_policy (SecurityPolicy | None, optional): The security policy to use for the session.
                Defaults to None.
            default_timeout (float | None, optional): The default timeout for the session.
                Defaults to None.
            execution_timeout (float | None, optional): The execution timeout for the session.
                Defaults to None.
            session_timeout (float | None, optional): The session timeout for the session.
                Defaults to None.
            container_id (str | None, optional): ID of existing container to connect to.
                Defaults to None.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(
            client=client,
            image=image,
            dockerfile=dockerfile,
            lang=lang,
            keep_template=keep_template,
            verbose=verbose,
            mounts=mounts,
            commit_container=commit_container,
            stream=stream,
            runtime_configs=runtime_configs or {},
            workdir=workdir,
            security_policy=security_policy,
            default_timeout=default_timeout,
            execution_timeout=execution_timeout,
            session_timeout=session_timeout,
            container_id=container_id,
            **kwargs,
        )

        self.environment = environment

        self.container_api = MicromambaContainerAPI(self.client, environment, stream)
