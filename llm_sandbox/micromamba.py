from typing import Any

import docker
from docker.types import Mount

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.docker import ConsoleOutput, SandboxDockerSession


class MicromambaSession(SandboxDockerSession):
    r"""Extends `SandboxDockerSession` to execute commands within a Micromamba environment.

    This session leverages a Docker container (typically one with Micromamba pre-installed,
    like "mambaorg/micromamba:latest") and wraps executed commands with `micromamba run`
    to ensure they operate within a specified Micromamba environment.

    It inherits most of its functionality from `SandboxDockerSession`, overriding
    `execute_command` to inject the Micromamba activation.

    Reference: https://github.com/vndee/llm-sandbox/pull/3
    """

    def __init__(
        self,
        client: docker.DockerClient | None = None,
        image: str | None = "mambaorg/micromamba:latest",
        dockerfile: str | None = None,
        lang: str = SupportedLanguage.PYTHON,  # Language primarily for file extensions
        keep_template: bool = False,
        verbose: bool = False,
        mounts: list[Mount] | None = None,
        environment: str = "base",  # Name of the Micromamba environment
        # Allow SandboxDockerSession specific args to be passed via kwargs
        commit_container: bool = False,
        stream: bool = True,
        runtime_configs: dict | None = None,
        workdir: str | None = "/sandbox",
        **kwargs: dict[str, Any],  # For any other unforeseen or future parent args
    ) -> None:
        r"""Initialize a new Micromamba-enabled sandbox session.

        Args:
            client (docker.DockerClient | None, optional): An existing Docker client instance.
                If None, a new client is created from the local Docker environment. Defaults to None.
            image (str | None, optional): The Docker image to use, which should have Micromamba installed.
                Defaults to "mambaorg/micromamba:latest".
            dockerfile (str | None, optional): Path to a Dockerfile to build a custom image.
                The resulting image should have Micromamba. Defaults to None.
            lang (str, optional): The primary programming language. This mainly influences default file
                extensions for code execution via the `run` method inherited from `SandboxDockerSession`.
                Defaults to SupportedLanguage.PYTHON.
            keep_template (bool, optional): If True, the Docker image will not be removed after the
                session ends. Defaults to False.
            verbose (bool, optional): If True, print detailed log messages. Defaults to False.
            mounts (list[Mount] | None, optional): A list of Docker `Mount` objects to be mounted
                into the container. Defaults to None.
            environment (str, optional): The name of the Micromamba environment to activate and run
                commands within (e.g., "base", "my_env"). Defaults to "base".
            commit_container (bool, optional): If True, the Docker container's state will be committed
                to a new image after the session ends. Inherited from `SandboxDockerSession`. Defaults to False.
            stream (bool, optional): If True, the output from `execute_command` will be streamed.
                Inherited from `SandboxDockerSession`. Defaults to True.
            runtime_configs (dict | None, optional): Additional configurations for the container runtime.
                Inherited from `SandboxDockerSession`. Defaults to None.
            workdir (str | None, optional): The working directory inside the container.
                Inherited from `SandboxDockerSession`. Defaults to "/sandbox".
            **kwargs: Additional keyword arguments to pass to the `SandboxDockerSession` parent constructor.

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
            runtime_configs=runtime_configs,
            workdir=workdir,
            **kwargs,  # Pass any remaining kwargs
        )

        self.environment = environment

    def execute_command(self, command: str | None, workdir: str | None = None) -> ConsoleOutput:
        r"""Execute a command within the specified Micromamba environment in the Docker container.

        This method overrides the parent `execute_command`. It prepends `micromamba run -n <environment>`
        to the given command before execution, ensuring it runs inside the target Micromamba environment.

        Args:
            command (str | None): The command string to execute. If None or empty, behavior might
                                depend on the parent class or raise an error.
            workdir (str | None, optional): The working directory within the container where the
                                        command should be executed. Defaults to None (uses container's
                                        default or what's set in `SandboxDockerSession`).

        Returns:
            ConsoleOutput: An object containing the stdout, stderr, and exit code of the command.

        """
        # Ensure command is not None before formatting, though parent might handle None.
        # For clarity, one might add an explicit check for `command is None` here if specific handling is needed.
        full_command = f"micromamba run -n {self.environment} {command if command else ''}"
        return super().execute_command(full_command, workdir)
