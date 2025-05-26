from typing import Any

import docker
from docker.types import Mount
from llm_sandbox.const import SupportedLanguage
from llm_sandbox.docker import ConsoleOutput, SandboxDockerSession


class MicromambaSession(SandboxDockerSession):
    """MicromambaSession extends SandboxDockerSession to allow activation of micromamba.

    Reference: https://github.com/vndee/llm-sandbox/pull/3

    This class is used to create a sandbox session that allows the execution of commands
    in a micromamba environment.
    """

    def __init__(
        self,
        client: docker.DockerClient | None = None,
        image: str | None = "mambaorg/micromamba:latest",
        dockerfile: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        verbose: bool = False,
        mounts: list[Mount] | None = None,
        environment: str = "base",
        **kwargs: dict[str, Any],
    ) -> None:
        """Create a new sandbox session.

        :param client: Docker client, if not provided, a new client will be created
                        based on local Docker context
        :param image: Docker image to use
        :param dockerfile: Path to the Dockerfile, if image is not provided
        :param lang: Language of the code
        :param keep_template: if True, the image and container will not be removed
                            after the session ends
        :param verbose: if True, print messages
        :param mounts: List of mounts to be mounted to the container
        :param environment: Name of the micromamba environment to use
        """
        super().__init__(
            client=client,
            image=image,
            dockerfile=dockerfile,
            lang=lang,
            keep_template=keep_template,
            verbose=verbose,
            mounts=mounts,
            **kwargs,
        )

        self.environment = environment

    def execute_command(self, command: str | None, workdir: str | None = None) -> ConsoleOutput:
        """Execute a command in the micromamba environment.

        :param command: Command to execute
        :param workdir: Working directory to execute the command in
        :return: ConsoleOutput object containing the output and exit code
        """
        command = f"micromamba run -n {self.environment} {command}"
        return super().execute_command(command, workdir)
