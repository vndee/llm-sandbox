import io
import os
import tarfile
import tempfile
from typing import List, Optional, Union
from collections.abc import Iterator

from podman import PodmanClient
from podman.errors import NotFound
from podman.domain.containers import Container
from podman.domain.images import Image
from llm_sandbox.utils import (
    get_libraries_installation_command,
    get_code_file_extension,
    get_code_execution_command,
)
from llm_sandbox.base import Session, ConsoleOutput
from llm_sandbox.const import (
    SupportedLanguage,
    SupportedLanguageValues,
    DefaultImage,
    NotSupportedLibraryInstallation,
)


class SandboxPodmanSession(Session):
    def __init__(
        self,
        client: Optional[PodmanClient] = None,
        image: Optional[str] = None,
        dockerfile: Optional[str] = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        commit_container: bool = True,
        verbose: bool = False,
        mounts: Optional[list] = None,
        runtime_configs: Optional[dict] = None,
    ):
        """
        Create a new sandbox session
        :param client: Podman client, if not provided, a new client will be created based on local podman context
        :param image: Podman image to use
        :param dockerfile: Path to the Dockerfile, if image is not provided
        :param lang: Language of the code
        :param keep_template: if True, the image and container will not be removed after the session ends
        :param commit_container: if True, the podman container will be commited after the session ends
        :param verbose: if True, print messages
        :param mounts: List of mounts to be mounted to the container
        :param runtime_configs: Additional configurations for the container, i.e. resources limits (cpu_count, mem_limit), etc.
        """
        super().__init__(lang, verbose)
        if image and dockerfile:
            raise ValueError("Only one of image or dockerfile should be provided")

        if lang not in SupportedLanguageValues:
            raise ValueError(
                f"Language {lang} is not supported. Must be one of {SupportedLanguageValues}"
            )

        if not image and not dockerfile:
            image = DefaultImage.__dict__[lang.upper()]

        self.lang: str = lang
        self.client: Optional[PodmanClient] = client or PodmanClient()
        self.image: Union[Image, str] = image
        self.dockerfile: Optional[str] = dockerfile
        self.container: Optional[Container] = None
        self.path = None
        self.keep_template = keep_template
        self.commit_container = commit_container
        self.is_create_template: bool = False
        self.verbose = verbose
        self.mounts = mounts
        self.runtime_configs = runtime_configs

    def open(self):
        warning_str = (
            "Since the `keep_template` flag is set to True, the Podman image will not be removed after the session ends "
            "and remains for future use."
        )

        # Build image if a Dockerfile is provided
        if self.dockerfile:
            self.path = os.path.dirname(self.dockerfile)
            if self.verbose:
                f_str = f"Building Podman image from {self.dockerfile}"
                f_str = f"{f_str}\n{warning_str}" if self.keep_template else f_str
                print(f_str)

            self.image, _ = self.client.images.build(
                path=self.path,
                dockerfile=os.path.basename(self.dockerfile),
                tag=f"sandbox-{self.lang.lower()}-{os.path.basename(self.path)}",
            )
            self.is_create_template = True

        # Check or pull the image
        if isinstance(self.image, str):
            try:
                # Try to get the image locally
                self.image = self.client.images.get(self.image)
                if self.verbose:
                    print(f"Using image {self.image.tags[-1]}")
            except NotFound:
                if self.verbose:
                    print(
                        f"Image {self.image} not found locally. Attempting to pull..."
                    )

                try:
                    # Attempt to pull the image
                    self.image = self.client.images.pull(self.image)
                    if self.verbose:
                        print(f"Successfully pulled image {self.image.tags[-1]}")
                    self.is_create_template = True
                except Exception as e:
                    raise RuntimeError(f"Failed to pull image {self.image}: {e}")

        # Ensure mounts is an iterable (empty list if None)
        mounts = self.mounts if self.mounts is not None else []

        # Create the container
        self.container = self.client.containers.create(
            image=self.image.id if isinstance(self.image, Image) else self.image,
            tty=True,
            mounts=mounts,  # Use the adjusted mounts
            **self.runtime_configs if self.runtime_configs else {},
        )
        self.container.start()

    def close(self):
        if self.container:
            if self.commit_container and isinstance(self.image, Image):
                # Extract repository and tag
                if self.image.tags:
                    full_tag = self.image.tags[-1]
                    if ":" in full_tag:
                        repository, tag = full_tag.rsplit(":", 1)
                    else:
                        repository = full_tag
                        tag = "latest"
                try:
                    # Commit the container with repository and tag
                    self.container.commit(repository=repository, tag=tag)
                    if self.verbose:
                        print(f"Committed container as image {repository}:{tag}")
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to commit container: {e}")
                    raise

            # Stop and remove the container
            self.container.stop()
            self.container.remove(force=True)
            self.container = None

        if self.is_create_template and not self.keep_template:
            # check if the image is used by any other container
            containers = self.client.containers.list(all=True)
            image_id = (
                self.image.id
                if isinstance(self.image, Image)
                else self.client.images.get(self.image).id
            )
            image_in_use = any(
                container.image.id == image_id for container in containers
            )

            if not image_in_use:
                if isinstance(self.image, str):
                    self.client.images.remove(self.image)
                elif isinstance(self.image, Image):
                    self.image.remove(force=True)
                else:
                    raise ValueError("Invalid image type")
            else:
                if self.verbose:
                    print(
                        f"Image {self.image.tags[-1]} is in use by other containers. Skipping removal.."
                    )

    def run(self, code: str, libraries: Optional[List] = None) -> ConsoleOutput:
        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before running code."
            )

        if libraries:
            if self.lang.upper() in NotSupportedLibraryInstallation:
                raise ValueError(
                    f"Library installation has not been supported for {self.lang} yet!"
                )
            if self.lang == SupportedLanguage.GO:
                self.execute_command("mkdir -p /example")
                self.execute_command("go mod init example", workdir="/example")
                self.execute_command("go mod tidy", workdir="/example")

                for library in libraries:
                    command = get_libraries_installation_command(self.lang, library)
                    _ = self.execute_command(command, workdir="/example")
            else:
                for library in libraries:
                    command = get_libraries_installation_command(self.lang, library)
                    _ = self.execute_command(command)
        with tempfile.TemporaryDirectory() as directory_name:
            code_file = os.path.join(
                directory_name, f"code.{get_code_file_extension(self.lang)}"
            )
            if self.lang == SupportedLanguage.GO:
                code_dest_file = "/example/code.go"
            else:
                code_dest_file = (
                    f"/tmp/code.{get_code_file_extension(self.lang)}"  # code_file
                )

            with open(code_file, "w") as f:
                f.write(code)

            self.copy_to_runtime(code_file, code_dest_file)

            output = ConsoleOutput(exit_code=0, text="")
            commands = get_code_execution_command(self.lang, code_dest_file)
            for command in commands:
                if self.lang == SupportedLanguage.GO:
                    output = self.execute_command(command, workdir="/example")
                else:
                    output = self.execute_command(command)

            return output

    def copy_from_runtime(self, src: str, dest: str):
        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before copying files."
            )

        if self.verbose:
            print(f"Copying {self.container.short_id}:{src} to {dest}..")

        bits, stat = self.container.get_archive(src)
        if stat["size"] == 0:
            raise FileNotFoundError(f"File {src} not found in the container")

        tarstream = io.BytesIO(b"".join(bits))
        with tarfile.open(fileobj=tarstream, mode="r") as tar:
            tar.extractall(os.path.dirname(dest))

    def copy_to_runtime(self, src: str, dest: str):
        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before copying files."
            )

        is_created_dir = False
        directory = os.path.dirname(dest)
        if directory and not self.container.exec_run(f"test -d {directory}")[0] == 0:
            self.container.exec_run(f"mkdir -p {directory}")
            is_created_dir = True

        if self.verbose:
            if is_created_dir:
                print(f"Creating directory {self.container.short_id}:{directory}")
            print(f"Copying {src} to {self.container.short_id}:{dest}..")

        tarstream = io.BytesIO()
        with tarfile.open(fileobj=tarstream, mode="w") as tar:
            tar.add(src, arcname=os.path.basename(src))

        tarstream.seek(0)
        self.container.put_archive(os.path.dirname(dest), tarstream)

    def execute_command(
        self, command: Optional[str], workdir: Optional[str] = None
    ) -> ConsoleOutput:
        if not command:
            raise ValueError("Command cannot be empty")

        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before executing commands."
            )

        if self.verbose:
            print(f"Executing command: {command}")

        if workdir:
            exit_code, exec_log = self.container.exec_run(
                command, stream=True, tty=True, workdir=workdir
            )
        else:
            exit_code, exec_log = self.container.exec_run(
                command, stream=True, tty=True
            )
        if isinstance(exec_log, Iterator):
            output = ""
            for chunk in exec_log:
                chunk_str = chunk.decode("utf-8")
                output += chunk_str
                if self.verbose:
                    print(chunk_str, end="")
        else:
            output = exec_log.decode("utf-8")

        if self.verbose:
            print(output)

        return ConsoleOutput(text=output, exit_code=exit_code)
