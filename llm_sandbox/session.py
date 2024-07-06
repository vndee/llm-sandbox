import os
import docker
from typing import List, Optional, Union

from docker.models.images import Image
from llm_sandbox.utils import image_exists
from llm_sandbox.const import SupportedLanguage, SupportedLanguageValues


class SandboxSession:
    def __init__(
        self,
        image: Optional[str] = None,
        dockerfile: Optional[str] = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        verbose: bool = True,
    ):
        """
        Create a new sandbox session
        :param image: Docker image to use
        :param dockerfile: Path to the Dockerfile, if image is not provided
        :param lang: Language of the code
        :param keep_template: if True, the image and container will not be removed after the session ends
        :param verbose: if True, print messages
        """
        if image and dockerfile:
            raise ValueError("Only one of image or dockerfile should be provided")

        if not image and not dockerfile:
            raise ValueError("Either image or dockerfile should be provided")

        if lang not in SupportedLanguageValues:
            raise ValueError(
                f"Language {lang} is not supported. Must be one of {SupportedLanguageValues}"
            )

        self.lang: str = lang
        self.client: docker.DockerClient = docker.from_env()
        self.image: Union[Image, str] = image
        self.dockerfile: Optional[str] = dockerfile
        self.container = None
        self.path = None
        self.keep_template = keep_template
        self.is_create_template: bool = False
        self.verbose = verbose

    def open(self):
        warning_str = (
            "Since the `keep_image` flag is set to True the image and container will not be removed after the session "
            "ends and remains for future use."
        )
        if self.dockerfile:
            self.path = os.path.dirname(self.dockerfile)
            if self.verbose:
                f_str = f"Building docker image from {self.dockerfile}"
                f_str = f"{f_str}. {warning_str}" if self.keep_template else f_str
                print(f_str)

            self.image, _ = self.client.images.build(
                path=self.path,
                dockerfile=os.path.basename(self.dockerfile),
                tag="sandbox",
            )
            self.is_create_template = True

        if isinstance(self.image, str):
            if not image_exists(self.client, self.image):
                if self.verbose:
                    f_str = f"Pulling image {self.image}"
                    f_str = f"{f_str}. {warning_str}" if self.keep_template else f_str
                    print(f_str)

                self.image = self.client.images.pull(self.image)
                self.is_create_template = True
            else:
                self.image = self.client.images.get(self.image)
                if self.verbose:
                    print(f"Using image {self.image.tags[-1]}")

        self.container = self.client.containers.run(self.image, detach=True, tty=True)

    def close(self):
        if self.container and not self.keep_template:
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

    def run(self, code: str, libraries: List = []):
        raise NotImplementedError

    def copy_from_runtime(self, path: str):
        raise NotImplementedError

    def copy_to_runtime(self, path: str):
        raise NotImplementedError

    def execute_command(self, command: str, shell: str = "/bin/sh"):
        raise NotImplementedError

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    with SandboxSession(
        dockerfile="tests/busybox.Dockerfile", keep_template=False, lang="python"
    ) as session:
        session.run("print('Hello, World!')")
