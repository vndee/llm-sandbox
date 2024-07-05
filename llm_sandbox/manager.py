import os
import docker


class SandboxManager:
    def __init__(self):
        self.client = docker.from_env()

    def create_sandbox(self, image: str, name: str, dockerfile: str) -> str:
        """
        Create a new sandbox container
        :param image: Docker image to use
        :param name: Name of the container
        :param dockerfile: Path to the Dockerfile
        :return: Docker container id
        """

        # raise error if both the image and dockerfile exist since we only need one
        if image and dockerfile:
            raise ValueError("Only one of image or dockerfile should be provided")

        # raise error if neither the image nor dockerfile exist since we need one
        if not image and not dockerfile:
            raise ValueError("Either image or dockerfile should be provided")
