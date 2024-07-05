import docker
import docker.errors

from typing import List
from docker import DockerClient
from docker.models.images import Image


def image_exists(client: DockerClient, image: str) -> bool:
    """
    Check if a Docker image exists
    :param client: Docker client
    :param image: Docker image
    :return: True if the image exists, False otherwise
    """
    try:
        client.images.get(image)
        return True
    except docker.errors.ImageNotFound:
        return False
    except Exception as e:
        raise e


def pull_image(client: DockerClient, image: str, tag: str = "latest") -> Image:
    """
    Pull a Docker image
    :param client: Docker client
    :param image: Docker image
    :param tag: Tag for the image, default is "latest"
    """
    image = client.images.pull(image, tag)
    return image


def build_image(client: DockerClient, path: str | None = None, tag: str = "llm-sandbox", dockerfile: str | None = None) -> Image:
    """
    Build a Docker image
    :param client: Docker client
    :param path: Path to the directory containing the Dockerfile
    :param tag: Tag for the image
    :param dockerfile: Path to the Dockerfile, default is None
    """
    image = client.images.build(path=path, dockerfile=dockerfile, tag=tag)
    return image[0]
