import docker
import docker.errors

from typing import List
from docker import DockerClient


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


def pull_image(client: DockerClient, image: str) -> List[str]:
    """
    Pull a Docker image
    :param client: Docker client
    :param image: Docker image
    """
    try:
        image = client.images.pull(image)
        return image.tags
    except Exception as e:
        raise e


def build_image(client: DockerClient, path: str, tag: str) -> List:
    """
    Build a Docker image
    :param client: Docker client
    :param path: Path to the Dockerfile
    :param tag: Tag for the image
    """
    try:
        return client.images.build(path=path, tag=tag)
    except Exception as e:
        raise e


def create_container(client: DockerClient, image: str, name: str) -> str:
    """
    Create a Docker container
    :param client: Docker client
    :param image: Docker image
    :param name: Name of the container
    :return: Docker container id
    """
    try:
        container = client.containers.create(image, name=name)
        return container.id
    except Exception as e:
        raise e


def start_container(client: DockerClient, container_id: str):
    """
    Start a Docker container
    :param client: Docker client
    :param container_id: Docker container id
    """
    try:
        container = client.containers.get(container_id)
        container.start()
    except Exception as e:
        raise e


def stop_container(client: DockerClient, container_id: str):
    """
    Stop a Docker container
    :param client: Docker client
    :param container_id: Docker container id
    """
    try:
        container = client.containers.get(container_id)
        container.stop()
    except Exception as e:
        raise e


def remove_container(client: DockerClient, container_id: str):
    """
    Remove a Docker container
    :param client: Docker client
    :param container_id: Docker container id
    """
    try:
        container = client.containers.get(container_id)
        container.remove()
    except Exception as e:
        raise e


def remove_image(client: DockerClient, image: str):
    """
    Remove a Docker image
    :param client: Docker client
    :param image: Docker image
    """
    try:
        client.images.remove(image)
    except Exception as e:
        raise e


def get_container_logs(client: DockerClient, container_id: str) -> str:
    """
    Get the logs of a Docker container
    :param client: Docker client
    :param container_id: Docker container id
    :return: Logs of the container
    """
    try:
        container = client.containers.get(container_id)
        return container.logs()
    except Exception as e:
        raise e
