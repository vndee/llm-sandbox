import docker
import docker.errors
from typing import List, Optional

from docker import DockerClient
from llm_sandbox.const import SupportedLanguage


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


def get_libraries_installation_command(
    lang: str, libraries: List[str]
) -> Optional[str]:
    """
    Get the command to install libraries for the given language
    :param lang: Programming language
    :param libraries: List of libraries
    :return: Installation command
    """
    if lang == SupportedLanguage.PYTHON:
        return f"pip install {' '.join(libraries)}"
    elif lang == SupportedLanguage.JAVA:
        return f"mvn install:install-file -Dfile={' '.join(libraries)}"
    elif lang == SupportedLanguage.JAVASCRIPT:
        return f"yarn add {' '.join(libraries)}"
    elif lang == SupportedLanguage.CPP:
        return f"apt-get install {' '.join(libraries)}"
    elif lang == SupportedLanguage.GO:
        return f"go get {' '.join(libraries)}"
    elif lang == SupportedLanguage.RUBY:
        return f"gem install {' '.join(libraries)}"
    else:
        raise ValueError(f"Language {lang} is not supported")


def get_code_file_extension(lang: str) -> str:
    """
    Get the file extension for the given language
    :param lang: Programming language
    :return: File extension
    """
    if lang == SupportedLanguage.PYTHON:
        return "py"
    elif lang == SupportedLanguage.JAVA:
        return "java"
    elif lang == SupportedLanguage.JAVASCRIPT:
        return "js"
    elif lang == SupportedLanguage.CPP:
        return "cpp"
    elif lang == SupportedLanguage.GO:
        return "go"
    elif lang == SupportedLanguage.RUBY:
        return "rb"
    else:
        raise ValueError(f"Language {lang} is not supported")


def get_code_execution_command(lang: str, code_file: str) -> str:
    """
    Get the command to execute the code
    :param lang: Programming language
    :param code_file: Path to the code file
    :return: Execution command
    """
    if lang == SupportedLanguage.PYTHON:
        return f"python {code_file}"
    elif lang == SupportedLanguage.JAVA:
        return f"java {code_file}"
    elif lang == SupportedLanguage.JAVASCRIPT:
        return f"node {code_file}"
    elif lang == SupportedLanguage.CPP:
        return f"./{code_file}"
    elif lang == SupportedLanguage.GO:
        return f"go run {code_file}"
    elif lang == SupportedLanguage.RUBY:
        return f"ruby {code_file}"
    else:
        raise ValueError(f"Language {lang} is not supported")
