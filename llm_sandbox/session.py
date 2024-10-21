import docker
from typing import Optional, Union
from kubernetes import client as k8s_client
from llm_sandbox.const import SupportedLanguage
from llm_sandbox.docker import SandboxDockerSession, PythonInteractiveSandboxDockerSession
from llm_sandbox.kubernetes import SandboxKubernetesSession


class SandboxSession:
    def __new__(
        cls,
        client: Union[docker.DockerClient, k8s_client.CoreV1Api] = None,
        image: Optional[str] = None,
        dockerfile: Optional[str] = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        verbose: bool = False,
        use_kubernetes: bool = False,
        kube_namespace: Optional[str] = "default",
        **kwargs,
    ):
        """
        Create a new sandbox session
        :param client: Either Docker or Kubernetes client, if not provided, a new client will be created based on local context
        :param image: Docker image to use
        :param dockerfile: Path to the Dockerfile, if image is not provided
        :param lang: Language of the code
        :param keep_template: if True, the image and container will not be removed after the session ends
        :param verbose: if True, print messages (default is True)
        :param use_kubernetes: if True, use Kubernetes instead of Docker (default is False)
        :param kube_namespace: Kubernetes namespace to use (only if 'use_kubernetes' is True), default is 'default'
        """
        if use_kubernetes:
            return SandboxKubernetesSession(
                client=client,
                image=image,
                dockerfile=dockerfile,
                lang=lang,
                keep_template=keep_template,
                verbose=verbose,
                kube_namespace=kube_namespace,
            )

        return SandboxDockerSession(
            client=client,
            image=image,
            dockerfile=dockerfile,
            lang=lang,
            keep_template=keep_template,
            verbose=verbose,
        )


class PythonInteractiveSandboxSession:
    def __new__(
        cls,
        client: Union[docker.DockerClient, k8s_client.CoreV1Api] = None,
        image: Optional[str] = None,
        dockerfile: Optional[str] = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        verbose: bool = False,
        use_kubernetes: bool = False,
        kube_namespace: Optional[str] = "default",
    ):
        """
        Create a new sandbox session
        :param client: Either Docker or Kubernetes client, if not provided, a new client will be created based on local context
        :param image: Docker image to use
        :param dockerfile: Path to the Dockerfile, if image is not provided
        :param lang: Language of the code
        :param keep_template: if True, the image and container will not be removed after the session ends
        :param verbose: if True, print messages (default is True)
        :param use_kubernetes: if True, use Kubernetes instead of Docker (default is False)
        :param kube_namespace: Kubernetes namespace to use (only if 'use_kubernetes' is True), default is 'default'
        """
        if use_kubernetes:
            return SandboxKubernetesSession(
                client=client,
                image=image,
                dockerfile=dockerfile,
                lang=lang,
                keep_template=keep_template,
                verbose=verbose,
                kube_namespace=kube_namespace,
            )

        return PythonInteractiveSandboxDockerSession(
            client=client,
            image=image,
            dockerfile=dockerfile,
            lang=lang,
            keep_template=keep_template,
            verbose=verbose,
        )
