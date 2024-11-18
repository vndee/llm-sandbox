import docker
from typing import Optional, Union
from kubernetes import client as k8s_client
from llm_sandbox.const import SupportedLanguage
from llm_sandbox.docker import SandboxDockerSession
from llm_sandbox.kubernetes import SandboxKubernetesSession


class SandboxSession:
    def __new__(
        cls,
        client: Union[docker.DockerClient, k8s_client.CoreV1Api] = None,
        image: Optional[str] = None,
        dockerfile: Optional[str] = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        commit_container: bool = True,
        verbose: bool = False,
        use_kubernetes: bool = False,
        kube_namespace: Optional[str] = "default",
        container_configs: Optional[dict] = None,
    ):
        """
        Create a new sandbox session
        :param client: Either Docker or Kubernetes client, if not provided, a new client will be created based on local context
        :param image: Docker image to use
        :param dockerfile: Path to the Dockerfile, if image is not provided
        :param lang: Language of the code
        :param keep_template: if True, the image and container will not be removed after the session ends
        :param commit_container: if True, the Docker container will be commited after the session ends
        :param verbose: if True, print messages (default is True)
        :param use_kubernetes: if True, use Kubernetes instead of Docker (default is False)
        :param kube_namespace: Kubernetes namespace to use (only if 'use_kubernetes' is True), default is 'default'
        :param container_configs: Additional configurations for the Docker container, i.e. resources limits (cpu_count, mem_limit), etc.
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
            commit_container=commit_container,
            verbose=verbose,
            container_configs=container_configs,
        )
