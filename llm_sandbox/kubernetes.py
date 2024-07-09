import time
from typing import List, Optional

from kubernetes import client as k8s_client, config
from kubernetes.stream import stream
from llm_sandbox.base import Session
from llm_sandbox.utils import (
    get_libraries_installation_command,
    get_code_file_extension,
    get_code_execution_command,
)
from llm_sandbox.const import SupportedLanguage, SupportedLanguageValues, DefaultImage


class SandboxKubernetesSession(Session):
    def __init__(
        self,
        client: Optional[k8s_client.CoreV1Api] = None,
        image: Optional[str] = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        verbose: bool = True,
        kube_namespace: str = "default",
    ):
        """
        Create a new sandbox session
        :param client: Kubernetes client, if not provided, a new client will be created based on local Kubernetes context
        :param image: Docker image to use
        :param lang: Language of the code
        :param keep_template: if True, the image and container will not be removed after the session ends
        :param verbose: if True, print messages
        :param kube_namespace: Kubernetes namespace to use, default is 'default'
        """
        super().__init__(lang, verbose)
        if lang not in SupportedLanguageValues:
            raise ValueError(
                f"Language {lang} is not supported. Must be one of {SupportedLanguageValues}"
            )

        if not image:
            image = DefaultImage.__dict__[lang.upper()]

        if not client:
            print("Using local Kubernetes context since client is not provided..")
            config.load_kube_config()
            self.client = k8s_client.CoreV1Api()
        else:
            self.client = client

        self.image = image
        self.kube_namespace = kube_namespace
        self.pod_name = f"sandbox-{lang.lower()}"
        self.keep_template = keep_template
        self.container = None

    def open(self):
        self._create_kubernetes_pod()

    def _create_kubernetes_pod(self):
        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self.pod_name,
                "namespace": self.kube_namespace,
                "labels": {"app": "sandbox"},
            },
            "spec": {
                "containers": [
                    {"name": "sandbox-container", "image": self.image, "tty": True}
                ]
            },
        }
        self.client.create_namespaced_pod(
            namespace=self.kube_namespace, body=pod_manifest
        )

        while True:
            pod = self.client.read_namespaced_pod(
                name=self.pod_name, namespace=self.kube_namespace
            )
            if pod.status.phase == "Running":
                break
            time.sleep(1)

        self.container = self.pod_name

    def close(self):
        self._delete_kubernetes_pod()

    def _delete_kubernetes_pod(self):
        if not self.keep_template:
            self.client.delete_namespaced_pod(
                name=self.pod_name,
                namespace=self.kube_namespace,
                body=k8s_client.V1DeleteOptions(),
            )

    def run(self, code: str, libraries: Optional[List] = None):
        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before running code."
            )

        if libraries:
            command = get_libraries_installation_command(self.lang, libraries)
            self.execute_command(command)

        code_file = f"/tmp/code.{get_code_file_extension(self.lang)}"
        with open(code_file, "w") as f:
            f.write(code)

        self.copy_to_runtime(code_file, code_file)
        commands = get_code_execution_command(self.lang, code_file)

        output = ""
        for command in commands:
            exit_code, output = self.execute_command(command)
            if exit_code != 0:
                break
        return exit_code, output

    def copy_to_runtime(self, src: str, dest: str):
        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before copying files."
            )

        if self.verbose:
            print(f"Copying {src} to {self.container}:{dest}..")

        with open(src, "rb") as f:
            exec_command = ["tar", "xvf", "-", "-C", dest]
            resp = stream(
                self.client.connect_get_namespaced_pod_exec,
                self.container,
                self.kube_namespace,
                command=exec_command,
                stderr=True,
                stdin=True,
                stdout=True,
                tty=False,
                _preload_content=False,
            )
            while resp.is_open():
                resp.update(timeout=1)
                if resp.peek_stdout():
                    print(resp.read_stdout())
                if resp.peek_stderr():
                    print(resp.read_stderr())
                resp.write_stdin(f.read())
            resp.close()

    def copy_from_runtime(self, src: str, dest: str):
        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before copying files."
            )

        if self.verbose:
            print(f"Copying {self.container}:{src} to {dest}..")

        exec_command = ["tar", "cf", "-", src]
        resp = stream(
            self.client.connect_get_namespaced_pod_exec,
            self.container,
            self.kube_namespace,
            command=exec_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
        )
        with open(dest, "wb") as f:
            while resp.is_open():
                resp.update(timeout=1)
                if resp.peek_stdout():
                    f.write(resp.read_stdout())
                if resp.peek_stderr():
                    print(resp.read_stderr())

    def execute_command(self, command: str):
        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before executing commands."
            )

        if self.verbose:
            print(f"Executing command: {command}")

        exec_command = ["/bin/sh", "-c", command]
        resp = stream(
            self.client.connect_get_namespaced_pod_exec,
            self.container,
            self.kube_namespace,
            command=exec_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
        )
        output = resp.read_stdout()
        exit_code = 0 if resp.returncode is None else resp.returncode
        return exit_code, output
