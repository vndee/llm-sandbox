import io
import os
import tempfile
import time
import uuid
import tarfile
from pathlib import Path
from typing import List, Optional

from kubernetes import client as k8s_client, config
from kubernetes.stream import stream
from llm_sandbox.base import Session, ConsoleOutput
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
        dockerfile: Optional[str] = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        verbose: bool = False,
        kube_namespace: Optional[str] = "default",
        env_vars: Optional[dict] = None,
        pod_manifest: Optional[dict] = None,
        **kwargs,
    ):
        """
        Create a new sandbox session
        :param client: Kubernetes client, if not provided, a new client will be created based on local Kubernetes context
        :param image: Docker image to use
        :param dockerfile: Path to the Dockerfile, if image is not provided
        :param lang: Language of the code
        :param keep_template: if True, the image and container will not be removed after the session ends
        :param verbose: if True, print messages
        :param kube_namespace: Kubernetes namespace to use, default is 'default'
        :param env_vars: Environment variables to use
        :param pod_manifest: Pod manifest to use (ignores other settings: `image`, `kube_namespace` and `env_vars`)
        """
        super().__init__(lang, verbose)
        if lang not in SupportedLanguageValues:
            raise ValueError(
                f"Language {lang} is not supported. Must be one of {SupportedLanguageValues}"
            )

        if not image:
            image = DefaultImage.__dict__[lang.upper()]

        if not client:
            if self.verbose:
                print("Using local Kubernetes context since client is not provided..")

            config.load_kube_config()
            self.client = k8s_client.CoreV1Api()
        else:
            self.client = client

        self.image = image
        self.kube_namespace = kube_namespace
        self.pod_name = f"sandbox-{lang.lower()}-{uuid.uuid4().hex}"
        self.keep_template = keep_template
        self.container = None
        self.env_vars = env_vars
        self.pod_manifest = pod_manifest or self._default_pod_manifest()
        self._reconfigure_with_pod_manifest()

    def _default_pod_manifest(self) -> dict:
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
        # Add environment variables if provided
        if self.env_vars:
            pod_manifest["spec"]["containers"][0]["env"] = [  # type: ignore[index]
                {"name": key, "value": value} for key, value in self.env_vars.items()
            ]
        return pod_manifest

    def _reconfigure_with_pod_manifest(self):
        self.pod_name = self.pod_manifest.get("metadata", {}).get("name", self.pod_name)
        self.kube_namespace = self.pod_manifest.get("metadata", {}).get(
            "namespace", self.kube_namespace
        )

    def open(self):
        self._create_kubernetes_pod()

    def _create_kubernetes_pod(self):
        self.client.create_namespaced_pod(
            namespace=self.kube_namespace, body=self.pod_manifest
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
        self.client.delete_namespaced_pod(
            name=self.pod_name,
            namespace=self.kube_namespace,
            body=k8s_client.V1DeleteOptions(),
        )

    def run(self, code: str, libraries: Optional[List] = None) -> ConsoleOutput:
        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before running code."
            )

        if libraries:
            if self.lang == SupportedLanguage.GO:
                self.execute_command("mkdir -p /example")
                self.execute_command("go mod init example", workdir="/example")
                self.execute_command("go mod tidy", workdir="/example")

                for library in libraries:
                    install_command = get_libraries_installation_command(
                        self.lang, library
                    )
                    output = self.execute_command(install_command, workdir="/example")
                    if output.exit_code != 0:
                        raise RuntimeError(
                            f"Failed to install library {library}: {output}"
                        )
            else:
                for library in libraries:
                    install_command = get_libraries_installation_command(
                        self.lang, library
                    )
                    output = self.execute_command(install_command)
                    if output.exit_code != 0:
                        raise RuntimeError(
                            f"Failed to install library {library}: {output}"
                        )

        code_file_name = f"code.{get_code_file_extension(self.lang)}"
        if self.lang == SupportedLanguage.GO:
            code_dest_file = "/example/code.go"
        else:
            code_dest_file = f"/tmp/{code_file_name}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            code_file = Path(tmp_dir) / code_file_name
            with open(code_file, "w") as f:
                f.write(code)
            self.copy_to_runtime(str(code_file), code_dest_file)

        commands = get_code_execution_command(self.lang, code_dest_file)

        output = ConsoleOutput(exit_code=0, text="")
        for command in commands:
            if self.lang == SupportedLanguage.GO:
                output = self.execute_command(command, workdir="/example")
            else:
                output = self.execute_command(command)

            if output.exit_code != 0:
                break

        return ConsoleOutput(output.exit_code, output.text)

    def copy_to_runtime(self, src: str, dest: str):
        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before copying files."
            )

        start_time = time.time()
        if self.verbose:
            print(f"Copying {src} to {self.container}:{dest}..")

        dest_dir = os.path.dirname(dest)
        dest_file = os.path.basename(dest)

        if dest_dir:
            self.execute_command(f"mkdir -p {dest_dir}")

        with open(src, "rb") as f:
            tarstream = io.BytesIO()
            with tarfile.open(fileobj=tarstream, mode="w") as tar:
                tarinfo = tarfile.TarInfo(name=dest_file)
                tarinfo.size = os.path.getsize(src)
                tar.addfile(tarinfo, f)
            tarstream.seek(0)

            exec_command = ["tar", "xvf", "-", "-C", dest_dir]
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
                resp.write_stdin(tarstream.read(4096))
            resp.close()

        end_time = time.time()
        if self.verbose:
            print(
                f"Copied {src} to {self.container}:{dest} in {end_time - start_time:.2f} seconds"
            )

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

    def execute_command(
        self, command: str, workdir: Optional[str] = None
    ) -> ConsoleOutput:
        if not self.container:
            raise RuntimeError(
                "Session is not open. Please call open() method before executing commands."
            )

        if self.verbose:
            print(f"Executing command: {command}")

        if workdir:
            exec_command = ["sh", "-c", f"cd {workdir} && {command}"]
        else:
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
            _preload_content=False,
        )

        output = ""
        if self.verbose:
            print("Output:", end=" ")

        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                chunk = resp.read_stdout()
                output += chunk
                if self.verbose:
                    print(chunk, end="")
            if resp.peek_stderr():
                chunk = resp.read_stderr()
                output += chunk
                if self.verbose:
                    print(chunk, end="")

        exit_code = resp.returncode
        return ConsoleOutput(exit_code=exit_code, text=output)
