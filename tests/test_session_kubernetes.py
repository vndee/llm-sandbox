import unittest
from unittest.mock import patch, MagicMock
from llm_sandbox.kubernetes import SandboxKubernetesSession


class TestSandboxKubernetesSession(unittest.TestCase):
    @patch("kubernetes.config.load_kube_config")
    def setUp(self, mock_kube_config):
        self.image = "python:3.9.19-bullseye"
        self.dockerfile = None
        self.lang = "python"
        self.keep_template = False
        self.verbose = False

        self.session = SandboxKubernetesSession(
            image=self.image,
            dockerfile=self.dockerfile,
            lang=self.lang,
            keep_template=self.keep_template,
            verbose=self.verbose,
        )

    @patch("kubernetes.config.load_kube_config")
    def test_with_pod_manifest(self, mock_kube_config):
        pod_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "test",
                "namespace": "test",
                "labels": {"app": "sandbox"},
            },
            "spec": {
                "containers": [
                    {
                        "name": "sandbox-container",
                        "image": "test",
                        "tty": True,
                        "volumeMounts": {
                            "name": "tmp",
                            "mountPath": "/tmp",
                        },
                    }
                ],
                "volumes": [{"name": "tmp", "emptyDir": {"sizeLimit": "5Gi"}}],
            },
        }
        self.session = SandboxKubernetesSession(
            image=self.image,
            dockerfile=self.dockerfile,
            lang=self.lang,
            keep_template=self.keep_template,
            verbose=self.verbose,
            pod_manifest=pod_manifest,
        )

        self.session.client = MagicMock()
        self.session.client.read_namespaced_pod.return_value.status.phase = "Running"
        self.session.open()

        self.session.client.create_namespaced_pod.assert_called_with(
            namespace="test",
            body=pod_manifest,
        )
