from llm_sandbox.docker import SandboxDockerSession
from llm_sandbox.kubernetes import SandboxKubernetesSession


class SandboxSession:
    def __new__(cls, use_kubernetes: bool = False, *args, **kwargs):
        if use_kubernetes:
            return SandboxKubernetesSession(*args, **kwargs)

        return SandboxDockerSession(*args, **kwargs)
