# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for DeepAgents.

    pip install 'llm-sandbox[docker]' deepagents

Verified against deepagents 0.7.1. `create_deep_agent` accepts LangChain tools
or plain callables, so the LangChain `@tool` decorator is the idiomatic way in.
DeepAgents spawns sub-agents that each get the tool, which is the case container
pooling is built for -- see `examples/pool_basic_demo.py`.
"""

import logging

from deepagents import create_deep_agent
from docker.errors import DockerException
from langchain_core.tools import tool

from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import SandboxError

logger = logging.getLogger(__name__)

# Code written by a model is untrusted by construction: anything the agent read
# can steer it. These controls are enforced by the container runtime, unlike
# llm-sandbox's SecurityPolicy, which is advisory -- session.is_safe() returns a
# verdict and session.run() executes regardless of it.
#
# read_only=True and cap_drop=["ALL"] are omitted on purpose: both break the
# code-copy step (verified against llm-sandbox 0.3.43). read_only makes Docker
# reject put_archive; cap_drop=ALL removes DAC_OVERRIDE so the copied file
# cannot be read.
SANDBOX_RUNTIME = {
    "network_mode": "none",  # no egress: injected code cannot exfiltrate or fetch a second stage
    "mem_limit": "512m",
    "pids_limit": 128,  # bounds fork bombs
    "security_opt": ["no-new-privileges:true"],
}

TOOL_DESCRIPTION = (
    "Execute Python code in an isolated container and return whatever it prints "
    "to stdout. Use this for calculations, data manipulation, and anything that "
    "is easier to compute than to reason about. Always print the result."
)


def run_python(code: str) -> str:
    """Run model-authored Python in a hardened container and return its stdout."""
    try:
        with SandboxSession(
            lang="python",
            verbose=False,
            # Without this the image is removed on close and re-pulled next
            # call -- roughly 1.6 GB per agent step.
            keep_template=True,
            runtime_configs=SANDBOX_RUNTIME,
        ) as session:
            result = session.run(code, timeout=30)
            exit_code, stdout, stderr = result.exit_code, result.stdout, result.stderr
    except (SandboxError, DockerException):
        # Logged host-side, not returned: the text can carry the DOCKER_HOST
        # socket path, which is reconnaissance for a model under injection.
        logger.exception("sandbox execution failed")
        return "sandbox error: execution environment unavailable"

    if exit_code != 0:
        return f"exit {exit_code}\n{stderr or stdout}".strip()
    return stdout.strip() or "(no output)"


@tool(description=TOOL_DESCRIPTION)
def execute_python(code: str) -> str:
    """Run Python in a sandboxed container and return stdout."""
    return run_python(code)


agent = create_deep_agent(
    tools=[execute_python],
    system_prompt="You solve problems by writing and running Python. Always print results.",
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "Sum the primes below 1000."}]})
    print(result["messages"][-1].content)
