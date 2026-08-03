# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for Pydantic AI.

    pip install 'llm-sandbox[docker]' pydantic-ai

Verified against pydantic-ai 2.22.0. `tool_plain` is the right decorator here
because the sandbox does not need the run context; use `tool` if you want
`RunContext` for per-run dependencies such as a shared container pool.

The agent is built inside `build_agent()` rather than at module scope: Pydantic
AI resolves the model provider eagerly, so a module-level `Agent(...)` raises if
`OPENAI_API_KEY` is unset -- which would break `import` for anyone reading the
file without credentials configured.
"""

import logging

from docker.errors import DockerException
from pydantic_ai import Agent

from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import SandboxError

logger = logging.getLogger(__name__)

# Code written by a model is untrusted by construction: anything the agent read
# can steer it. These controls are enforced by the container runtime, unlike
# llm-sandbox's SecurityPolicy, which is advisory -- session.is_safe() returns a
# verdict and session.run() executes regardless of it.
#
# read_only=True is deliberately absent: Docker rejects the code copy with
# "container rootfs is marked read-only", with or without a tmpfs on the
# workdir. Verified against llm-sandbox 0.3.43.
SANDBOX_RUNTIME = {
    "network_mode": "none",  # no egress: injected code cannot exfiltrate or fetch a second stage
    "mem_limit": "512m",
    "pids_limit": 128,  # bounds fork bombs
    "cap_drop": ["ALL"],
    # DAC_OVERRIDE has to go back or the container cannot read the source file
    # llm-sandbox copies in. Everything else stays dropped: verified CapEff
    # 0000000000000002 inside the container.
    "cap_add": ["DAC_OVERRIDE"],
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


def execute_python(code: str) -> str:
    """Run Python code in a sandboxed container and return its stdout.

    Args:
        code: The Python source to execute. Print anything you want returned.

    """
    return run_python(code)


def build_agent(model: str = "openai:gpt-4o") -> Agent:
    """Construct an agent with the sandbox registered as a plain tool."""
    agent = Agent(
        model,
        system_prompt="You solve problems by writing and running Python. Always print results.",
    )
    agent.tool_plain(name="execute_python", description=TOOL_DESCRIPTION)(execute_python)
    return agent


if __name__ == "__main__":
    result = build_agent().run_sync("What is the standard deviation of the first 50 primes?")
    print(result.output)
