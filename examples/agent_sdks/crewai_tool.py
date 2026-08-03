# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for CrewAI.

    pip install 'llm-sandbox[docker]' crewai

Verified against crewai 1.15.10. The `@tool` decorator takes the tool name as
its argument; the docstring becomes the description the model sees.
"""

import logging

from crewai import Agent, Crew, Task
from crewai.tools import tool
from docker.errors import DockerException

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


@tool("Execute Python")
def execute_python(code: str) -> str:
    """Execute Python code in an isolated container and return its stdout.

    Use this for calculations and data manipulation. Always print the result.
    """
    return run_python(code)


analyst = Agent(
    role="Data Analyst",
    goal="Answer quantitative questions by writing and running Python",
    backstory="You prefer computing an answer over estimating one.",
    tools=[execute_python],
)

task = Task(
    description="Compute the mean and standard deviation of the first 100 primes.",
    expected_output="The two numbers, with the code used to obtain them.",
    agent=analyst,
)

if __name__ == "__main__":
    print(Crew(agents=[analyst], tasks=[task]).kickoff())
