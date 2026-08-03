# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for the Claude Agent SDK.

    pip install 'llm-sandbox[docker]' claude-agent-sdk

Verified against claude-agent-sdk 0.2.128. This SDK takes tools as an in-process
MCP server built with `create_sdk_mcp_server`, so `tool()` needs an explicit
name, description and input schema rather than inferring them.
"""

import asyncio
import logging

from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query, tool
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


@tool("execute_python", TOOL_DESCRIPTION, {"code": str})
async def execute_python(args: dict) -> dict:
    """Run Python in a container and return stdout as a text content block."""
    # Offloaded to a thread: run_python blocks for as long as the container
    # takes, and this coroutine shares the loop that drains the SDK transport.
    output = await asyncio.to_thread(run_python, args["code"])
    return {"content": [{"type": "text", "text": output}]}


sandbox_server = create_sdk_mcp_server(name="llm-sandbox", version="1.0.0", tools=[execute_python])

options = ClaudeAgentOptions(
    mcp_servers={"sandbox": sandbox_server},
    # allowed_tools is an approval allowlist, not a restriction: it pre-approves
    # the sandbox tool but does not remove the SDK's built-in host-side tools.
    # Deny those explicitly so the container is the only execution path.
    allowed_tools=["mcp__sandbox__execute_python"],
    disallowed_tools=["Bash", "Read", "Write", "Edit"],
)


async def main() -> None:
    """Ask Claude to compute something by running code."""
    async for message in query(prompt="Use Python to compute the 20th Fibonacci number.", options=options):
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
