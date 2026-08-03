# ruff: noqa: INP001
"""Shared sandbox helper for the agent SDK examples.

Every example in this directory exposes the same capability to a different
agent framework: run a snippet of Python in an isolated container and return
what it printed. Keeping that logic here means each example file shows only
the part that differs -- how *that* SDK wants a tool declared.
"""

from __future__ import annotations

import logging

from docker.errors import DockerException

from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import SandboxError

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0

# Code reaching this helper was written by a language model and is untrusted by
# construction -- any prompt-injected content the agent read can steer it. These
# controls are enforced by the container runtime, which matters because
# llm-sandbox's SecurityPolicy is *advisory*: session.is_safe(code) returns a
# verdict, and session.run() executes regardless of it.
#
# Two controls you will see recommended elsewhere are deliberately absent,
# because they break llm-sandbox's code-copy step (verified against 0.3.43):
#   read_only=True     -> Docker rejects put_archive with "container rootfs is
#                         marked read-only", even with a tmpfs on the workdir.
#   cap_drop=["ALL"]   -> drops DAC_OVERRIDE, so the copied file becomes
#                         unreadable: "[Errno 13] Permission denied".
SANDBOX_RUNTIME: dict[str, object] = {
    "network_mode": "none",  # no egress: injected code cannot exfiltrate or fetch a second stage
    "mem_limit": "512m",
    "pids_limit": 128,  # bounds fork bombs
    "security_opt": ["no-new-privileges:true"],
}


def run_python(code: str, libraries: list[str] | None = None) -> str:
    """Execute Python source in a hardened container and return its output.

    Args:
        code: Python source to execute.
        libraries: Optional packages to install first. Requires network access,
            so it does not work under the default `network_mode="none"`; pre-bake
            packages into a custom image instead.

    Returns:
        stdout on success. On a non-zero exit, the exit code followed by stderr
        (or stdout when stderr is empty). Agents cope better with a returned
        error string than with an exception, which most frameworks surface as a
        hard tool failure.

    """
    try:
        with SandboxSession(
            lang="python",
            verbose=False,
            # Without this the image is removed on close and re-pulled on the
            # next call -- roughly 1.6 GB per agent step.
            keep_template=True,
            runtime_configs=SANDBOX_RUNTIME,
        ) as session:
            result = session.run(code, libraries=libraries, timeout=DEFAULT_TIMEOUT)
            exit_code, stdout, stderr = result.exit_code, result.stdout, result.stderr
    except (SandboxError, DockerException):
        # Logged host-side rather than returned: the exception text can carry the
        # DOCKER_HOST socket path or registry URLs, which is host reconnaissance
        # for a model that may be under injection control.
        logger.exception("sandbox execution failed")
        return "sandbox error: execution environment unavailable"

    if exit_code != 0:
        return f"exit {exit_code}\n{stderr or stdout}".strip()
    return stdout.strip() or "(no output)"


# Describes the capability without asserting a security guarantee. The isolation
# a caller actually gets depends on SANDBOX_RUNTIME above, not on this sentence.
TOOL_DESCRIPTION = (
    "Execute Python code in an isolated container and return whatever it prints "
    "to stdout. Use this for calculations, data manipulation, and anything that "
    "is easier to compute than to reason about. Always print the result."
)
