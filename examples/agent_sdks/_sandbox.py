# ruff: noqa: INP001
"""Shared sandbox helper for the agent SDK examples.

Every example in this directory exposes the same capability to a different
agent framework: run a snippet of Python in an isolated container and return
what it printed. Keeping that logic here means each example file shows only
the part that differs -- how *that* SDK wants a tool declared.

The helper is deliberately small. Real deployments will want a pool (see
``examples/pool_basic_demo.py``) so container creation is not on the hot path,
and a security policy if the code is genuinely untrusted.
"""

from __future__ import annotations

from llm_sandbox import SandboxSession

DEFAULT_TIMEOUT = 30.0


def run_python(code: str, libraries: list[str] | None = None) -> str:
    """Execute Python source in a container and return its combined output.

    Args:
        code: Python source to execute.
        libraries: Optional packages to install before running.

    Returns:
        stdout when the run succeeds, otherwise stderr prefixed with the exit
        code. Agents cope better with a returned error string than with an
        exception, which most frameworks surface as a hard tool failure.

    """
    try:
        with SandboxSession(lang="python", verbose=False) as session:
            result = session.run(code, libraries=libraries, timeout=DEFAULT_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - surface infra errors to the model
        return f"sandbox error: {type(exc).__name__}: {exc}"

    if result.exit_code != 0:
        return f"exit {result.exit_code}\n{result.stderr or result.stdout}".strip()
    return result.stdout.strip() or "(no output)"


TOOL_DESCRIPTION = (
    "Execute Python code in a secure sandboxed container and return whatever it "
    "prints to stdout. Use this for calculations, data manipulation, and anything "
    "that is easier to compute than to reason about. Always print the result."
)
