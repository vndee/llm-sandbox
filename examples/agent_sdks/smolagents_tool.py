# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for smolagents.

    pip install 'llm-sandbox[docker]' smolagents

Verified against smolagents 1.26.0.

This example deliberately uses `ToolCallingAgent` rather than `CodeAgent`.
`CodeAgent` writes Python and executes it *itself*, and its `executor_type`
defaults to `"local"` -- meaning on the host. Handing it a sandbox tool does not
change that: it would still run its own generated code outside the container,
which is the opposite of what this library is for. `ToolCallingAgent` only
invokes tools, so all execution goes through the sandbox.

If you specifically want `CodeAgent`, sandbox its executor rather than relying
on this tool: `CodeAgent(..., executor_type="docker")`.
"""

from _sandbox import TOOL_DESCRIPTION, run_python
from smolagents import InferenceClientModel, ToolCallingAgent, tool


@tool
def execute_python(code: str) -> str:
    """Execute Python code in a secure sandboxed container and return its stdout.

    Args:
        code: The Python source to execute. Print anything you want returned.

    """
    return run_python(code)


execute_python.description = TOOL_DESCRIPTION


def build_agent() -> ToolCallingAgent:
    """Construct the agent lazily, so importing this file needs no credentials."""
    return ToolCallingAgent(tools=[execute_python], model=InferenceClientModel())


if __name__ == "__main__":
    print(build_agent().run("Compute the sum of squares from 1 to 100 by running Python."))
