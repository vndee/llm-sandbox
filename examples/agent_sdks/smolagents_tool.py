# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for smolagents.

    pip install 'llm-sandbox[docker]' smolagents

Verified against smolagents 1.26.0. Worth noting that smolagents' own CodeAgent
writes and runs Python itself; pointing that execution at a container instead of
the host is exactly what this tool is for.
"""

from _sandbox import TOOL_DESCRIPTION, run_python
from smolagents import CodeAgent, InferenceClientModel, tool


@tool
def execute_python(code: str) -> str:
    """Execute Python code in a secure sandboxed container and return its stdout.

    Args:
        code: The Python source to execute. Print anything you want returned.

    """
    return run_python(code)


execute_python.description = TOOL_DESCRIPTION

agent = CodeAgent(tools=[execute_python], model=InferenceClientModel())

if __name__ == "__main__":
    print(agent.run("Compute the sum of squares from 1 to 100 by running Python."))
