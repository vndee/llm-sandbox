# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for the Google Agent Development Kit (ADK).

    pip install 'llm-sandbox[docker]' google-adk

Verified against google-adk 2.6.1. ADK derives the tool schema from the type
hints and docstring, so `FunctionTool` only needs the callable.
"""

from _sandbox import run_python
from google.adk.agents import Agent
from google.adk.tools import FunctionTool


def execute_python(code: str) -> str:
    """Execute Python code in a secure sandboxed container and return its stdout.

    Use this for calculations and data manipulation. Always print the result.

    Args:
        code: The Python source to execute.

    Returns:
        Whatever the snippet printed, or an error string if it failed.

    """
    return run_python(code)


root_agent = Agent(
    name="sandbox_analyst",
    model="gemini-2.0-flash",
    instruction="You solve problems by writing and running Python. Always print results.",
    tools=[FunctionTool(execute_python)],
)

if __name__ == "__main__":
    print("Run with: adk run google_adk_tool.py")
    print("Tools registered:", [t.name for t in root_agent.tools])
