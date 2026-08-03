# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for Strands Agents.

    pip install 'llm-sandbox[docker]' strands-agents

Verified against strands-agents 1.50.2. The `@tool` decorator infers the schema
from type hints and the docstring.
"""

from _sandbox import run_python
from strands import Agent, tool


@tool
def execute_python(code: str) -> str:
    """Execute Python code in a secure sandboxed container and return its stdout.

    Use this for calculations and data manipulation. Always print the result.

    Args:
        code: The Python source to execute.

    """
    return run_python(code)


agent = Agent(
    tools=[execute_python],
    system_prompt="You solve problems by writing and running Python. Always print results.",
)

if __name__ == "__main__":
    print(agent("What is the 30th Fibonacci number?"))
