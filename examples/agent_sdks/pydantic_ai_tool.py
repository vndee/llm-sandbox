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

from _sandbox import TOOL_DESCRIPTION, run_python
from pydantic_ai import Agent


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
