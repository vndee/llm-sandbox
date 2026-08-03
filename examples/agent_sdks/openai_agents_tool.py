# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for the OpenAI Agents SDK.

    pip install 'llm-sandbox[docker]' openai-agents

Verified against openai-agents 0.19.2. Note this is the Agents SDK, not the
older function-calling API -- `@function_tool` builds the JSON schema from the
type hints and docstring, so no hand-written schema is needed.
"""

import asyncio

from _sandbox import TOOL_DESCRIPTION, run_python
from agents import Agent, Runner, function_tool


@function_tool(description_override=TOOL_DESCRIPTION)
def execute_python(code: str) -> str:
    """Run Python code in a sandboxed container.

    Args:
        code: The Python source to execute. Print anything you want returned.

    """
    return run_python(code)


agent = Agent(
    name="Data analyst",
    instructions="You solve problems by writing and running Python. Always print results.",
    tools=[execute_python],
)


async def main() -> None:
    """Ask the agent something that requires actually running code."""
    result = await Runner.run(agent, "What are the first 10 Fibonacci numbers, and their sum?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
