# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for LlamaIndex.

    pip install 'llm-sandbox[docker]' llama-index

Verified against llama-index-core 0.14.23. `FunctionCallingAgentWorker`, which
this integration previously used, has been removed; `FunctionAgent` from
`llama_index.core.agent.workflow` is the replacement and is awaitable.
"""

import asyncio

from _sandbox import TOOL_DESCRIPTION, run_python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI


def execute_python(code: str) -> str:
    """Run Python in a sandboxed container and return stdout."""
    return run_python(code)


# Wrapped rather than passing run_python directly: FunctionTool derives the
# schema from the signature, so the raw function would expose `libraries` and
# let the model install arbitrary PyPI packages.
sandbox_tool = FunctionTool.from_defaults(
    fn=execute_python,
    name="execute_python",
    description=TOOL_DESCRIPTION,
)

agent = FunctionAgent(
    tools=[sandbox_tool],
    llm=OpenAI(model="gpt-4o"),
    system_prompt="You solve problems by writing and running Python. Always print results.",
)


async def main() -> None:
    """Ask the agent a question that needs real execution."""
    print(await agent.run("What is the standard deviation of the first 50 primes?"))


if __name__ == "__main__":
    asyncio.run(main())
