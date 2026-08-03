# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for AG2 (formerly AutoGen).

    pip install 'llm-sandbox[docker]' ag2

Verified against ag2 1.0.1. Note that 1.0 was a rewrite: the old
`autogen.ConversableAgent` / `register_function` pair is gone, replaced by
`ag2.Agent` and the `ag2.tool` decorator. Examples written against AutoGen 0.x
will not run here -- the system message is now the positional `prompt`
argument, and tools are attached with `tools=` or `add_tool()`.
"""

import ag2
from _sandbox import TOOL_DESCRIPTION, run_python


@ag2.tool(description=TOOL_DESCRIPTION)
def execute_python(code: str) -> str:
    """Run Python in a sandboxed container and return stdout."""
    return run_python(code)


agent = ag2.Agent(
    "analyst",
    "You solve problems by writing and running Python. Always print results.",
    tools=[execute_python],
)

if __name__ == "__main__":
    print(agent.run("What is the sum of the first 100 square numbers?"))
