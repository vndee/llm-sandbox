# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for LangChain.

    pip install 'llm-sandbox[docker]' langchain langchain-openai

Verified against langchain 1.3.14. LangChain 1.0 removed the agent stack this
integration used to be written against: `langchain.hub`, `AgentExecutor` and
`create_tool_calling_agent` are all gone, replaced by `create_agent`, which
returns a LangGraph graph you invoke directly.
"""

from typing import Any

from _sandbox import TOOL_DESCRIPTION, run_python
from langchain.agents import create_agent
from langchain_core.tools import tool


@tool(description=TOOL_DESCRIPTION)
def execute_python(code: str) -> str:
    """Run Python in a sandboxed container and return stdout."""
    return run_python(code)


def build_agent(model: str = "openai:gpt-4o") -> Any:
    """Construct an agent with the sandbox tool attached.

    Built lazily rather than at module scope because `create_agent` resolves the
    model provider immediately, so a module-level call raises without
    `OPENAI_API_KEY` set -- which would break `import` for anyone just reading
    the file.
    """
    return create_agent(
        model=model,
        tools=[execute_python],
        system_prompt="You solve problems by writing and running Python. Always print results.",
    )


if __name__ == "__main__":
    result = build_agent().invoke({"messages": [{"role": "user", "content": "Sum the primes below 1000."}]})
    print(result["messages"][-1].content)
