# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for DeepAgents.

    pip install 'llm-sandbox[docker]' deepagents

Verified against deepagents 0.7.1. `create_deep_agent` accepts LangChain tools
or plain callables, so the LangChain `@tool` decorator is the idiomatic way in.
DeepAgents spawns sub-agents that each get the tool, which is the case container
pooling is built for -- see `examples/pool_basic_demo.py`.
"""

from _sandbox import TOOL_DESCRIPTION, run_python
from deepagents import create_deep_agent
from langchain_core.tools import tool


@tool(description=TOOL_DESCRIPTION)
def execute_python(code: str) -> str:
    """Run Python in a sandboxed container and return stdout."""
    return run_python(code)


agent = create_deep_agent(
    tools=[execute_python],
    system_prompt="You solve problems by writing and running Python. Always print results.",
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "Sum the primes below 1000."}]})
    print(result["messages"][-1].content)
