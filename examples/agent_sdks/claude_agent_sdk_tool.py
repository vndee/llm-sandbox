# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for the Claude Agent SDK.

    pip install 'llm-sandbox[docker]' claude-agent-sdk

Verified against claude-agent-sdk 0.2.128. This SDK takes tools as an in-process
MCP server built with `create_sdk_mcp_server`, so `tool()` needs an explicit
name, description and input schema rather than inferring them.
"""

import asyncio

from _sandbox import TOOL_DESCRIPTION, run_python
from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, query, tool


@tool("execute_python", TOOL_DESCRIPTION, {"code": str})
async def execute_python(args: dict) -> dict:
    """Run Python in a container and return stdout as a text content block."""
    output = run_python(args["code"])
    return {"content": [{"type": "text", "text": output}]}


sandbox_server = create_sdk_mcp_server(name="llm-sandbox", version="1.0.0", tools=[execute_python])

options = ClaudeAgentOptions(
    mcp_servers={"sandbox": sandbox_server},
    allowed_tools=["mcp__sandbox__execute_python"],
)


async def main() -> None:
    """Ask Claude to compute something by running code."""
    async for message in query(prompt="Use Python to compute the 20th Fibonacci number.", options=options):
        print(message)


if __name__ == "__main__":
    asyncio.run(main())
