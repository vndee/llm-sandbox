# Agent SDK integrations

One example per framework, each exposing the same capability — run Python in an
isolated container and return what it printed. The sandbox logic lives in
[`_sandbox.py`](_sandbox.py) so each file shows only what differs: how that SDK
wants a tool declared.

| Framework | Example | Verified against |
| --- | --- | --- |
| OpenAI Agents SDK | [`openai_agents_tool.py`](openai_agents_tool.py) | openai-agents 0.19.2 |
| Claude Agent SDK | [`claude_agent_sdk_tool.py`](claude_agent_sdk_tool.py) | claude-agent-sdk 0.2.128 |
| LangChain | [`langchain_tool.py`](langchain_tool.py) | langchain 1.3.14 |
| LangGraph / DeepAgents | [`deepagents_tool.py`](deepagents_tool.py) | deepagents 0.7.1 |
| LlamaIndex | [`llamaindex_tool.py`](llamaindex_tool.py) | llama-index-core 0.14.23 |
| Google ADK | [`google_adk_tool.py`](google_adk_tool.py) | google-adk 2.6.1 |
| CrewAI | [`crewai_tool.py`](crewai_tool.py) | crewai 1.15.10 |
| Pydantic AI | [`pydantic_ai_tool.py`](pydantic_ai_tool.py) | pydantic-ai 2.22.0 |
| smolagents | [`smolagents_tool.py`](smolagents_tool.py) | smolagents 1.26.0 |
| Strands Agents | [`strands_tool.py`](strands_tool.py) | strands-agents 1.50.2 |
| AG2 | [`ag2_tool.py`](ag2_tool.py) | ag2 1.0.1 |

Each example was import-checked against the version listed at the time of writing
(there is no CI job doing this yet). Versions are stated
because these APIs move: LangChain 1.0 removed `AgentExecutor` and
`langchain.hub`, LlamaIndex dropped `FunctionCallingAgentWorker`, and AG2 1.0
replaced `ConversableAgent`/`register_function` outright. If an example fails
against a newer release, the version column tells you where the drift started.

## Running one

```bash
pip install 'llm-sandbox[docker]' openai-agents   # swap for your framework
cd examples/agent_sdks
export OPENAI_API_KEY=...                          # or the relevant provider key
python openai_agents_tool.py
```

Each file needs Docker running and the model provider credentials that its
framework expects.

## Two things worth carrying into real use

**Pool the containers.** These examples create a container per call, which is
fine for a demo and wrong for an agent loop — container creation dominates
per-call cost. See [`../pool_basic_demo.py`](../pool_basic_demo.py).

**These examples run untrusted code — that is the whole point.** Anything the
agent reads can steer what gets executed, so `_sandbox.py` applies container
controls that the runtime actually enforces: `network_mode="none"`,
`mem_limit`, `pids_limit` and `no-new-privileges`. Verified: egress is blocked.

Two controls you will see recommended elsewhere, including in this project's own
docs, **do not work** with llm-sandbox 0.3.43 and are deliberately omitted —
`read_only=True` makes Docker reject the code copy (`container rootfs is marked
read-only`), and `cap_drop=["ALL"]` drops `DAC_OVERRIDE` so the copied file
cannot be read.

**Security policies are advisory.** `session.is_safe(code)` returns a verdict;
it does not block execution, and `run()` executes code the policy flagged. If
you rely on it, check it yourself before calling `run()`. See
[the security guide](https://vndee.github.io/llm-sandbox/security/).
