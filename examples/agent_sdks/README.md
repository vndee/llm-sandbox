# Agent SDK integrations

One example per framework, each exposing the same capability — run Python in an
isolated container and return what it printed.

Every file is **self-contained**: it shows the `SandboxSession` call and the
container hardening inline, then the part that actually differs — how that SDK
wants a tool declared. That means the sandbox block repeats across files. This
is deliberate: these are meant to be copied out whole, and a shared import would
leave a copy-paster with an `ImportError` and no idea what hardening they just
dropped.

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

Each example is checked by [`agent-sdk-smoke`](../../.github/workflows/agent-sdk-smoke.yml),
which runs weekly against the *current* release of each framework — not the pinned
one — so upstream drift surfaces here rather than in your traceback. It verifies
the module imports, drives `SandboxSession` directly, applies the hardening, and
exposes exactly one tool parameter (`code`). Versions are stated
because these APIs move: LangChain 1.0 removed `AgentExecutor` and
`langchain.hub`, LlamaIndex dropped `FunctionCallingAgentWorker`, and AG2 1.0
replaced `ConversableAgent`/`register_function` outright. If an example fails
against a newer release, the version column tells you where the drift started.

## Running one

```bash
# Pin to the verified version from the table above; the examples are
# checked against those releases, not against latest.
pip install 'llm-sandbox[docker]' 'openai-agents==0.19.2'
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
agent reads can steer what gets executed, so each file applies container
controls that the runtime actually enforces: `network_mode="none"`,
`mem_limit`, `pids_limit`, `cap_drop=["ALL"]` and `no-new-privileges`.
Verified inside the container: egress blocked, `CapEff` is `0000000000000002`.

`DAC_OVERRIDE` is added back deliberately — without it the container cannot read
the source file llm-sandbox copies in. `read_only=True` is omitted because Docker
rejects that copy outright against a read-only rootfs. See
[the security guide](https://vndee.github.io/llm-sandbox/security/) for both.

**Security policies are advisory.** `session.is_safe(code)` returns a verdict;
it does not block execution, and `run()` executes code the policy flagged. If
you rely on it, check it yourself before calling `run()`. See
[the security guide](https://vndee.github.io/llm-sandbox/security/).
