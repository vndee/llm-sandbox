# Why LLM Sandbox

You are building something that runs code an LLM wrote. You have three options:
write your own `docker run` wrapper, rent a hosted sandbox API, or use a library.
This page explains where LLM Sandbox fits, and — just as importantly — where it
doesn't.

## The one-sentence version

**LLM Sandbox is the sandbox that runs on your infrastructure, costs nothing per
execution, and doesn't ask you to send your code anywhere.**

```bash
pip install 'llm-sandbox[docker]'
```

```python
from llm_sandbox import SandboxSession

with SandboxSession(lang="python") as session:
    result = session.run("print('hello')")
    print(result.stdout)
```

That is the entire setup. No account, no API key, no egress.

## What you get that a `docker run` wrapper doesn't

Most teams start by shelling out to Docker themselves. It works until it doesn't.
The things that break, in roughly the order they break:

- **Getting artifacts back out.** Plots written inside the container vanish when it
  dies. LLM Sandbox captures matplotlib and ggplot2 output and hands it back as
  structured artifacts, base64-encoded and ready to render.
- **Installing dependencies.** `libraries=["numpy", "pandas"]` works across seven
  languages with the right package manager for each.
- **Cold-start latency.** Once you're executing code in an agent loop, container
  creation dominates. Container pooling pre-warms and recycles containers with
  configurable size bounds, idle timeouts, health checks, and exhaustion strategies.
- **Statefulness.** Agents often need notebook semantics — define a variable in one
  step, use it in the next. `InteractiveSandboxSession` keeps a live IPython kernel
  so state, imports, and magics survive across calls.
- **Cleanup semantics.** Timeouts, orphaned containers, and partial failures are
  handled once, in a library with tests, instead of in every project separately.

## What you get that a hosted sandbox doesn't

Hosted sandbox APIs are genuinely good products, and for many teams they're the
right call. What they cannot offer is these three things:

- **Your code stays on your infrastructure.** Nothing is transmitted to a third
  party. For regulated data, internal source code, or customer datasets, this is
  frequently not a preference but a requirement.
- **No per-execution cost.** Evaluation harnesses and batch pipelines that execute
  hundreds of thousands of programs have a cost profile that per-call pricing
  handles badly. Self-hosted execution costs whatever your compute costs.
- **No dependency on someone else's roadmap.** MIT-licensed, versioned on PyPI,
  pinnable. Reproducing a result two years from now means pinning a version
  number, not hoping an endpoint still behaves the same way.

## Where it runs

The same code runs on all three container backends — you change one argument:

| Backend        | Use it when                                                   |
| -------------- | ------------------------------------------------------------- |
| **Docker**     | Default. Laptops, single servers, CI.                         |
| **Podman**     | You want rootless containers.                                 |
| **Kubernetes** | You're scaling out, with custom pod manifests and namespaces. |

There's also `MicromambaSession`, which isn't a fourth backend — it's a
specialisation of the Docker one for Micromamba images, when you need
conda-managed scientific environments.

## Honest limitations

- **This is container isolation, not VM isolation.** LLM Sandbox inherits the
  threat model of the backend you choose, and containers share the host kernel —
  a kernel vulnerability is a sandbox-escape vulnerability. If you're executing
  deliberately adversarial code from untrusted third parties, pair it with a
  hardened runtime (gVisor, Kata) or use a microVM-based service. We'd rather say
  this plainly than let you discover it later.
- **You operate it.** No hosted control plane means no one else is patching your
  base images or watching your capacity.
- **Cold starts are yours to manage.** Pooling helps a great deal, but the first
  container still has to start.

## Who's using it

- **[SkyPilot](https://blog.skypilot.co/skypilot-llm-sandbox/)** — self-hosted
  agent sandbox on your own cloud, built on LLM Sandbox for multi-language
  execution.
- **[Code Sandbox MCP](https://www.philschmid.de/code-sandbox-mcp)** (Philipp
  Schmid) — an MCP code interpreter using LLM Sandbox for containerized execution.
- **[EffiBench-X](https://arxiv.org/abs/2505.13004)** (NeurIPS 2025) — efficiency
  benchmarking of LLM-generated code, which vendors LLM Sandbox into its
  evaluation harness and runs all six benchmark languages through it.
- **[Cloudfleet](https://cloudfleet.ai/tutorials/machine-learning/add-code-interpreter-into-your-llm-apps-with-llm-sandbox/)** —
  code interpreters on managed Kubernetes with on-demand GPU nodes.

The package has passed **3.9M downloads on PyPI**, with more than 600k in the
last 30 days (as of August 2026).

## Try it

```bash
pip install 'llm-sandbox[docker]'
```

→ [Getting Started](https://vndee.github.io/llm-sandbox/getting-started/) ·
[Security](https://vndee.github.io/llm-sandbox/security/) ·
[Backends](https://vndee.github.io/llm-sandbox/backends/)
