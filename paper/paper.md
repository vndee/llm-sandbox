---
title: "LLM Sandbox: A Portable, Self-Hosted Runtime for Securely Executing LLM-Generated Code"
tags:
  - Python
  - large language models
  - AI agents
  - code interpreter
  - sandboxing
  - containers
  - reproducibility
authors:
  - name: Duy V. Huynh
    orcid: 0009-0002-5001-7815
    affiliation: 1
affiliations:
  - name: University of Science, Vietnam National University Ho Chi Minh City, Vietnam
    index: 1
date: 2 August 2026
bibliography: paper.bib
---

# Summary

Large language models increasingly solve tasks by _writing and running code_ rather than
by answering directly. This shifts a burden onto the surrounding system: model-generated
code is untrusted by construction, yet it must be executed with enough fidelity — real
interpreters, real package installation, real plotting — for the result to be useful.

`LLM Sandbox` [@llmsandbox] is a Python library that provides this execution layer. A single
context-managed `SandboxSession` accepts a code string and returns its stdout, stderr,
exit code, and any generated artifacts, having run the code inside an isolated container.
The same API is served by three interchangeable container backends — Docker, Podman and
Kubernetes — so that a workflow prototyped on a laptop runs unchanged on a cluster. A
fourth session type specialises the Docker backend for Micromamba images, giving
conda-managed scientific environments without a separate backend implementation. Seven
languages are supported (Python, JavaScript, Java, C++, Go, R, Ruby) with automatic
dependency installation, and plots written by `matplotlib` or `ggplot2` are captured and
returned as structured artifacts rather than being lost inside the container.

Beyond the core session, the library provides configurable security policies that
statically screen code before execution, resource and network restrictions, container
pooling for low-latency reuse, notebook-style interactive sessions backed by a persistent
IPython kernel, and a Model Context Protocol [@mcp] server that exposes sandboxed
execution directly to MCP-compatible assistants.

# Statement of need

Research and engineering workflows that execute model-generated code currently choose
between two unsatisfying options. Writing bespoke `docker run` wrappers is the common
default, but such wrappers accumulate subtle correctness problems — artifact retrieval,
dependency installation, timeout and cleanup semantics, capability dropping — that are
solved repeatedly and inconsistently across projects. The alternative is a hosted
sandbox service, which removes that burden but introduces per-execution cost, network
dependence, and the requirement that potentially sensitive code and data leave the
user's infrastructure. Neither option serves work under data-governance constraints,
air-gapped evaluation, or high-volume batch execution where per-call pricing dominates.

`LLM Sandbox` targets this gap. It is installable with `pip`, runs entirely on
infrastructure the user already controls, and imposes no vendor per-execution fee. Its intended
audience is researchers building evaluation harnesses over generated code, and engineers
embedding code execution into agent systems under privacy or cost constraints.

This combination matters most for reproducible evaluation. Benchmarks that measure
properties of generated code must execute large volumes of untrusted programs under
conditions other researchers can recreate exactly. A self-hosted, versioned, MIT-licensed
runtime can be pinned and re-run years later; a commercial endpoint whose behaviour may
change, or be withdrawn, cannot.

# State of the field

Several systems address sandboxed execution of model-generated code, and they differ from
`LLM Sandbox` primarily in *architecture* rather than in feature coverage.

`E2B` [@e2b] and `microsandbox` [@microsandbox] both execute code inside microVMs —
Firecracker and libkrun respectively — providing hardware-level isolation stronger than
containers. `Daytona` [@daytona] provides elastic container-based infrastructure for agent
workloads. Self-hosting any of these means adopting and operating its execution substrate:
a microVM runtime, a daemon, or a control plane. `E2B` and `Daytona` additionally offer
hosted services, which reintroduce the per-execution cost and data-egress properties
described above. `Daytona`'s public repository has moreover stated since June 2026 that it
is no longer maintained, with development moved to a private codebase — an illustration of
the roadmap dependency that self-hosted, permissively licensed infrastructure avoids.

`MPLSandbox` [@mplsandbox] is the closest system in research intent, providing multi-language
sandboxed execution for LLMs. Its design centre, however, is unified compiler and
static-analysis feedback aimed at improving model training and code-quality analysis,
rather than embedding execution into applications and agent loops. That paper cites an
early description of `LLM Sandbox` as prior work, characterising it as covering only a few
programming languages; the library has since grown to seven, and the comparison is included
here to make the lineage explicit rather than to dispute it.

`LLM Sandbox` was built rather than contributed to these projects because its design
centre is incompatible with theirs. It ships *no runtime of its own*: it is a library that
adapts to container infrastructure an organisation already operates, whether that is
Docker on a laptop or a managed Kubernetes cluster. Adding a Kubernetes backend to a
microVM-based platform would contradict that platform's central premise, so the
multi-backend abstraction — one session API over whichever container runtime is already
in place — could not have been contributed upstream. Two further capabilities follow from the library form factor and are
not the focus of the alternatives: structured extraction of plots and files produced
inside the container, and container pooling that removes creation latency from agent loops
executing code many times per task.

The trade-off is explicit. Because `LLM Sandbox` targets existing container runtimes, it
provides weaker isolation than the microVM systems above. Users executing deliberately
adversarial code should prefer a microVM platform, or pair this library with a hardened
runtime such as gVisor or Kata Containers.

# Software design

The library separates _session semantics_ from _container mechanics_. A backend
implements a narrow container API — create, execute, copy in, copy out, destroy — while
session classes layer language handling, dependency installation, artifact extraction,
and policy enforcement on top. This boundary is what makes the multi-backend claim
tractable: adding a backend does not require touching language support, and adding a
language does not require touching any backend. The cost is an abstraction that must
express the least-common-denominator of three container systems, which is why
backend-specific options are surfaced explicitly rather than hidden.

Three session types share one interface: `SandboxSession` for one-shot execution,
`ArtifactSandboxSession` for runs whose plots and files must be recovered, and
`InteractiveSandboxSession` for stateful, notebook-like sequences in which variables and
imports persist across calls. An optional pool manager pre-warms and recycles containers,
with configurable size bounds, idle and lifetime limits, health checks, and explicit
exhaustion strategies.

Pooling exists because for short-running code — the common case in an agent loop —
container creation rather than execution dominates the cost. Executing a trivial snippet
through a fresh session — creating, setting up and
destroying a container per call — took a median of 17.9 s, against 223 ms when the same
snippet borrowed a pre-warmed container from a pool, an approximately 80-fold reduction in
per-execution overhead for this workload. Those figures come from `benchmarks/pooling_latency.py`, included in the
repository, run over 8 iterations per arm on Docker Desktop 28.2.2 under macOS on
Apple silicon with four CPUs allocated to the Docker VM; image download and the pool's
first acquisition are excluded from timing. The absolute numbers are specific to that
environment — Docker Desktop's virtual machine makes container creation costlier on macOS
than on native Linux, so the ratio should be read as an illustration of where the time
goes rather than a portable constant.

Security is layered rather than singular, and the layers differ in how they are enforced.
A configurable `SecurityPolicy` screens code against patterns before execution, but it is
deliberately *advisory*: `session.is_safe(code)` returns a verdict and the caller decides
what to do with it, and `session.run()` does not reject code on the policy's behalf. This
is a design choice rather than an omission — pattern matching over source code is evadable,
so presenting it as a gate would misrepresent its strength, and applications differ in
whether a violation should abort, warn, or downgrade privileges. The layer that is actually
enforced is the container runtime: network isolation, read-only root filesystems,
capability dropping, and CPU and memory limits are applied by Docker, Podman, or the
Kubernetes pod specification when configured, and hold regardless of what the static
screen concluded. As noted above, the library does not claim kernel-level isolation.

# Research impact statement

`LLM Sandbox` has been adopted as execution infrastructure by both research and
engineering projects. EffiBench-X [@effibenchx], a multi-language benchmark for the
efficiency of LLM-generated code published at NeurIPS 2025, vendors the library into its
evaluation harness and drives execution across all six of its benchmark languages through
the session API — pinning the runtime alongside the benchmark rather than depending on an
external service, which is precisely the reproducibility property described above. The library is
additionally cited as prior work in `MPLSandbox` [@mplsandbox], published at ACL 2025
System Demonstrations.

The library is also embedded as an execution backend in other open-source tooling.
`LiteLLM` [@litellm], a widely used gateway for large language model APIs, builds its
skills sandbox executor on `SandboxSession`; `Mellea` [@mellea], a library for generative
programs, exposes it as a first-class `LLMSandboxEnvironment` execution environment; and
`EvoSkill` [@evoskill], a framework for synthesising reusable agent skills, uses it to
execute and score candidate solutions when evaluating against LiveCodeBench. It further
serves as the execution layer of self-hosted agent-sandbox deployments on cloud
infrastructure [@skypilot] and of MCP code-interpreter servers [@codesandboxmcp], and is
distributed through the official Model Context Protocol registry.

The project shows sustained open development rather than a single release. It has been
developed publicly since June 2024 across more than twenty months of commit activity, with
43 tagged releases, 19 contributors, merged pull requests from 18 developers outside the
core author, and 65 issues opened by external users. It has been downloaded more than 3.9
million times from PyPI, with over 600,000 in the preceding 30 days (GitHub and PyPI, measured 3 August 2026).

The wider trajectory of the field supports the need for this infrastructure: the emergence
of Kubernetes-native agent-sandbox primitives [@agentsandbox] illustrates growing interest in treating isolated
execution as shared infrastructure rather than a per-project concern.

# AI usage disclosure

Generative AI tools were used in the development of this software and in the preparation
of this manuscript, and are disclosed here in accordance with the journal's AI usage
policy.

**Tools.** Several assistants were used over the project's development, each attributed in
the public commit history: GitHub Copilot, including Copilot Autofix; Anthropic's Claude
models (Claude Sonnet 4.5 and Claude Opus 4.6) via the Claude Code command-line assistant;
OpenAI Codex; and Crush. CodeRabbit was additionally used for automated review of pull
requests.

**Scope of assistance.** These tools contributed across four areas: (i) *source code* —
code completion, drafting and refactoring portions of the library, including backend
implementations and language handlers; (ii) *tests* — generating and scaffolding portions
of the test suite; (iii) *documentation* — drafting sections of the project documentation
and README; and (iv) *this manuscript* — drafting prose, structuring sections, and
verifying and formatting the bibliography.

**Human oversight.** The author made all architectural and design decisions, including the
backend abstraction, session model, pooling strategy, and security architecture. All
AI-generated code was reviewed, modified where necessary, and validated against the
project's automated test suite, type checking, and linting before being merged. All
factual claims in this manuscript — including the characterisations of related systems,
the downstream adoption described above, and every bibliographic entry — were verified
against primary sources by the author. The author takes full responsibility for the
accuracy and originality of the software and the manuscript.

# Acknowledgements

We thank the open-source contributors who have submitted backends, language support, and
bug reports to the project.

# References
