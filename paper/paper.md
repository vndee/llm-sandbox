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

`LLM Sandbox` is a Python library that provides this execution layer. A single
context-managed `SandboxSession` accepts a code string and returns its stdout, stderr,
exit code, and any generated artifacts, having run the code inside an isolated container.
The same API is served by four interchangeable backends — Docker, Podman, Kubernetes, and
Micromamba — so that a workflow prototyped on a laptop runs unchanged on a cluster. Seven
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
sandbox API, which removes that burden but introduces per-execution cost, network
dependence, and the requirement that potentially sensitive code and data leave the
user's infrastructure. Neither option serves work under data-governance constraints,
air-gapped evaluation, or high-volume batch execution where per-call pricing dominates.

`LLM Sandbox` targets this gap: it is installable with `pip`, runs entirely on
infrastructure the user already controls, and imposes no per-execution cost. This
combination is particularly relevant to reproducible evaluation. Benchmarks that measure
properties of generated code must execute large volumes of untrusted programs under
conditions that other researchers can recreate exactly; a self-hosted, versioned,
MIT-licensed runtime is easier to pin and re-run than a commercial endpoint whose
behaviour may change.

This requirement is visible in practice. EffiBench-X, a multi-language benchmark for the
efficiency of LLM-generated code [@effibenchx], vendors `LLM Sandbox` directly into its
evaluation harness and drives execution across all six of its benchmark languages through
the library's session API — pinning the runtime alongside the benchmark rather than
depending on an external service. The library is also used as the execution layer of
self-hosted agent-sandbox deployments [@skypilot] and of MCP code-interpreter servers
[@codesandboxmcp]. Related systems have independently identified multi-language sandboxed
execution as core infrastructure for code-oriented LLM research [@mplsandbox], and the
emergence of Kubernetes-native agent-sandbox primitives [@agentsandbox] indicates that
isolated execution is becoming a standard component of agent infrastructure rather than a
per-project concern.

# Design

The library separates _session semantics_ from _container mechanics_. A backend
implements a narrow container API — create, execute, copy in, copy out, destroy — while
session classes layer language handling, dependency installation, artifact extraction,
and policy enforcement on top. Adding a backend therefore does not require touching
language support, and adding a language does not require touching any backend.

Three session types share one interface: `SandboxSession` for one-shot execution,
`ArtifactSandboxSession` for runs whose plots and files must be recovered, and
`InteractiveSandboxSession` for stateful, notebook-like sequences in which variables and
imports persist across calls. An optional pool manager pre-warms and recycles containers,
with configurable size bounds, idle and lifetime limits, health checks, and explicit
exhaustion strategies, which removes container-creation latency from the hot path of
agent loops that execute code many times per task.

Security is layered rather than singular. Static policies reject code matching
configurable patterns before any container starts; container-level controls then apply
network isolation, read-only root filesystems, capability dropping, and CPU and memory
limits. The library does not claim kernel-level isolation: it inherits the container
threat model of its chosen backend, and users requiring stronger guarantees should pair
it with a hardened runtime or a virtual-machine backend.

# Acknowledgements

We thank the open-source contributors who have submitted backends, language support, and
bug reports to the project.

# References
