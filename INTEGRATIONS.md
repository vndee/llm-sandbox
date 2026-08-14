# Backend Integrations

LLM Sandbox ships with several backends - Docker, Podman, and Kubernetes. All of them run on infrastructure the user controls, which is the core promise of this project.

This page explains which backends belong in core, which are better as plugins, and why. The terms are the same for everyone and are stated here so they are known before anyone writes code.

## What belongs in core

**Open-source runtimes that execute entirely on the end user's own system.**

If a backend meets that description, we want it in core and we will help you land it. Another container runtime, a microVM runtime, a WASM runtime, a language sandbox - if it is open source and a user can run it on their own machine or cluster without an account or a network call to a vendor, open an issue and let's talk about it.

The requirement is practical: we run it in CI, debug it without anyone's credentials, and keep it working across releases. That is what makes this project dependable for the systems built on it.

## What we will not accept into core

- Backends that cannot be tested in CI without a third-party account.
- Integrations that change default behaviour, or that route execution anywhere
  other than the backend the user explicitly selected.
- Anything that makes a network service a required dependency of core.

## What belongs in a plugin

**Commercial and hosted execution services.**

If executing code requires an account, an API key, or a call to infrastructure you operate, the backend belongs in a plugin rather than in core. This is not a judgement on the service. It is that we cannot test it in CI without your credentials, cannot debug it when it breaks, and cannot promise our users it still works after a release — and shipping something in core carries all three of those promises.

A backend plugin is a standalone package that depends on `llm-sandbox` and registers itself through the backend entry-point interface. Once installed, it is selectable like any built-in backend:

```bash
pip install llm-sandbox-<service>
```

```python
SandboxSession(backend="<service>")
```

The package is yours. You own the release cadence, the support burden, and the compatibility story with your own API — which means you can ship a fix the day your SDK changes instead of waiting for a release of this project.

**You do not need our approval to publish one.** LLM Sandbox is MIT-licensed and the entry-point interface is public. There is no application, no review, and no fee. If you build one, open an issue and we will add it to the community plugin index; that listing records that the plugin exists and is community-maintained, and implies no endorsement or guarantee from us.
Naming convention: `llm-sandbox-<service>`. Please do not publish a package named `llm-sandbox-*` that is a fork or a vendored copy rather than a dependent — it confuses users about which package they are installing.

## Disclosure

If you are an employee of, contractor for, or otherwise compensated by the service your contribution integrates, please say so in the pull request. This is not a barrier — most vendor integrations are written by the vendor, and that is expected.

The same applies in the other direction: if the maintainers of this project develop a commercial interest in the code-execution space, it will be disclosed here.

## Supporting the project

Separately from all of the above: LLM Sandbox is maintained by one person and has no commercial backing, while being installed several hundred thousand times a month and depended on by production systems and published research.

If your company builds on it, sponsorship funds the maintenance, security review, and support that keeps it working: **[GitHub Sponsors](https://github.com/sponsors/vndee)**.

This is not a requirement. It has no bearing on whether a plugin can be published, whether a backend is accepted into core, how issues are prioritised, or anything else described on this page. Nothing on this page is for sale.

## Why this policy exists

A large number of people chose LLM Sandbox specifically because their code does not leave their infrastructure. This split keeps that default intact: core stays fully self-hostable and testable, while commercial services integrate through plugins that are opt-in, clearly labelled, and maintained by the people who understand the service best.

Questions: open a discussion, or email vndee.huynh@gmail.com.
