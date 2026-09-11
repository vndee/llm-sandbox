# Security Policy

Thanks for taking the time to help keep `llm-sandbox` safe. This document
explains how to report a vulnerability and what to expect back.

## Reporting a vulnerability

**Please do not open a public GitHub issue** for security findings. Public
issues are indexed instantly and disclosing before a fix is available puts
downstream users at risk.

Preferred channels, in order:

1. **GitHub private vulnerability reporting** — the "Report a vulnerability"
   button on the [Security tab](https://github.com/vndee/llm-sandbox/security/advisories/new).
   Reports open a private draft advisory that only maintainers can see.
2. **Email** — `vndee.huynh@gmail.com`. Subject line prefix `[llm-sandbox security]`
   helps it get triaged quickly. Encrypt only if you have a key from the maintainer;
   plain email is fine.

### What to include

A useful report contains:

- Affected version(s) — commit SHA is ideal, tag or `pip show llm-sandbox`
  output is fine.
- Component — e.g. `SecurityPolicy.is_safe()`, a specific backend, a
  language handler.
- Impact — what an attacker can achieve, and a rough severity estimate
  (Low / Medium / High / Critical).
- A minimal reproduction — a Python snippet against a local clone is
  enough; a runnable script is even better.
- Suggested fix — optional, but appreciated.

### What to expect

- Acknowledgement within **72 hours** on business days.
- A triage decision (accepted / needs more info / out of scope) within
  **7 days**.
- Coordinated disclosure timeline once triaged. Default embargo is
  **90 days** from acknowledgement, or until a fix is released, whichever
  comes first. We can shorten or extend that by agreement.
- Credit in the release notes and the resulting GitHub Security Advisory
  unless you ask to remain anonymous.

## Scope

**In scope:**

- Container escapes or privilege escalation out of the sandbox.
- Bypasses of `SecurityPolicy` / `is_safe()` pre-execution screening.
- Command injection or unsafe shell construction inside any backend,
  language handler, or MCP tool.
- Secret / token exfiltration from `llm-sandbox` itself.
- Supply-chain issues in code we own (build config, release workflows,
  Docker images under `dockers/`).

**Out of scope:**

- Prompt-injection or jailbreaks of the language model calling into
  `llm-sandbox`. This project is a sandbox for LLM-generated code, not a
  guardrail against adversarial prompts.
- Vulnerabilities in upstream base images (`python:*`, `openjdk:*`, etc.)
  that are not exploitable through `llm-sandbox`'s use of them — report
  those to the image maintainer.
- Denial-of-service by giving the sandbox arbitrarily large or slow code
  when no resource limits were configured. Configure `runtime_configs`
  (memory, CPU, timeout) instead.
- Findings that require an attacker who already has host / cluster
  root — that is not a threat model this project defends against.

If you are unsure whether something is in scope, report it anyway and
we will sort it out together.

## Supported versions

Security fixes are cut against the **latest released minor version** on
PyPI. Older versions are not maintained; the fix is to upgrade.

| Version | Supported          |
|---------|--------------------|
| latest  | :white_check_mark: |
| older   | :x:                |

## Disclosure history

Once we have shipped fixes, published advisories will be listed at
<https://github.com/vndee/llm-sandbox/security/advisories>.
