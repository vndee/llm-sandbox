# Community plugin index

Backends that LLM Sandbox does not ship, published as separate packages by their maintainers.

```bash
pip install llm-sandbox-<service>
```

```python
from llm_sandbox import SandboxSession

with SandboxSession(backend="<service>", lang="python") as session:
    print(session.run("print('hello')").stdout)
```

!!! danger "These are not our packages"

    A backend is the component that executes untrusted code — the most security-sensitive
    part of this system. Installing one of these means running third-party code in your
    environment and handing that package's authors your credentials, your infrastructure, and
    the code you send to be executed.

    **Listing here records that a plugin exists and is community-maintained. It is not a
    review, an endorsement, or a guarantee.** We do not audit these packages, we cannot test
    them in CI, and we cannot debug them. When you choose a community backend you are
    trusting its authors, not us.

    Evaluate one the way you would evaluate any dependency that handles your secrets: read
    the source, check who maintains it and how actively, confirm it passes the compliance
    kit, and look at what it sends off your machine.

    The built-in Docker, Podman, Kubernetes, and Micromamba backends are maintained and
    CI-tested in this repository. If your code must not leave your infrastructure, use those.

## Plugins

| Name | Service | Maintainer | Repository |
|---|---|---|---|
| `llm-sandbox-example` | — (local subprocess reference implementation) | LLM Sandbox | [examples/plugins/llm-sandbox-example][example] |

[example]: https://github.com/vndee/llm-sandbox/tree/main/examples/plugins/llm-sandbox-example

!!! warning "`llm-sandbox-example` is not a sandbox"

    The reference plugin runs code directly on the host machine with no isolation at all. It
    is published as source to copy, not to depend on, and is not on PyPI. Use it to learn the
    interface; never point it at untrusted code.

## Getting listed

[Open an issue](https://github.com/vndee/llm-sandbox/issues) with:

- the package name (convention: `llm-sandbox-<service>`)
- the service it executes code on
- the maintainer
- the repository URL
- confirmation that it passes [the compliance kit](authoring.md#5-run-the-compliance-kit)

You do not need approval to publish a plugin — LLM Sandbox is MIT-licensed and the
entry-point interface is public. This index exists so people can find your plugin, not to
gate it.

If you are an employee of, contractor for, or otherwise compensated by the service your
plugin integrates, please say so in the issue. Most vendor integrations are written by the
vendor, and that is expected.

## What belongs in a plugin rather than core

Commercial and hosted execution services. If running code requires an account, an API key, or
a call to infrastructure you operate, it belongs in a plugin — not as a judgement on the
service, but because this project cannot test it in CI without your credentials, cannot debug
it when it breaks, and cannot promise users it still works after a release.

Open-source runtimes that execute entirely on the end user's own machine are wanted in core.

[INTEGRATIONS.md](https://github.com/vndee/llm-sandbox/blob/main/INTEGRATIONS.md) has the
full policy and the reasoning behind it.

## See also

- [Writing a backend plugin](authoring.md)
- [Container Backends](../backends.md) — the built-in backends
