#!/usr/bin/env python3
# ruff: noqa: T201, INP001
"""Smoke-check one agent SDK example without calling a model.

Importing an example proves its APIs still exist; it does not prove the example
would run. The AG2 example shipped with `config=None`, which imports cleanly and
only fails inside `run()` -- an import check could never have caught it.

This goes one level deeper, for free:

  * the module imports;
  * it drives the sandbox itself (references `SandboxSession`) rather than
    delegating to a helper;
  * it applies container hardening;
  * the tool the model sees takes exactly one parameter, `code`. This is the
    check that catches a leaked `libraries` parameter, which would let a model
    install arbitrary PyPI packages.

With --docker it also executes a snippet in a real container, which exercises
the hardened runtime config end to end.

Each framework needs its own environment (several pin conflicting LangChain
versions), so this checks one module per invocation:

    python scripts/smoke_agent_sdks.py openai_agents_tool --docker
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "agent_sdks"

# How to reach the parameter schema the model is shown. Frameworks that expose
# no introspectable schema fall back to the wrapper function's signature.
SCHEMA_ACCESSORS: dict[str, Any] = {
    "openai_agents_tool": lambda m: m.execute_python.params_json_schema["properties"],
    "smolagents_tool": lambda m: m.execute_python.inputs,
    "crewai_tool": lambda m: m.execute_python.args_schema.model_json_schema()["properties"],
    "deepagents_tool": lambda m: m.execute_python.args_schema.model_json_schema()["properties"],
    "langchain_tool": lambda m: m.execute_python.args_schema.model_json_schema()["properties"],
    "llamaindex_tool": lambda m: m.sandbox_tool.metadata.get_parameters_dict()["properties"],
    # These two declare their schema separately from the handler signature, so
    # the signature fallback reads the wrong thing (`args`, or the decorator's
    # own injected params) and reports a false failure.
    "claude_agent_sdk_tool": lambda m: m.execute_python.input_schema,
    "ag2_tool": lambda m: m.execute_python.schema.function.parameters["properties"],
}

REQUIRED_HARDENING = {"network_mode", "mem_limit", "pids_limit", "cap_drop", "security_opt"}


def _fallback_schema(module: Any) -> dict[str, Any]:
    """Read parameter names off the tool wrapper when the SDK exposes no schema."""
    fn = getattr(module, "execute_python", None)
    if fn is None:
        msg = "module defines no `execute_python` tool"
        raise AssertionError(msg)
    fn = getattr(fn, "__wrapped__", getattr(fn, "func", fn))
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    return {name: None for name in params if name not in {"self", "args", "kwargs"}}


def check(name: str, *, with_docker: bool) -> list[str]:
    """Run every check against one example module, returning failures."""
    failures: list[str] = []
    sys.path.insert(0, str(EXAMPLES_DIR))
    module = importlib.import_module(name)

    source = Path(module.__file__).read_text()

    # The examples exist to teach this library; delegating to a helper hides it.
    if "SandboxSession" not in source:
        failures.append("does not reference SandboxSession directly")

    runtime = getattr(module, "SANDBOX_RUNTIME", None)
    if not isinstance(runtime, dict):
        failures.append("missing SANDBOX_RUNTIME")
    else:
        missing = REQUIRED_HARDENING - set(runtime)
        if missing:
            failures.append(f"SANDBOX_RUNTIME missing {sorted(missing)}")
        if runtime.get("network_mode") != "none":
            failures.append(f"network_mode is {runtime.get('network_mode')!r}, expected 'none'")

    accessor = SCHEMA_ACCESSORS.get(name)
    try:
        params = set(accessor(module)) if accessor else set(_fallback_schema(module))
    except Exception as exc:  # noqa: BLE001 - any failure here is a real finding
        failures.append(f"could not read tool schema: {type(exc).__name__}: {exc}")
    else:
        if params != {"code"}:
            # `libraries` leaking through is the specific regression guarded here.
            failures.append(f"tool exposes {sorted(params)}, expected ['code']")

    if with_docker:
        run_python = getattr(module, "run_python", None)
        if run_python is None:
            failures.append("no run_python to execute")
        else:
            out = run_python("print(6 * 7)")
            if out.strip() != "42":
                failures.append(f"run_python returned {out!r}, expected '42'")

    return failures


def main() -> int:
    """Check one example and report pass or fail."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", help="example module name, e.g. openai_agents_tool")
    parser.add_argument("--docker", action="store_true", help="also execute code in a real container")
    args = parser.parse_args()

    try:
        failures = check(args.module, with_docker=args.docker)
    except Exception as exc:  # noqa: BLE001 - import errors are the headline failure
        print(f"FAIL {args.module}: {type(exc).__name__}: {exc}")
        return 1

    if failures:
        print(f"FAIL {args.module}")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"ok   {args.module}" + (" (with docker)" if args.docker else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
