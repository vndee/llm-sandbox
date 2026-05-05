"""Runtime hardening profiles for container backends.

This module centralises the per-backend defaults applied by
:class:`llm_sandbox.const.RuntimeProfile`. Keeping the definitions here lets the
Docker/Podman/Kubernetes sessions stay thin and makes the strict baseline easy
to audit in one place.

The :data:`STRICT_DOCKER_DEFAULTS` mapping intentionally uses keyword-argument
names that match ``docker.client.containers.create``; the Podman session reuses
them through inheritance after its own normalisation pass. The Kubernetes
helper builds a ``securityContext`` block compatible with the standard pod
manifest schema.

The conventions used by :func:`apply_strict_runtime_configs`:

* User-supplied values in ``runtime_configs`` always win over profile defaults.
* Passing ``None`` for a key explicitly drops that knob (useful for images
  that need an unrestricted network or larger memory than the profile sets).
"""

from __future__ import annotations

from typing import Any

STRICT_USER = "1000:1000"

STRICT_DOCKER_DEFAULTS: dict[str, Any] = {
    "user": STRICT_USER,
    "cap_drop": ["ALL"],
    "security_opt": ["no-new-privileges:true"],
    "network_mode": "none",
    "pids_limit": 512,
    "mem_limit": "512m",
}

STRICT_K8S_POD_SECURITY_CONTEXT: dict[str, Any] = {
    "runAsUser": 1000,
    "runAsGroup": 1000,
    "runAsNonRoot": True,
    "fsGroup": 1000,
    "seccompProfile": {"type": "RuntimeDefault"},
}

STRICT_K8S_CONTAINER_SECURITY_CONTEXT: dict[str, Any] = {
    "runAsUser": 1000,
    "runAsGroup": 1000,
    "runAsNonRoot": True,
    "allowPrivilegeEscalation": False,
    "capabilities": {"drop": ["ALL"]},
    "seccompProfile": {"type": "RuntimeDefault"},
}


def apply_strict_runtime_configs(runtime_configs: dict[str, Any]) -> dict[str, Any]:
    """Layer the strict Docker/Podman defaults under user-supplied configs.

    User-supplied keys win unconditionally. A key explicitly mapped to ``None``
    is treated as an opt-out and is removed from the merged result, allowing
    callers to disable individual hardening knobs without leaving the strict
    profile.
    """
    merged: dict[str, Any] = {**STRICT_DOCKER_DEFAULTS}
    opt_outs: set[str] = set()
    for key, value in runtime_configs.items():
        if value is None and key in STRICT_DOCKER_DEFAULTS:
            opt_outs.add(key)
            continue
        merged[key] = value
    for key in opt_outs:
        merged.pop(key, None)
    return merged


def strict_pod_security_context() -> dict[str, Any]:
    """Return a fresh copy of the strict pod-level securityContext."""
    return {**STRICT_K8S_POD_SECURITY_CONTEXT, "seccompProfile": {"type": "RuntimeDefault"}}


def strict_container_security_context() -> dict[str, Any]:
    """Return a fresh copy of the strict container-level securityContext."""
    return {
        **STRICT_K8S_CONTAINER_SECURITY_CONTEXT,
        "capabilities": {"drop": ["ALL"]},
        "seccompProfile": {"type": "RuntimeDefault"},
    }
