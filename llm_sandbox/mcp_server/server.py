"""LLM Sandbox MCP Server.

A Model Context Protocol server that provides secure code execution capabilities using llm-sandbox.
"""

import json
import logging
import os
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from llm_sandbox import ArtifactSandboxSession, SandboxBackend, SandboxSession, SupportedLanguage, ValidationError
from llm_sandbox.data import ExecutionResult
from llm_sandbox.mcp_server.const import LANGUAGE_RESOURCES
from llm_sandbox.session import _check_dependency

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("llm-sandbox-mcp")

mcp = FastMCP("llm-sandbox")

_TRUE_ENV_VALUES = {"true", "1", "yes", "on"}
_FALSE_ENV_VALUES = {"false", "0", "no", "off"}
_RUNTIME_CONFIG_ENV_VARS = {
    "SANDBOX_NETWORK_MODE",
    "SANDBOX_READ_ONLY",
    "SANDBOX_MEMORY",
    "SANDBOX_MEM_LIMIT",
    "SANDBOX_CPUS",
    "SANDBOX_CPU_COUNT",
    "SANDBOX_CAP_DROP",
    "SANDBOX_SECURITY_OPT",
    "SANDBOX_PRIVILEGED",
}
_RUNTIME_CONFIG_SUPPORTED_BACKENDS = {
    SandboxBackend.DOCKER,
    SandboxBackend.PODMAN,
    SandboxBackend.MICROMAMBA,
}


def _get_backend() -> SandboxBackend:
    """Get the backend to use for the sandbox session."""
    backend = SandboxBackend(os.environ.get("BACKEND", "docker"))
    _check_dependency(backend)
    return backend


def _get_commit_container() -> bool:
    """Get the commit_container setting from environment variable.

    Defaults to ``False`` because MCP clients pass attacker-controlled code into the
    sandbox; persisting container state to the source image by default would let one
    request's side effects bleed into later sessions.
    """
    commit_container_env = os.environ.get("COMMIT_CONTAINER", "false").lower()
    return commit_container_env in _TRUE_ENV_VALUES


def _get_keep_template() -> bool:
    """Get the keep_template setting from environment variable.

    Defaults to ``True``: the template image is the pristine upstream base image
    (e.g. ``python:3.11-bullseye``). It contains no untrusted state, and reusing it
    between MCP requests avoids re-pulling on every call. Untrusted state is gated
    separately by ``COMMIT_CONTAINER``.
    """
    keep_template_env = os.environ.get("KEEP_TEMPLATE", "true").lower()
    return keep_template_env in _TRUE_ENV_VALUES


def _build_commit_image_tag(language: str) -> str:
    """Build a unique opt-in commit tag so commits never overwrite the source image.

    Format: ``llm-sandbox-mcp/<language>:<short-uuid>``. A fresh tag per session means
    one request's persisted state can never silently land in another session that
    happened to reuse the same base image.
    """
    short = uuid.uuid4().hex[:12]
    return f"llm-sandbox-mcp/{language.lower()}:{short}"


def _get_kube_namespace() -> str:
    """Get the Kubernetes namespace from environment variable."""
    return os.environ.get("NAMESPACE", "default")


def _get_optional_bool_env(var_name: str) -> bool | None:
    """Parse an optional boolean environment variable."""
    value = os.environ.get(var_name)
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False

    msg = f"{var_name} must be one of: true, false, 1, 0, yes, no, on, off"
    raise ValidationError(msg)


def _get_optional_list_env(var_name: str) -> list[str] | None:
    """Parse an optional comma-separated list environment variable."""
    value = os.environ.get(var_name)
    if value is None:
        return None

    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _get_optional_str_env(var_name: str) -> str | None:
    """Parse an optional string environment variable."""
    value = os.environ.get(var_name)
    if value is None:
        return None

    normalized = value.strip()
    return normalized or None


def _build_cpu_runtime_configs(cpu_units: Decimal, var_name: str) -> dict[str, int]:
    """Translate CPU units into Linux-compatible runtime config keys."""
    cpu_period = 100_000
    cpu_quota = int(cpu_units * cpu_period)
    if cpu_quota <= 0:
        msg = f"{var_name} is too small to translate into cpu_quota"
        raise ValidationError(msg)

    return {"cpu_period": cpu_period, "cpu_quota": cpu_quota}


def _parse_cpu_count_env(value: str, var_name: str) -> dict[str, int]:
    """Parse SANDBOX_CPU_COUNT into Linux-compatible CPU runtime configs."""
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation as exc:
        msg = f"{var_name} must be a positive number"
        raise ValidationError(msg) from exc

    if parsed <= 0:
        msg = f"{var_name} must be greater than 0"
        raise ValidationError(msg)
    if parsed != parsed.to_integral_value():
        msg = f"{var_name} must be a whole number when mapped to cpu_period/cpu_quota"
        raise ValidationError(msg)

    return _build_cpu_runtime_configs(parsed, var_name)


def _parse_cpus_env(value: str) -> dict[str, int]:
    """Parse SANDBOX_CPUS into Linux-compatible CPU runtime configs."""
    try:
        parsed = Decimal(value.strip())
    except InvalidOperation as exc:
        msg = "SANDBOX_CPUS must be a positive number"
        raise ValidationError(msg) from exc

    if parsed <= 0:
        msg = "SANDBOX_CPUS must be greater than 0"
        raise ValidationError(msg)

    return _build_cpu_runtime_configs(parsed, "SANDBOX_CPUS")


def _get_runtime_configs() -> dict[str, bool | str | int | list[str]]:
    """Build runtime_configs from MCP server environment variables."""
    runtime_configs: dict[str, bool | str | int | list[str]] = {}

    passthrough_env_to_config: dict[str, str] = {
        "SANDBOX_NETWORK_MODE": "network_mode",
    }
    for env_name, config_key in passthrough_env_to_config.items():
        value = _get_optional_str_env(env_name)
        if value is not None:
            runtime_configs[config_key] = value

    mem_limit = _get_optional_str_env("SANDBOX_MEM_LIMIT") or _get_optional_str_env("SANDBOX_MEMORY")
    if mem_limit is not None:
        runtime_configs["mem_limit"] = mem_limit

    cpu_count = os.environ.get("SANDBOX_CPU_COUNT")
    cpus = os.environ.get("SANDBOX_CPUS")
    if cpu_count is not None:
        runtime_configs.update(_parse_cpu_count_env(cpu_count, "SANDBOX_CPU_COUNT"))
    elif cpus is not None:
        runtime_configs.update(_parse_cpus_env(cpus))

    bool_env_to_config: dict[str, str] = {
        "SANDBOX_READ_ONLY": "read_only",
        "SANDBOX_PRIVILEGED": "privileged",
    }
    for env_name, config_key in bool_env_to_config.items():
        bool_value = _get_optional_bool_env(env_name)
        if bool_value is not None:
            runtime_configs[config_key] = bool_value

    list_env_to_config: dict[str, str] = {
        "SANDBOX_CAP_DROP": "cap_drop",
        "SANDBOX_SECURITY_OPT": "security_opt",
    }
    for env_name, config_key in list_env_to_config.items():
        list_value = _get_optional_list_env(env_name)
        if list_value is not None:
            runtime_configs[config_key] = list_value

    return runtime_configs


def _validate_backend_runtime_configs(
    backend: SandboxBackend,
    runtime_configs: dict[str, bool | str | int | list[str]] | None = None,
) -> None:
    """Raise ValidationError if runtime configs are used with an unsupported backend."""
    if backend in _RUNTIME_CONFIG_SUPPORTED_BACKENDS:
        return

    msg = (
        f"SANDBOX_* runtime config environment variables are not supported for backend '{backend.value}' "
        "in this MCP server. Kubernetes resource and security controls require a custom pod_manifest "
        "via a custom MCP wrapper or direct llm-sandbox usage."
    )

    if any(env_name in os.environ for env_name in _RUNTIME_CONFIG_ENV_VARS):
        raise ValidationError(msg)
    if runtime_configs:
        raise ValidationError(msg)


def _supports_visualization(language: str) -> bool:
    """Check if a language supports visualization capture."""
    lang_details = LANGUAGE_RESOURCES.get(language)
    return lang_details.get("visualization_support", False) if lang_details else False


@mcp.tool()
def execute_code(
    code: str,
    language: str = "python",
    libraries: list[str] | None = None,
    timeout: int = 30,
) -> list[ImageContent | TextContent]:
    """Execute code in a secure sandbox environment and automatic visualization capture.

    Args:
        code: The code to execute
        language: Programming language (python, javascript, java, cpp, go, r, ruby)
        libraries: List of libraries/packages to install
        timeout: Execution timeout in seconds (default: 30)

    Returns:
        List of content items including execution results and any generated visualizations

    """
    results: list[ImageContent | TextContent] = []

    try:
        backend = _get_backend()
        _validate_backend_runtime_configs(backend)
        runtime_configs = _get_runtime_configs()
        _validate_backend_runtime_configs(backend, runtime_configs)

        use_artifact_session = _supports_visualization(language)
        session_cls = ArtifactSandboxSession if use_artifact_session else SandboxSession

        commit_container = _get_commit_container()
        session_kwargs: dict[str, Any] = {
            "lang": language,
            "keep_template": _get_keep_template(),
            "commit_container": commit_container,
            "verbose": False,
            "backend": backend,
            "session_timeout": timeout,
            "kube_namespace": _get_kube_namespace(),
        }
        if commit_container and backend in {SandboxBackend.DOCKER, SandboxBackend.MICROMAMBA}:
            session_kwargs["commit_image_tag"] = _build_commit_image_tag(language)
        if runtime_configs:
            session_kwargs["runtime_configs"] = runtime_configs

        with session_cls(**session_kwargs) as session:
            if use_artifact_session:
                result = session.run(  # type: ignore[call-arg]
                    code=code, libraries=libraries or [], timeout=timeout, clear_plots=True
                )
            else:
                result = session.run(code=code, libraries=libraries or [], timeout=timeout)

            if use_artifact_session and hasattr(result, "plots") and result.plots:
                plot = result.plots[-1]
                results.append(
                    ImageContent(
                        data=plot.content_base64,
                        mimeType=f"image/{plot.format.value}",
                        type="image",
                    )
                )

            results.append(TextContent(text=result.to_json(include_plots=False), type="text"))

    except Exception as e:
        logger.exception("Error executing code")
        return [TextContent(text=ExecutionResult(exit_code=1, stderr=str(e)).to_json(), type="text")]

    else:
        return results


@mcp.tool()
def get_supported_languages() -> TextContent:
    """Get the list of supported languages.

    Returns:
        TextContent: The list of supported languages

    """
    return TextContent(text=json.dumps([lang.value for lang in SupportedLanguage], indent=2), type="text")


@mcp.tool()
def get_language_details(language: str) -> TextContent:
    """Get the details of a language.

    Args:
        language: The language to get the details of

    Returns:
        TextContent: The details of the language

    """
    try:
        lang = SupportedLanguage(language)
        lang_details = LANGUAGE_RESOURCES.get(lang)
        return TextContent(
            text=json.dumps(lang_details, indent=2),
            type="text",
        )
    except ValueError:
        return TextContent(
            text=json.dumps({"error": f"Unsupported language: {language}"}),
            type="text",
        )


@mcp.resource("sandbox://languages")
def language_details() -> str:
    """Resource containing detailed information about supported languages.

    Returns:
        str: The details of the languages

    """
    return json.dumps(LANGUAGE_RESOURCES, indent=2)


def main() -> None:
    """Set up and run the server."""
    logger.info("Starting MCP server with backend: %s", _get_backend())
    mcp.run()


if __name__ == "__main__":
    main()
