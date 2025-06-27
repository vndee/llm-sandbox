"""LLM Sandbox MCP Server.

A Model Context Protocol server that provides secure code execution capabilities using llm-sandbox.
"""

import json
import logging
import os

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from llm_sandbox import ArtifactSandboxSession, SandboxBackend, SandboxSession, SupportedLanguage
from llm_sandbox.data import ExecutionResult
from llm_sandbox.mcp_server.const import LANGUAGE_RESOURCES
from llm_sandbox.session import _check_dependency

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("llm-sandbox-mcp")

mcp = FastMCP("llm-sandbox")


def _get_backend() -> SandboxBackend:
    """Get the backend to use for the sandbox session."""
    backend = SandboxBackend(os.environ.get("BACKEND", "docker"))
    _check_dependency(backend)
    return backend


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
        use_artifact_session = _supports_visualization(language)
        session_cls = ArtifactSandboxSession if use_artifact_session else SandboxSession

        with session_cls(
            lang=language,
            keep_template=True,
            verbose=False,
            backend=_get_backend(),
            session_timeout=timeout,
        ) as session:
            result = session.run(
                code=code,
                libraries=libraries or [],
                timeout=timeout,
            )

            if use_artifact_session and hasattr(result, "plots") and result.plots:
                plot = result.plots[0]
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
