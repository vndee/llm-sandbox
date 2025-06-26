"""LLM Sandbox MCP Server.

A Model Context Protocol server that provides secure code execution
capabilities using llm-sandbox with Docker containers.
"""

import json
import logging
import os
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from llm_sandbox import ArtifactSandboxSession, SandboxBackend, SandboxSession
from llm_sandbox.mcp_server.const import LANGUAGE_RESOURCES
from llm_sandbox.session import _check_dependency

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("llm-sandbox-mcp")

mcp = FastMCP("llm-sandbox")


def get_backend() -> SandboxBackend:
    """Get the backend to use for the sandbox session."""
    backend = SandboxBackend(os.environ.get("BACKEND", "docker"))
    _check_dependency(backend)
    return backend


@mcp.tool()
def execute_code(
    code: str,
    language: str = "python",
    libraries: list[str] | None = None,
    timeout: int = 30,
) -> str:
    """Execute code in a secure sandbox environment.

    Args:
        code: The code to execute
        language: Programming language (python, javascript, java, cpp, go, r)
        libraries: List of libraries/packages to install
        timeout: Execution timeout in seconds (default: 30)

    Returns:
        JSON string containing execution results

    """
    try:
        with SandboxSession(
            lang=language,
            keep_template=True,
            verbose=False,
            backend=get_backend(),
            session_timeout=timeout,
        ) as session:
            result = session.run(
                code=code,
                libraries=libraries or [],
                timeout=timeout,
            )

            return json.dumps({
                "success": result.exit_code == 0,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "language": language,
                "libraries": libraries or [],
            })

    except Exception:
        logger.exception("Error executing code")
        return json.dumps({"success": False, "language": language})


@mcp.tool()
def create_visualization(
    code: str,
    language: str = "python",
    libraries: list[str] | None = None,
    timeout: int = 45,
) -> list[ImageContent | TextContent]:
    """Create data visualizations and capture the generated plots.

    Args:
        code: Code that generates plots/visualizations
        language: Programming language (python or r recommended for plots)
        libraries: List of visualization libraries to install
        timeout: Execution timeout in seconds (default: 45)

    Returns:
        A file object containing the generated plot.

    """
    try:
        with ArtifactSandboxSession(
            lang=language,
            keep_template=True,
            verbose=False,
            backend=get_backend(),
            session_timeout=timeout,
        ) as session:
            result = session.run(
                code=code,
                libraries=libraries,
                timeout=timeout,
            )

            results: list[ImageContent | TextContent] = []
            if hasattr(result, "plots") and result.plots:
                plot = result.plots[0]
                results.append(
                    ImageContent(
                        data=plot.content_base64,
                        mimeType=f"image/{plot.format.value}",
                        type="image",
                    )
                )

            if hasattr(result, "stdout") and result.stdout:
                results.append(TextContent(text=result.stdout, type="text"))

            if hasattr(result, "stderr") and result.stderr:
                results.append(TextContent(text=result.stderr, type="text"))

            return results

    except Exception:
        logger.exception("Error creating visualization")
        return []


@mcp.tool()
def execute_code_with_files(
    code: str,
    language: str = "python",
    files: dict[str, str] | None = None,
    libraries: list[str] | None = None,
    timeout: int = 30,
) -> str:
    """Execute code with additional files in a secure sandbox environment.

    Args:
        code: The code to execute
        language: Programming language (python, javascript, java, cpp, go, r)
        files: Dictionary of filename -> content to create in sandbox
        libraries: List of libraries/packages to install
        timeout: Execution timeout in seconds (default: 30)

    Returns:
        JSON string containing execution results

    """
    try:
        with SandboxSession(
            lang=language,
            keep_template=True,
            verbose=False,
            backend=get_backend(),
            session_timeout=timeout,
        ) as session:
            # Create files in sandbox if provided
            if files:
                with tempfile.TemporaryDirectory() as temp_dir:
                    for filename, content in files.items():
                        # Create local file
                        local_path = Path(temp_dir) / filename
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        local_path.write_text(content)

                        # Copy to sandbox
                        sandbox_path = f"/sandbox/{filename}"
                        session.copy_to_runtime(str(local_path), sandbox_path)

            # Execute the code
            result = session.run(
                code=code,
                libraries=libraries or [],
                timeout=timeout,
            )

            return json.dumps({
                "success": result.exit_code == 0,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "language": language,
                "libraries": libraries or [],
                "files_created": list(files.keys()) if files else [],
            })

    except Exception:
        logger.exception("Error executing code with files")
        return json.dumps({"success": False, "language": language})


@mcp.tool()
def create_visualization_with_files(
    code: str,
    language: str = "python",
    files: dict[str, str] | None = None,
    libraries: list[str] | None = None,
    timeout: int = 45,
) -> list[ImageContent | TextContent]:
    """Create data visualizations and capture the generated plots with additional files in a secure sandbox environment.

    Args:
        code: Code that generates plots/visualizations
        language: Programming language (python or r recommended for plots)
        files: Dictionary of filename -> content to create in sandbox
        libraries: List of visualization libraries to install
        timeout: Execution timeout in seconds (default: 45)

    Returns:
        A file object containing the generated plot.

    """
    try:
        with ArtifactSandboxSession(
            lang=language,
            keep_template=True,
            verbose=False,
            backend=get_backend(),
            session_timeout=timeout,
        ) as session:
            if files:
                with tempfile.TemporaryDirectory() as temp_dir:
                    for filename, content in files.items():
                        # Create local file
                        local_path = Path(temp_dir) / filename
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        local_path.write_text(content)

                        # Copy to sandbox
                        sandbox_path = f"/sandbox/{filename}"
                        session.copy_to_runtime(str(local_path), sandbox_path)

            result = session.run(
                code=code,
                libraries=libraries,
                timeout=timeout,
            )

            results: list[ImageContent | TextContent] = []
            if hasattr(result, "plots") and result.plots:
                plot = result.plots[0]
                results.append(
                    ImageContent(
                        data=plot.content_base64,
                        mimeType=f"image/{plot.format.value}",
                        type="image",
                    )
                )

            if hasattr(result, "stdout") and result.stdout:
                results.append(TextContent(text=result.stdout, type="text"))

            if hasattr(result, "stderr") and result.stderr:
                results.append(TextContent(text=result.stderr, type="text"))

            return results

    except Exception:
        logger.exception("Error creating visualization with files")
        return []


@mcp.resource("sandbox://languages")
def get_sandbox_execution_language_support() -> str:
    """Resource containing detailed information about supported languages."""
    return json.dumps(LANGUAGE_RESOURCES, indent=2)


def main() -> None:
    """Set up and run the server."""
    logger.info("Starting MCP server with backend: %s", get_backend())
    mcp.run()


if __name__ == "__main__":
    main()
