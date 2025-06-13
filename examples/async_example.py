"""Async execution example for LLM Sandbox."""

import asyncio
from typing import Any, Protocol

from llm_sandbox import SandboxSession


class CodeExecutor(Protocol):
    """Protocol for code execution integration."""

    def execute(
        self, code: str, language: str = "python", libraries: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Execute code and return results."""
        ...


class SandboxCodeExecutor:
    """Sandbox implementation of CodeExecutor."""

    def __init__(self, default_security_policy: Any = None) -> None:
        """Initialize the executor with optional security policy."""
        self.default_security_policy = default_security_policy

    def execute(
        self,
        code: str,
        language: str = "python",
        libraries: list[str] | None = None,
        security_policy: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Execute code in sandbox.

        Returns:
            Dictionary with stdout, stderr, exit_code, and plots

        """
        policy = security_policy or self.default_security_policy

        try:
            with SandboxSession(lang=language, security_policy=policy, **kwargs) as session:
                result = session.run(code, libraries)

                return {
                    "success": result.exit_code == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "plots": getattr(result, "plots", []),
                }
        except Exception as e:  # noqa: BLE001
            return {"success": False, "error": str(e), "exit_code": -1}

    async def execute_async(self, code: str, **kwargs: Any) -> dict[str, Any]:
        """Async wrapper for execution."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.execute(code, **kwargs))


async def main() -> None:
    """Demonstrate both sync and async execution."""
    # Use in any framework
    executor = SandboxCodeExecutor(default_security_policy=None)

    # Sync execution
    result = executor.execute("print('Hello, World!')", language="python")
    print(f"Sync result: {result['stdout']}")  # noqa: T201

    # Async execution
    result = await executor.execute_async("print('Hello, Async!')", language="python")
    print(f"Async result: {result['stdout']}")  # noqa: T201


# Run async
if __name__ == "__main__":
    asyncio.run(main())
