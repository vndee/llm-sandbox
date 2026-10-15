"""Run native Windows Python with an installed MXC helper and PSEC-capable host."""

import logging

from llm_sandbox import SandboxSession, WindowsMxcConfig

logger = logging.getLogger(__name__)


def main() -> None:
    """Use the bundled helper and current Python installation without networking."""
    with SandboxSession(
        backend="windows-mxc",
        lang="python",
        sandbox_config=WindowsMxcConfig(network="none"),
        execution_timeout=20.0,
    ) as session:
        result = session.run("import platform; print(platform.system()); print('Hello from native MXC')")
        logger.info("%s", result.stdout)
        if result.exit_code:
            logger.error("%s", result.stderr)
            raise SystemExit(result.exit_code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
