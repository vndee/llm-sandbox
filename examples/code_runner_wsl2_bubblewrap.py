"""Run from Windows after installing the matching package in a WSL2 guest."""

import argparse
import logging

from llm_sandbox import LocalSandboxConfig, SandboxSession

logger = logging.getLogger(__name__)


def main() -> None:
    """Select a configured distribution and its trusted worker interpreter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", required=True, help="Exact WSL2 distribution name")
    parser.add_argument(
        "--guest-python", default="/usr/bin/python3", help="Absolute guest Python with llm-sandbox installed"
    )
    args = parser.parse_args()
    with SandboxSession(
        backend="wsl2-bubblewrap",
        distribution=args.distribution,
        guest_python=args.guest_python,
        sandbox_config=LocalSandboxConfig(network="none"),
        execution_timeout=20.0,
    ) as session:
        result = session.run("import platform; print(platform.system()); print('Hello from WSL2 Bubblewrap')")
        logger.info("%s", result.stdout)
        if result.exit_code:
            logger.error("%s", result.stderr)
            raise SystemExit(result.exit_code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
