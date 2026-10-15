"""Run Python offline with NsJail, namespaces, and its native seccomp filter.

Requires Linux with permitted user namespaces and new mount API syscalls,
nsjail with Kafel and --experimental_mnt new support, and /usr/bin/python3
with venv/ensurepip support. No Docker daemon is needed.
"""

import logging

from llm_sandbox import SandboxSession

logger = logging.getLogger(__name__)


def main() -> None:
    """Run code with networking disabled and a private session workspace."""
    with SandboxSession(backend="nsjail", lang="python") as session:
        result = session.run("print('Hello from NsJail + seccomp!')")
        logger.info("%s", result.stdout)
        if result.exit_code:
            logger.error("%s", result.stderr)
            raise SystemExit(result.exit_code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
