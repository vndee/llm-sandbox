"""Run Python offline with Bubblewrap, namespaces, and the default seccomp filter.

Requires Linux with permitted user namespaces, bwrap, system libseccomp, and
/usr/bin/python3 with venv/ensurepip support. No Docker daemon is needed.
"""

import logging

from llm_sandbox import SandboxSession

logger = logging.getLogger(__name__)


def main() -> None:
    """Run code with networking disabled and a private session workspace."""
    with SandboxSession(backend="bubblewrap", lang="python") as session:
        result = session.run("print('Hello from Bubblewrap + seccomp!')")
        logger.info("%s", result.stdout)
        if result.exit_code:
            logger.error("%s", result.stderr)
            raise SystemExit(result.exit_code)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
