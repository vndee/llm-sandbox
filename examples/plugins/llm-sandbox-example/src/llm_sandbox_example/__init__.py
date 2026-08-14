"""Reference backend plugin for llm-sandbox.

Copy this package as the starting point for a real backend. See the README for what to
change.

Warning:
    This backend executes code directly on the host machine and provides no isolation
    whatsoever. It exists to demonstrate the plugin interface end to end.

"""

from llm_sandbox_example.backend import ExampleBackend

__all__ = ["ExampleBackend"]
