# LLM Sandbox - AI Assistant Guide

This document provides context and guidance for AI assistants working with the LLM Sandbox codebase.

## Project Overview

**LLM Sandbox** is a lightweight and portable sandbox environment designed to run Large Language Model (LLM) generated code in a safe and isolated mode. It provides secure execution environments for AI-generated code while offering flexibility in container backends and comprehensive language support.

**Version:** 0.3.13
**License:** MIT
**Python Support:** 3.10+
**Documentation:** https://vndee.github.io/llm-sandbox/
**Repository:** https://github.com/vndee/llm-sandbox

## Core Architecture

### Key Components

1. **Session Management** (`llm_sandbox/session.py`)
   - `SandboxSession`: Basic code execution in isolated containers
   - `ArtifactSandboxSession`: Execution with automatic artifact capture (plots, visualizations)
   - `InteractiveSandboxSession`: Stateful sessions maintaining Python interpreter state
   - `create_session()`: Factory function for session creation

2. **Container Backends** (`llm_sandbox/docker.py`, `llm_sandbox/kubernetes.py`, `llm_sandbox/podman.py`)
   - Docker (default, most common)
   - Kubernetes (enterprise-grade orchestration)
   - Podman (rootless containers)

3. **Language Handlers** (`llm_sandbox/language_handlers/`)
   - Python (`python_handler.py`)
   - JavaScript/Node.js (`javascript_handler.py`)
   - Java (`java_handler.py`)
   - C++ (`cpp_handler.py`)
   - Go (`go_handler.py`)
   - R (`r_handler.py`)
   - Ruby (`ruby_handler.py`)

4. **Container Pooling** (`llm_sandbox/pool/`)
   - Pre-warmed container management for performance
   - Thread-safe concurrent execution
   - Configurable pool sizes and lifecycle management

5. **Security** (`llm_sandbox/security.py`)
   - Security policies and pattern matching
   - Code scanning for vulnerabilities
   - Resource limits and network isolation

6. **MCP Server** (`llm_sandbox/mcp_server/`)
   - Model Context Protocol integration
   - Enables Claude Desktop and other MCP clients to execute code

## Project Structure

```
llm-sandbox/
├── llm_sandbox/              # Main package
│   ├── __init__.py          # Public API exports
│   ├── session.py           # Session implementations
│   ├── docker.py            # Docker backend
│   ├── kubernetes.py        # Kubernetes backend
│   ├── podman.py            # Podman backend
│   ├── security.py          # Security policies
│   ├── interactive.py       # Interactive session support
│   ├── data.py              # Data models (ExecutionResult, PlotOutput)
│   ├── const.py             # Constants and enums
│   ├── exceptions.py        # Custom exceptions
│   ├── core/                # Core abstractions
│   │   ├── config.py        # Configuration models
│   │   ├── session_base.py  # Base session class
│   │   └── mixins.py        # Shared functionality
│   ├── language_handlers/   # Language-specific handlers
│   │   ├── base.py          # Base handler interface
│   │   ├── python_handler.py
│   │   ├── javascript_handler.py
│   │   ├── java_handler.py
│   │   ├── cpp_handler.py
│   │   ├── go_handler.py
│   │   ├── r_handler.py
│   │   ├── ruby_handler.py
│   │   └── artifact_detection/  # Plot/artifact detection
│   ├── pool/                # Container pooling
│   │   ├── base.py          # Base pool manager
│   │   ├── config.py        # Pool configuration
│   │   ├── docker_pool.py   # Docker pool implementation
│   │   ├── kubernetes_pool.py
│   │   ├── podman_pool.py
│   │   ├── session.py       # Pooled session wrapper
│   │   └── factory.py       # Pool factory
│   └── mcp_server/          # MCP integration
│       ├── server.py        # MCP server implementation
│       ├── types.py         # MCP data types
│       └── const.py         # MCP constants
├── tests/                   # Test suite
├── examples/                # Usage examples
├── docs/                    # Documentation source
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # User-facing documentation
└── CLAUDE.md               # This file

```

## Development Guidelines

### Code Style

- **Formatter:** Ruff (line length: 120)
- **Type Checking:** Mypy with strict settings
- **Testing:** pytest with integration tests for backends
- **Pre-commit hooks:** Configured for automated checks

### Key Design Patterns

1. **Context Managers**: All sessions use context managers for proper resource cleanup
2. **Factory Pattern**: `create_session()` and `create_pool_manager()` for object creation
3. **Strategy Pattern**: Language handlers implement common interface
4. **Pool Pattern**: Container pooling for performance optimization

### Testing Strategy

- Unit tests for core logic
- Integration tests marked with `@pytest.mark.integration`
- Tests require Docker/Podman/Kubernetes depending on backend
- Run tests: `make test` or `pytest tests/`

## Common Tasks for AI Assistants

### Adding a New Language Handler

1. Create handler in `llm_sandbox/language_handlers/`
2. Inherit from `BaseLanguageHandler`
3. Implement required methods:
   - `prepare_code()`
   - `get_install_command()`
   - `get_execution_command()`
4. Register in `factory.py`
5. Add to `SupportedLanguage` enum in `const.py`
6. Add tests in `tests/`

### Improving Security Features

- Security patterns defined in `security.py`
- Code scanning happens before execution
- Add new patterns to `SecurityPattern` dataclass
- Test with `tests/test_security_*.py`

### Enhancing Container Pooling

- Pool managers in `llm_sandbox/pool/`
- Configuration in `pool/config.py`
- Three exhaustion strategies: WAIT, FAIL_FAST, TEMPORARY
- Thread-safe implementation required

### Adding MCP Tools

- MCP server in `llm_sandbox/mcp_server/server.py`
- Tools defined using MCP decorators
- Must handle async operations
- Test with MCP client (e.g., Claude Desktop)

## Important Conventions

### Session Lifecycle

```python
# Always use context managers
with SandboxSession(lang="python") as session:
    result = session.run("print('hello')")
    # Container automatically cleaned up on exit
```

### Error Handling

- `SandboxError`: Base exception
- `ContainerError`: Container-related issues
- `SecurityError`: Security violations
- `ResourceError`: Resource exhaustion
- `ValidationError`: Input validation failures

### Library Installation

```python
# Libraries installed automatically when specified
session.run(code, libraries=["numpy", "pandas"])
```

### Artifact Extraction

```python
# Plots and visualizations automatically captured
with ArtifactSandboxSession(lang="python") as session:
    result = session.run("plt.plot([1,2,3]); plt.show()")
    for plot in result.plots:
        # plot.content_base64 contains image data
        pass
```

## Integration Points

### LangChain

```python
from langchain.tools import BaseTool
from llm_sandbox import SandboxSession

class PythonSandboxTool(BaseTool):
    name = "python_sandbox"
    description = "Execute Python code"

    def _run(self, code: str) -> str:
        with SandboxSession(lang="python") as session:
            result = session.run(code)
            return result.stdout
```

### OpenAI Functions

Define as function with parameters matching `SandboxSession.run()` signature.

### LlamaIndex

Use as custom tool with `SandboxSession` wrapper.

### MCP Clients

Configure in client (e.g., `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "llm-sandbox": {
      "command": "python3",
      "args": ["-m", "llm_sandbox.mcp_server.server"]
    }
  }
}
```

## Performance Considerations

1. **Use Container Pooling** for frequent executions (10x faster)
2. **Pre-install Libraries** in custom images or pool configuration
3. **Reuse Sessions** when possible (especially `InteractiveSandboxSession`)
4. **Set Appropriate Resource Limits** to prevent resource exhaustion
5. **Choose Right Backend**: Docker (development), Kubernetes (production scale)

## Security Best Practices

1. Always run untrusted code in sandbox
2. Set resource limits (CPU, memory, execution time)
3. Configure network isolation when needed
4. Use security policies for sensitive operations
5. Regularly update container images
6. Monitor container metrics in production

## Debugging Tips

1. **Enable Verbose Mode**: `SandboxSession(verbose=True)`
2. **Keep Containers**: `keep_template=True` for inspection
3. **Check Logs**: Container logs available via backend APIs
4. **Test Backends**: Verify Docker/Kubernetes/Podman availability
5. **Review Security Scanner**: Check for flagged patterns in code

## Common Pitfalls

1. **Not using context managers** - leads to resource leaks
2. **Forgetting to specify language** - required parameter
3. **Large library installations** - use custom images instead
4. **Not handling timeout errors** - set appropriate timeouts
5. **Mixing pool and non-pool sessions** - be consistent

## Configuration Files

### pyproject.toml

- Project metadata
- Dependencies (base + optional extras)
- Development dependencies
- Tool configurations (mypy, pytest, ruff)

### Key Dependencies

- **Required**: pydantic (data validation)
- **Docker**: docker (Docker API)
- **Kubernetes**: kubernetes (K8s API)
- **Podman**: docker + podman
- **MCP**: mcp (Model Context Protocol)

## Recent Changes

Check `git log` for recent commits. Notable features:

- Container pooling system (v0.3+)
- Interactive sessions with IPython kernel
- MCP server integration
- Multi-language support expansion (R, Ruby)
- Artifact detection improvements
- Security policy enhancements

## Resources

- **Documentation**: https://vndee.github.io/llm-sandbox/
- **GitHub**: https://github.com/vndee/llm-sandbox
- **PyPI**: https://pypi.org/project/llm-sandbox/
- **Issues**: https://github.com/vndee/llm-sandbox/issues
- **Discussions**: https://github.com/vndee/llm-sandbox/discussions

## Working with This Codebase

### Before Making Changes

1. Read existing code in the relevant module
2. Check tests for usage patterns
3. Review documentation for context
4. Consider security implications
5. Maintain backward compatibility when possible

### Adding Features

1. Start with tests (TDD approach when appropriate)
2. Update documentation
3. Add examples if user-facing
4. Consider performance impact
5. Test with all supported backends (if applicable)

### Bug Fixes

1. Add regression test first
2. Identify root cause
3. Fix minimal code necessary
4. Verify fix doesn't break other tests
5. Update documentation if behavior changes

## Questions to Ask

When implementing features, consider:

- Which backends does this affect?
- Are there security implications?
- How does this interact with pooling?
- Does this need language-specific handling?
- What are the performance characteristics?
- How should errors be handled?
- Is this a breaking change?

## Contact

- **Author**: Duy Huynh (vndee.huynh@gmail.com)
- **Issues**: GitHub Issues
- **Community**: GitHub Discussions
