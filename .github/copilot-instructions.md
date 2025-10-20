# GitHub Copilot Instructions for LLM Sandbox

## Project Overview

LLM Sandbox is a lightweight and portable sandbox environment designed to run Large Language Model (LLM) generated code in a safe and isolated mode. The project provides secure execution environments for AI-generated code with:

- **Multi-language support**: Python, JavaScript/Node.js, Java, C++, Go, R, and Ruby
- **Flexible backends**: Docker, Kubernetes, Podman, and Micromamba
- **Security-first design**: Isolated execution, security policies, resource limits, and network isolation
- **LLM framework integration**: Works with LangChain, LangGraph, LlamaIndex, OpenAI, and more
- **Model Context Protocol (MCP)**: Server implementation for MCP clients like Claude Desktop

## Code Style and Formatting

### Python Code Standards

- **Python version**: Support Python 3.10, 3.11, 3.12, and 3.13
- **Style guide**: Follow PEP 8 with Ruff enforced rules (see `pyproject.toml`)
- **Line length**: Maximum 120 characters
- **Type hints**: Always use type hints for function signatures and class attributes
- **Docstrings**: Use Google-style docstrings for all public functions, classes, and modules

Example:
```python
def execute_code(code: str, timeout: int = 30) -> ExecutionResult:
    """Execute code in the sandbox environment.

    Args:
        code: The code to execute
        timeout: Maximum execution time in seconds

    Returns:
        ExecutionResult containing stdout, stderr, and exit code

    Raises:
        TimeoutError: If execution exceeds timeout
        SecurityError: If code violates security policy
    """
    pass
```

### Code Organization

- Use Pydantic models for data structures and configuration
- Follow existing module structure:
  - `llm_sandbox/core/`: Core functionality and base classes
  - `llm_sandbox/language_handlers/`: Language-specific handlers
  - `llm_sandbox/mcp_server/`: MCP server implementation
  - Backend implementations: `docker.py`, `kubernetes.py`, `podman.py`, `micromamba.py`

### Import Style

```python
# Standard library imports
from typing import Any
from pathlib import Path

# Third-party imports
from pydantic import BaseModel, Field

# Local imports
from llm_sandbox.const import SupportedLanguage
from llm_sandbox.exceptions import SandboxError
```

## Testing Practices

### Test Organization

- All tests in `tests/` directory
- Test file naming: `test_<module_name>.py`
- Use pytest for all tests
- Organize tests into classes when testing multiple aspects of a feature

### Test Structure

Follow the Arrange-Act-Assert pattern:

```python
def test_feature_description() -> None:
    """Test description explaining what is being tested."""
    # Arrange
    session = SandboxSession(lang="python")
    code = "print('hello')"

    # Act
    result = session.run(code)

    # Assert
    assert result.stdout == "hello\n"
    assert result.exit_code == 0
```

### Test Fixtures and Mocking

- Use `conftest.py` for shared fixtures
- Mock external dependencies (Docker, Kubernetes, etc.) in unit tests
- Use `@patch` decorator for mocking:

```python
@patch("llm_sandbox.session.find_spec")
@patch("llm_sandbox.docker.docker.from_env")
def test_with_mocks(mock_docker: MagicMock, mock_find_spec: MagicMock) -> None:
    """Test with mocked dependencies."""
    mock_find_spec.return_value = MagicMock()
    # Test implementation
```

### Parametrized Tests

Use `@pytest.mark.parametrize` for testing multiple inputs:

```python
@pytest.mark.parametrize("language,expected_extension", [
    ("python", "py"),
    ("javascript", "js"),
    ("java", "java"),
])
def test_language_extensions(language: str, expected_extension: str) -> None:
    """Test language handler extensions."""
    handler = LanguageHandlerFactory.create_handler(language)
    assert handler.file_extension == expected_extension
```

### Coverage Expectations

- Maintain high test coverage (aim for >80%)
- Test both success and error cases
- Include edge cases and boundary conditions
- Security tests must not make real Docker connections (use `mock_docker_backend` fixture)

## Security Considerations

### Security Policy

The project implements a comprehensive security scanning system:

- **Pattern-based detection**: Identify dangerous code patterns
- **Module restrictions**: Block imports of restricted modules
- **Severity levels**: `SAFE`, `LOW`, `MEDIUM`, `HIGH`
- **Security threshold**: Configurable blocking threshold

When adding security features:

```python
# Define security patterns
pattern = SecurityPattern(
    pattern=r"import\s+os",
    description="Direct OS module access",
    severity=SecurityIssueSeverity.MEDIUM,
)

# Add to security policy
policy = SecurityPolicy(
    severity_threshold=SecurityIssueSeverity.MEDIUM,
    patterns=[pattern],
)
```

### Container Security

- All code execution happens in isolated containers
- Resource limits (CPU, memory, execution time) are enforced
- Network isolation can be configured
- File system access is restricted

### Best Practices

- Never trust user-provided code without security scanning
- Always use security policies for production deployments
- Validate all inputs using Pydantic models
- Handle exceptions gracefully and provide informative error messages
- Clean up containers and resources properly

## Development Workflow

### Setup

```bash
# Clone and setup
git clone https://github.com/vndee/llm-sandbox.git
cd llm-sandbox
make install  # Uses uv for dependency management
```

### Code Quality Checks

Run before committing:

```bash
make check  # Runs all quality checks
```

Individual checks:
- `uv run pre-commit run -a` - Linting and formatting (Ruff)
- `uv run mypy` - Type checking
- `uv run deptry .` - Dependency validation

### Testing

```bash
make test  # Run all tests with coverage
```

### Documentation

```bash
make docs  # Build and serve documentation locally
make docs-test  # Verify documentation builds without errors
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add Ruby language handler`
- `fix: resolve Docker cleanup timeout issue`
- `docs: update security policy examples`
- `test: add Kubernetes backend tests`
- `refactor: simplify session creation logic`
- `feat!: breaking change description` (for breaking changes)

With scope:
- `feat(security): add SQL injection pattern detection`
- `fix(docker): handle container cleanup on timeout`

## Common Patterns

### Creating Language Handlers

```python
class MyLanguageHandler(BaseLanguageHandler):
    """Handler for MyLanguage code execution."""

    name: str = "mylanguage"
    file_extension: str = "ml"
    default_image: str = "mylanguage:latest"

    def get_import_patterns(self, module: str) -> str:
        """Return regex pattern for detecting module imports."""
        return rf"\s*import\s+{re.escape(module)}"

    def filter_comments(self, code: str) -> str:
        """Remove comments from code for security scanning."""
        # Implementation
        pass
```

### Session Management

```python
# Context manager (recommended)
with SandboxSession(lang="python") as session:
    result = session.run("print('hello')")

# Manual management
session = SandboxSession(lang="python")
try:
    result = session.run("print('hello')")
finally:
    session.close()
```

### Error Handling

```python
from llm_sandbox.exceptions import (
    SandboxError,
    SecurityError,
    ResourceError,
    ValidationError,
)

try:
    session.run(code)
except SecurityError as e:
    # Handle security violations
    logger.error(f"Security violation: {e}")
except ResourceError as e:
    # Handle resource limit violations
    logger.error(f"Resource limit exceeded: {e}")
except SandboxError as e:
    # Handle general sandbox errors
    logger.error(f"Sandbox error: {e}")
```

## Dependency Management

- Use `uv` for dependency management (defined in `pyproject.toml`)
- Optional dependencies for different backends: `[docker]`, `[k8s]`, `[podman]`
- MCP server dependencies: `[mcp-docker]`, `[mcp-k8s]`, `[mcp-podman]`
- Development dependencies in `[dependency-groups]`

When adding dependencies:
1. Add to appropriate section in `pyproject.toml`
2. Run `uv lock` to update lock file
3. Update documentation if user-facing

## File Structure Standards

- Configuration classes use Pydantic `BaseModel`
- Data classes in `data.py` (e.g., `ExecutionResult`, `ConsoleOutput`)
- Exceptions in `exceptions.py` with descriptive messages
- Constants in `const.py` using Enums

## Integration Examples

When creating examples or documentation:

```python
# LangChain integration
from langchain.tools import Tool
from llm_sandbox import SandboxSession

def execute_code(code: str) -> str:
    with SandboxSession(lang="python") as session:
        result = session.run(code)
        return result.stdout

code_tool = Tool(
    name="code_executor",
    func=execute_code,
    description="Execute Python code safely",
)
```

## Performance Considerations

- Use `production=True` for faster container startup (skips environment setup)
- Consider `keep_template=True` to reuse container templates
- Set appropriate resource limits to prevent resource exhaustion
- Clean up sessions properly to avoid resource leaks

## Documentation Standards

- Keep README.md up to date with new features
- Document all public APIs
- Provide code examples for new features
- Update `docs/` directory for comprehensive guides
- Include security considerations in documentation

## Additional Notes

- The project supports both synchronous and asynchronous operations
- MCP server implementation enables integration with AI assistants
- Artifact extraction for plots and visualizations is automatic
- Multiple language versions can coexist (e.g., Python 3.10-3.13)
- Container images can be customized for specific use cases
