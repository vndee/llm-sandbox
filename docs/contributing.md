# Contributing to LLM Sandbox

Thank you for your interest in contributing to LLM Sandbox! This guide will help you get started with contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) for dependency management
- Docker (for testing Docker backend)
- Git

### Setting Up Development Environment

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/llm-sandbox.git
cd llm-sandbox
```

2. **Install dependencies using uv**

```bash
make install
```

This will:

- Create a virtual environment
- Install all dependencies
- Install pre-commit hooks

3. **Verify installation**

```bash
make test
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

Follow these guidelines:

- Write clean, readable code
- Follow PEP 8 style guide
- Add type hints where appropriate
- Include docstrings for all public functions/classes
- Keep commits focused and atomic

### 3. Run Quality Checks

```bash
# Run all checks
make check

# Individual checks
uv run pre-commit run -a  # Linting and formatting
uv run mypy              # Type checking
uv run deptry .          # Dependency checking
```

### 4. Write Tests

All new features should include tests:

```python
# tests/test_your_feature.py
import pytest
from llm_sandbox import YourFeature

def test_your_feature():
    """Test description"""
    # Arrange
    feature = YourFeature()

    # Act
    result = feature.do_something()

    # Assert
    assert result == expected_value

@pytest.mark.parametrize("input,expected", [
    ("input1", "output1"),
    ("input2", "output2"),
])
def test_parametrized(input, expected):
    """Test with multiple inputs"""
    assert process(input) == expected
```

Run tests:

```bash
make test
```

### 5. Update Documentation

If your changes affect the public API:

1. Update docstrings
2. Update relevant documentation in `docs/`
3. Add examples if applicable

Build and preview documentation:

```bash
make docs
# Visit http://localhost:8000
```

### 6. Commit Your Changes

Write meaningful commit messages following the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```bash
# Good
git commit -m "feat: add support for Ruby language handler"
git commit -m "fix: resolve memory leak in Docker session cleanup"
git commit -m "docs: update security policy examples"
git commit -m "test: add unit tests for Kubernetes backend"
git commit -m "refactor: simplify session factory logic"

# With scope
git commit -m "feat(security): add new pattern detection for SQL injection"
git commit -m "fix(docker): handle container cleanup on session timeout"

# Breaking changes
git commit -m "feat!: remove deprecated session.execute() method"
git commit -m "feat(api)!: change SecurityPolicy constructor signature"

# Bad
git commit -m "fixed stuff"
git commit -m "updates"
```

**Conventional Commit Format:**
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `build`: Changes that affect the build system or external dependencies
- `ci`: Changes to our CI configuration files and scripts
- `chore`: Other changes that don't modify src or test files
- `revert`: Reverts a previous commit

**Breaking Changes:**
- Add `!` after the type/scope to indicate breaking changes
- Or include `BREAKING CHANGE:` in the footer

**Examples with body and footer:**
```bash
git commit -m "feat(lang): add Rust language support

Add comprehensive Rust language handler with cargo support.
Includes compilation and execution pipeline for .rs files.

Closes #123"

git commit -m "fix(security): prevent code injection in eval patterns

The previous regex pattern allowed certain escape sequences
that could bypass security restrictions.

BREAKING CHANGE: SecurityPattern.pattern property is now read-only"
```

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:

- Clear description of changes
- Link to related issues
- Screenshots/examples if applicable

## Project Structure

```
llm-sandbox/
├── llm_sandbox/           # Main package
│   ├── __init__.py
│   ├── base.py           # Base classes
│   ├── session.py        # Session management
│   ├── docker.py         # Docker backend
│   ├── kubernetes.py     # Kubernetes backend
│   ├── podman.py         # Podman backend
│   ├── security.py       # Security policies
│   ├── data.py           # Data classes
│   ├── const.py          # Constants
│   ├── exceptions.py     # Custom exceptions
│   └── language_handlers/ # Language support
│       ├── base.py
│       ├── python_handler.py
│       ├── javascript_handler.py
│       └── ...
├── tests/                # Test files
├── examples/             # Example scripts
├── docs/                 # Documentation
└── pyproject.toml        # Project configuration
```

## Adding New Features

### Adding a New Language

1. Create handler in `llm_sandbox/language_handlers/`:

```python
# llm_sandbox/language_handlers/rust_handler.py
from .base import AbstractLanguageHandler, LanguageConfig

class RustHandler(AbstractLanguageHandler):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.config = LanguageConfig(
            name="rust",
            file_extension="rs",
            execution_commands=["rustc {file} -o /tmp/program && /tmp/program"],
            package_manager="cargo add",
        )

    def get_import_patterns(self, module: str) -> str:
        return rf"use\s+{module}"

    # Implement other required methods...
```

2. Register in factory:

```python
# llm_sandbox/language_handlers/factory.py
from .rust_handler import RustHandler

class LanguageHandlerFactory:
    _handlers = {
        # ...
        "rust": RustHandler,
    }
```

3. Update constants:

```python
# llm_sandbox/const.py
class SupportedLanguage(StrEnum):
    # ...
    RUST = "rust"

class DefaultImage:
    # ...
    RUST = "rust:latest"
```

4. Add tests:

```python
# tests/test_rust_handler.py
def test_rust_execution():
    with SandboxSession(lang="rust") as session:
        result = session.run('fn main() { println!("Hello"); }')
        assert result.stdout.strip() == "Hello"
```

### Adding a New Backend

1. Create backend implementation:

```python
# llm_sandbox/new_backend.py
from .base import Session

class SandboxNewBackendSession(Session):
    def open(self):
        # Implementation
        pass

    def close(self):
        # Implementation
        pass

    # Implement all abstract methods...
```

2. Update session factory:

```python
# llm_sandbox/session.py
def create_session(backend, **kwargs):
    match backend:
        # ...
        case SandboxBackend.NEW_BACKEND:
            from .new_backend import SandboxNewBackendSession
            return SandboxNewBackendSession(**kwargs)
```

3. Add to constants:

```python
# llm_sandbox/const.py
class SandboxBackend(StrEnum):
    # ...
    NEW_BACKEND = "new_backend"
```

## Testing Guidelines

### Test Structure

```python
# Use pytest fixtures for reusable components
@pytest.fixture
def sandbox_session():
    with SandboxSession(lang="python") as session:
        yield session

# Group related tests
class TestSecurityPolicies:
    def test_pattern_detection(self):
        # Test implementation
        pass

    def test_module_restriction(self):
        # Test implementation
        pass

# Use parametrize for multiple test cases
@pytest.mark.parametrize("backend", [
    SandboxBackend.DOCKER,
    SandboxBackend.PODMAN,
])
def test_cross_backend(backend):
    # Test implementation
    pass
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

@patch('docker.from_env')
def test_docker_connection(mock_docker):
    mock_client = Mock()
    mock_docker.return_value = mock_client

    session = SandboxSession(backend=SandboxBackend.DOCKER)
    # Test implementation
```

## Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def execute_code(code: str, language: str = "python") -> ConsoleOutput:
    """Execute code in a sandboxed environment.

    This function creates an isolated container environment and
    executes the provided code safely.

    Args:
        code: The source code to execute
        language: Programming language identifier

    Returns:
        ConsoleOutput containing stdout, stderr, and exit code

    Raises:
        SecurityError: If code violates security policy
        NotOpenSessionError: If session is not initialized

    Example:
        >>> result = execute_code("print('Hello')")
        >>> print(result.stdout)
        Hello
    """
```

### Type Hints

Always include type hints:

```python
from typing import Optional, List, Dict, Union, Any
from typing import Protocol  # for protocols

def process_data(
    data: List[Dict[str, Any]],
    options: Optional[Dict[str, Union[str, int]]] = None
) -> Dict[str, Any]:
    """Process data with options."""
    pass
```

## Performance Guidelines

### Optimization Tips

1. **Lazy Imports**: Import heavy dependencies only when needed

```python
def use_special_feature():
    # Import only when function is called
    import heavy_library
    return heavy_library.process()
```

2. **Resource Management**: Always use context managers

```python
# Good
with SandboxSession() as session:
    result = session.run(code)

# Avoid
session = SandboxSession()
session.open()
result = session.run(code)
session.close()  # May not be called if error occurs
```

3. **Caching**: Cache expensive operations

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_language_handler(language: str):
    return LanguageHandlerFactory.create_handler(language)
```

## Security Considerations

When contributing security-related features:

1. **Never Trust User Input**: Always validate and sanitize
2. **Principle of Least Privilege**: Request minimum permissions
3. **Defense in Depth**: Layer security measures
4. **Fail Secure**: Default to denying access
5. **Log Security Events**: But don't log sensitive data

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag: `git tag v0.x.x`
4. Push tag: `git push origin v0.x.x`
5. GitHub Actions will handle PyPI release

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Email**: vndee.huynh@gmail.com

## Recognition

Contributors are recognized in:

- CONTRIBUTORS.md file
- GitHub contributors page
- Release notes

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to LLM Sandbox! Your efforts help make code execution safer for everyone using LLMs.
