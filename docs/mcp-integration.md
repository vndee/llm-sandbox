# Model Context Protocol (MCP) Integration

LLM Sandbox provides a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that enables AI assistants like Claude Desktop to execute code securely in sandboxed environments. This integration allows LLMs to run code directly with automatic visualization capture and multi-language support.

## Features

- **Secure Code Execution**: Execute code in isolated containers with your preferred backend
- **Multi-Language Support**: Run Python, JavaScript, Java, C++, Go, R, and Ruby code
- **Automatic Visualization Capture**: Automatically capture and return plots and visualizations
- **Library Management**: Install packages and dependencies on-the-fly
- **Flexible Backend Support**: Choose from Docker, Podman, or Kubernetes backends

## Installation

Install LLM Sandbox with MCP support using your preferred backend:

```bash
# For Docker backend
pip install 'llm-sandbox[mcp-docker]'

# For Podman backend
pip install 'llm-sandbox[mcp-podman]'

# For Kubernetes backend
pip install 'llm-sandbox[mcp-k8s]'
```

## Configuration

Add the following configuration to your MCP client (e.g., `claude_desktop_config.json` for Claude Desktop):

```json
{
  "mcpServers": {
    "llm-sandbox": {
      "command": "python3",
      "args": ["-m", "llm_sandbox.mcp_server.server"],
    }
  }
}
```

### Backend-Specific Configuration

For specific backends, set the `BACKEND` environment variable:

**Docker (default):**
```json
{
  "mcpServers": {
    "llm-sandbox": {
      "command": "python3",
      "args": ["-m", "llm_sandbox.mcp_server.server"],
      "env": {
        "BACKEND": "docker"
      }
    }
  }
}
```

**Podman:**
```json
{
  "mcpServers": {
    "llm-sandbox": {
      "command": "python3",
      "args": ["-m", "llm_sandbox.mcp_server.server"],
      "env": {
        "BACKEND": "podman"
      }
    }
  }
}
```

**Kubernetes:**
```json
{
  "mcpServers": {
    "llm-sandbox": {
      "command": "python3",
      "args": ["-m", "llm_sandbox.mcp_server.server"],
      "env": {
        "BACKEND": "kubernetes"
      }
    }
  }
}
```

**Kubernetes with Custom Namespace:**
```json
{
  "mcpServers": {
    "llm-sandbox": {
      "command": "python3",
      "args": ["-m", "llm_sandbox.mcp_server.server"],
      "env": {
        "BACKEND": "kubernetes",
        "NAMESPACE": "llm-sandbox"
      }
    }
  }
}
```

## Troubleshooting

If you encounter connection issues with your backend, you may need to specify additional environment variables:

**Docker Connection Issues:**
```json
{
  "mcpServers": {
    "llm-sandbox": {
      "command": "python3",
      "args": ["-m", "llm_sandbox.mcp_server.server"],
      "env": {
        "BACKEND": "docker",
        "DOCKER_HOST": "unix:///var/run/docker.sock"
      }
    }
  }
}
```

**Podman Connection Issues:**
```json
{
  "mcpServers": {
    "llm-sandbox": {
      "command": "python3",
      "args": ["-m", "llm_sandbox.mcp_server.server"],
      "env": {
        "BACKEND": "podman",
        "DOCKER_HOST": "unix:///var/run/podman/podman.sock"
      }
    }
  }
}
```

**Kubernetes Connection Issues:**
```json
{
  "mcpServers": {
    "llm-sandbox": {
      "command": "python3",
      "args": ["-m", "llm_sandbox.mcp_server.server"],
      "env": {
        "BACKEND": "kubernetes",
        "KUBECONFIG": "/path/to/your/kubeconfig"
      }
    }
  }
}
```

**Common Environment Variables:**

- `DOCKER_HOST`: Specify the Docker daemon socket (default: `unix:///var/run/docker.sock`)
- `KUBECONFIG`: Path to your Kubernetes configuration file
- `BACKEND`: Choose your container backend (`docker`, `podman`, or `kubernetes`)
- `COMMIT_CONTAINER`: Control whether container changes are saved to the image (default: `false`)
- `KEEP_TEMPLATE`: Control whether template images are preserved between sessions (default: `true`)
- `NAMESPACE`: Specify Kubernetes namespace for pod creation (default: `default`)

### Persistence and security

The MCP server runs code that an AI client supplied, which means the code is effectively
attacker-controlled from the sandbox's point of view. Persisting state from one request
into the next can carry side effects of one request into another, so the defaults are
chosen to make persistence opt-in:

- `COMMIT_CONTAINER` defaults to `false`. When you opt in, the server commits to a
  unique tag of the form `llm-sandbox-mcp/<lang>:<short-hex>` instead of overwriting
  the source image's tag. The pristine source image (e.g. `python:3.11-bullseye`) is
  never modified, so a poisoned commit cannot silently land in a future session that
  pulls the source tag.
- `KEEP_TEMPLATE` defaults to `true`. The template image here is the upstream base
  image, not container state, so reusing it between requests is safe and avoids
  re-pulling on every call. If you would rather pay the pull cost for stricter
  hygiene, set `KEEP_TEMPLATE=false`.

If you want commit support, opt in explicitly:

```json
{
  "mcpServers": {
    "llm-sandbox": {
      "command": "python3",
      "args": ["-m", "llm_sandbox.mcp_server.server"],
      "env": {
        "BACKEND": "docker",
        "COMMIT_CONTAINER": "true"
      }
    }
  }
}
```

To also drop the template image after each session:

```json
{
  "mcpServers": {
    "llm-sandbox": {
      "command": "python3",
      "args": ["-m", "llm_sandbox.mcp_server.server"],
      "env": {
        "BACKEND": "docker",
        "KEEP_TEMPLATE": "false"
      }
    }
  }
}
```

The Kubernetes and Podman backends ignore the unique-tag commit logic; only Docker
and Micromamba support `commit_image_tag` in this version.

### Runtime Configuration via Environment Variables

For Docker, Podman, and Micromamba backends, the MCP server can translate `SANDBOX_*` environment
variables into `runtime_configs` for every sandbox session it creates.

```json
{
  "mcpServers": {
    "llm-sandbox": {
      "command": "python3",
      "args": ["-m", "llm_sandbox.mcp_server.server"],
      "env": {
        "BACKEND": "podman",
        "DOCKER_HOST": "unix:///run/podman/podman.sock",
        "SANDBOX_NETWORK_MODE": "none",
        "SANDBOX_READ_ONLY": "true",
        "SANDBOX_CAP_DROP": "ALL",
        "SANDBOX_SECURITY_OPT": "no-new-privileges:true",
        "SANDBOX_MEMORY": "4g",
        "SANDBOX_CPU_COUNT": "1"
      }
    }
  }
}
```

Supported environment variables:

- `SANDBOX_NETWORK_MODE` -> `runtime_configs["network_mode"]`
- `SANDBOX_READ_ONLY` -> `runtime_configs["read_only"]`
- `SANDBOX_MEMORY` -> `runtime_configs["mem_limit"]`
- `SANDBOX_MEM_LIMIT` -> `runtime_configs["mem_limit"]`
- `SANDBOX_CPUS` -> normalized CPU runtime configs (`cpu_period` and `cpu_quota`)
- `SANDBOX_CPU_COUNT` -> normalized CPU runtime configs (`cpu_period` and `cpu_quota`)
- `SANDBOX_CAP_DROP` -> `runtime_configs["cap_drop"]` (comma-separated)
- `SANDBOX_SECURITY_OPT` -> `runtime_configs["security_opt"]` (comma-separated)
- `SANDBOX_PRIVILEGED` -> `runtime_configs["privileged"]`

> Security note: prefer `SANDBOX_NETWORK_MODE=none`, `SANDBOX_READ_ONLY=true`,
> `SANDBOX_CAP_DROP=ALL`, and restrictive `SANDBOX_SECURITY_OPT` values for hardened
> sandboxes. Avoid `SANDBOX_PRIVILEGED=true` unless you explicitly need it, keep CPU
> and memory limits minimal, and audit any combination that re-enables privileges.
> `SANDBOX_*` settings do not apply to the Kubernetes backend in this MCP server.
> If you need Kubernetes-specific resource or security controls, provide a custom
> `pod_manifest` via a custom MCP wrapper or use the llm-sandbox Python API directly.

**Environment Variable Values:**
Both `COMMIT_CONTAINER` and `KEEP_TEMPLATE` accept:
- `"true"`, `"1"`, `"yes"`, `"on"` → `True`
- `"false"`, `"0"`, `"no"`, `"off"` → `False`

**Use Cases:**
- `COMMIT_CONTAINER=true`: Persist incremental container state across MCP requests
  (e.g. shared package installs). Commits land on a unique tag, never on the source image.
- `KEEP_TEMPLATE=false`: Drop the template image after every session. Slower (re-pulls
  on each call) but leaves nothing behind on disk.
- `NAMESPACE="custom-namespace"`: Organize Kubernetes pods in specific namespaces for multi-tenant environments

## Available Tools

The MCP server provides the following tools:

### execute_code

Execute code in a secure sandbox environment with automatic visualization capture.

**Parameters:**

- `code` (string): The code to execute
- `language` (string): Programming language (python, javascript, java, cpp, go, r, ruby)
- `libraries` (array, optional): List of libraries/packages to install
- `timeout` (integer, optional): Execution timeout in seconds (default: 30)

**Returns:**
List of content items including execution results and any generated visualizations.

### get_supported_languages

Get the list of supported programming languages.

**Returns:**
JSON array of supported language names.

### get_language_details

Get detailed information about a specific programming language.

**Parameters:**

- `language` (string): The language to get details for

**Returns:**
JSON object with language details including version, package manager, examples, and capabilities.

## Available Resources

### language_details

Resource endpoint `sandbox://languages` that provides comprehensive information about all supported languages including their capabilities, examples, and configuration options.

## Usage Examples

Once configured, you can ask your AI assistant to run code, and it will automatically use the LLM Sandbox MCP server:

### Basic Code Execution
```text
"Write a Python function to calculate the factorial of a number and test it with n=5"
```

### Data Visualization
```text
"Create a scatter plot showing the relationship between x and y data points using matplotlib"
```

### Multi-Language Support
```text
"Write a JavaScript function to sort an array of numbers and demonstrate it"
```

The assistant will execute the code in a secure sandbox and automatically capture any generated plots or visualizations.

## Development and Testing

For development and testing of the MCP server:

```bash
# Install in development mode
pip install -e '.[mcp-docker]'

# Run the MCP server directly
python -m llm_sandbox.mcp_server.server

# Test with MCP client tools
# Follow MCP client documentation for testing
```
