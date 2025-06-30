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
