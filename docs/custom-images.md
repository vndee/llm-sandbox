# Custom Docker Images and Pre-installed Libraries

This guide explains how to use custom Docker images and Dockerfiles with pre-installed libraries in LLM Sandbox.

## Overview

LLM Sandbox supports using custom Docker images and Dockerfiles that have libraries pre-installed. This is particularly useful for **Python** environments, where virtual environment isolation can sometimes prevent access to system-wide packages.

**Note**: This guide primarily applies to Python. Other languages (Go, R, Java, etc.) typically don't have the same isolation issues since they don't use virtual environments.

Benefits include:
- Faster execution times (no need to install libraries at runtime)
- Complex dependency setups
- Specific library versions or configurations
- Reproducible environments

## Using Custom Images

### Pre-built Custom Images

You can use any custom Docker image with pre-installed libraries:

```python
from llm_sandbox import SandboxSession

# Using a custom image with pre-installed data science libraries
with SandboxSession(
    lang="python",
    image="your-registry/python-datascience:latest"
) as session:
    # Libraries like pandas, numpy are already available
    result = session.run("""
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(df.to_json())
    """)
    # No need to specify libraries=["pandas", "numpy"]
    print(result.stdout)
```

### Building from Dockerfile

You can also build images from Dockerfiles with pre-installed libraries:

```python
from llm_sandbox import SandboxSession

with SandboxSession(
    lang="python",
    dockerfile="./custom/Dockerfile"
) as session:
    result = session.run("""
import numpy as np
import matplotlib.pyplot as plt

# Create sample data
data = np.random.randn(100)
print(f"Generated {len(data)} data points")
    """)
    print(result.stdout)
```

## Example Dockerfile

Here's an example Dockerfile with pre-installed Python libraries:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-install Python packages
RUN pip install \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    requests \
    fastapi

# Create working directory
WORKDIR /sandbox

# Optional: Create directories for output
RUN mkdir -p /tmp/sandbox_output /tmp/sandbox_plots
```

## How It Works (Python-Specific)

### Virtual Environment with System Packages

For **Python only**, LLM Sandbox creates a virtual environment using the `--system-site-packages` flag:

```bash
python -m venv --system-site-packages /tmp/venv
```

This allows the virtual environment to access packages installed in the system Python (including those in your custom image).

**Other Languages**: Languages like Go, R, Java, C++, etc. don't use virtual environments, so pre-installed libraries in custom images are automatically accessible without any special configuration.

### Library Installation Behavior

1. **Pre-installed libraries**: Available immediately without specifying in `libraries` parameter
2. **Additional libraries**: Can still be installed using the `libraries` parameter
3. **Hybrid approach**: Mix pre-installed and runtime-installed libraries

```python
# Pre-installed: pandas, numpy
# Runtime-installed: requests
result = session.run("""
import pandas as pd  # Pre-installed
import numpy as np   # Pre-installed
import requests      # Will be installed at runtime

data = pd.DataFrame({'x': np.random.randn(10)})
response = requests.get('https://api.github.com')
print(f"Data shape: {data.shape}, API status: {response.status_code}")
""", libraries=["requests"])  # Only need to specify requests
```

## Best Practices

### 1. Layer Optimization

Organize your Dockerfile for optimal layer caching:

```dockerfile
FROM python:3.11-slim

# Install system dependencies first (rarely change)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install stable/core libraries next
RUN pip install \
    numpy \
    pandas \
    matplotlib

# Install more volatile libraries last
RUN pip install \
    scikit-learn \
    requests

WORKDIR /sandbox
```

### 2. Pin Library Versions

For reproducible environments, pin library versions:

```dockerfile
RUN pip install \
    numpy==1.24.3 \
    pandas==2.0.3 \
    matplotlib==3.7.2 \
    scikit-learn==1.3.0
```

### 3. Use Multi-stage Builds

For smaller images, consider multi-stage builds:

```dockerfile
# Build stage
FROM python:3.11 as builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
WORKDIR /sandbox
```

### 4. Cache Dependencies

Set up pip cache for faster builds:

```dockerfile
ENV PIP_CACHE_DIR=/tmp/pip_cache
RUN mkdir -p /tmp/pip_cache
RUN pip install --cache-dir /tmp/pip_cache numpy pandas
```

## Language-Specific Examples

### Python Data Science Image (Requires --system-site-packages Fix)

**Python** requires the virtual environment fix to access pre-installed packages:

```dockerfile
FROM python:3.11-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Data science stack - accessible via --system-site-packages
RUN pip install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    jupyter \
    plotly

WORKDIR /sandbox
```

### Python Web Development Image

```dockerfile
FROM python:3.11-slim

# Pre-installed packages accessible via --system-site-packages
RUN pip install \
    fastapi \
    uvicorn \
    requests \
    pydantic \
    sqlalchemy \
    pytest

WORKDIR /sandbox
```

### R with Bioconductor (No Virtual Environment Issues)

**R** doesn't use virtual environments, so pre-installed packages work automatically:

```dockerfile
FROM rocker/r-ver:4.3.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libxml2-dev \
    libssl-dev \
    libcurl4-openssl-dev

# These packages are immediately accessible - no virtual environment isolation
RUN R -e "install.packages(c('tidyverse', 'data.table', 'plotly'))"
RUN R -e "BiocManager::install(c('DESeq2', 'edgeR'))"

WORKDIR /sandbox
```

## Troubleshooting

### Python Library Not Found

If a pre-installed Python library is not found:

1. **Check virtual environment creation**: Ensure `--system-site-packages` is used
2. **Verify installation path**: Libraries should be in system Python, not user-local
3. **Test directly**: Run `python -c "import library_name"` in your image

### Other Languages

For non-Python languages, pre-installed libraries should work automatically. If they don't:

1. **Verify installation**: Check that packages are properly installed in the image
2. **Check paths**: Ensure library paths are correctly configured
3. **Language-specific issues**: Check language-specific package manager configurations

### Conflicting Libraries

If you get version conflicts:

```python
# Check what's installed
result = session.run("""
import pkg_resources
installed = [str(d) for d in pkg_resources.working_set]
for package in sorted(installed):
    print(package)
""")
```

### Performance Issues

For faster startup:
- Use smaller base images (e.g., `python:3.11-slim` vs `python:3.11`)
- Pre-compile Python files: `RUN python -m compileall /usr/local/lib/python3.11`
- Use package wheels: `RUN pip install --only-binary=all numpy pandas`
