#!/usr/bin/env python3

import docker

from llm_sandbox.session import ArtifactSandboxSession, SandboxBackend

# Simple test to reproduce the ggplot2 issue
r_code = """
library(ggplot2)

# Simple test plot
data(mtcars)
p <- ggplot(mtcars, aes(x = wt, y = mpg)) + geom_point()
print(p)
"""

print("Testing R ggplot2 issue reproduction...")

client = docker.DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")

try:
    with ArtifactSandboxSession(
        client=client,
        lang="r",
        image="ghcr.io/vndee/sandbox-r-451-bullseye",
        verbose=True,
        backend=SandboxBackend.DOCKER,
        enable_plotting=True,
        keep_template=True,
    ) as session:
        print("Running simple ggplot2 test...")

        # Test with ggplot2 - all in one session
        result = session.run(r_code, libraries=["ggplot2"])

        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"Plots captured: {len(result.plots)}")

except Exception as e:
    print(f"Error: {e}")
