from pydantic import BaseModel, Field, model_validator

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.security import SecurityPolicy


class SessionConfig(BaseModel):
    """Configuration for a sandbox session."""

    # Core settings
    lang: SupportedLanguage = Field(
        default=SupportedLanguage.PYTHON,
        description="The programming language of the code to be run (e.g., 'python', 'java'). "
        "Determines default image and language-specific handlers.",
    )
    image: str | None = Field(
        default=None,
        description="The name of the Docker image to use (e.g., 'ghcr.io/vndee/sandbox-python-311-bullseye'). "
        "If None and `dockerfile` is also None, a default image for the specified `lang` is used.",
    )
    dockerfile: str | None = Field(
        default=None,
        description="The path to a Dockerfile to build an image from. Cannot be used if `image` is also provided.",
    )

    @model_validator(mode="after")
    def validate_image_and_dockerfile(self) -> "SessionConfig":
        """Validate that image and dockerfile are not both provided."""
        if self.image is not None and self.dockerfile is not None:
            msg = "Only one of 'image' or 'dockerfile' can be provided, not both"
            raise ValueError(msg)
        return self

    # Behaviour settings
    verbose: bool = Field(default=False, description="Whether to print verbose output.")
    workdir: str = Field(default="/sandbox", description="The working directory inside the container.")

    # Timeout settings
    default_timeout: float | None = Field(
        default=60.0,
        description="The default timeout for the session. If None, the session will run indefinitely.",
    )
    execution_timeout: float | None = Field(
        default=None,
        description="The timeout for the execution of the code. If None, the execution will run indefinitely.",
    )
    session_timeout: float | None = Field(
        default=None,
        description="The timeout for the session. If None, the session will run indefinitely.",
    )

    # Security and runtime settings
    security_policy: SecurityPolicy | None = Field(
        default=None,
        description="The security policy to use for the session.",
    )
    runtime_configs: dict = Field(
        default={},
        description="Additional configurations for the container runtime, such as resource limits "
        "(e.g., `cpu_count`, `mem_limit`) or user (`user='1000:1000'`). "
        "By default, containers run as the root user for maximum compatibility.",
    )

    def get_execution_timeout(self) -> float | None:
        """Get the execution timeout."""
        return self.execution_timeout or self.default_timeout
