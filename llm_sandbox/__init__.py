"""LLM Sandbox - A lightweight and portable LLM sandbox runtime."""

from .const import DefaultImage, SandboxBackend, SupportedLanguage
from .core.config import SessionConfig
from .data import ConsoleOutput, ExecutionResult, FileType, PlotOutput
from .exceptions import ContainerError, ResourceError, SandboxError, SecurityError, ValidationError
from .security import SecurityIssueSeverity, SecurityPattern, SecurityPolicy
from .session import ArtifactSandboxSession, SandboxSession, create_session

try:
    from .kubernetes_pool import KubernetesPodPool

    _HAS_KUBERNETES_POOL = True
except ImportError:
    _HAS_KUBERNETES_POOL = False

__all__ = [
    "ArtifactSandboxSession",
    "ConsoleOutput",
    "ContainerError",
    "DefaultImage",
    "ExecutionResult",
    "FileType",
    "PlotOutput",
    "ResourceError",
    "SandboxBackend",
    "SandboxError",
    "SandboxSession",
    "SecurityError",
    "SecurityIssueSeverity",
    "SecurityPattern",
    "SecurityPolicy",
    "SessionConfig",
    "SupportedLanguage",
    "ValidationError",
    "create_session",
]

if _HAS_KUBERNETES_POOL:
    __all__ += ["KubernetesPodPool"]
