"""Base session functionality for LLM Sandbox."""

import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from dataclasses import dataclass

from .exceptions import ContainerError, SecurityError, ResourceError
from .security import SecurityScanner
from .monitoring import ResourceMonitor, ResourceLimits, ResourceUsage

@dataclass
class ExecutionResult:
    """Result of code execution in sandbox."""
    exit_code: int
    output: str
    error: str
    execution_time: float
    resource_usage: Dict

class ConsoleOutput:
    """Console output from code execution."""
    def __init__(self, text: str, exit_code: int = 0):
        self._text = text
        self._exit_code = exit_code

    @property
    def text(self) -> str:
        return self._text

    @property
    def exit_code(self) -> int:
        return self._exit_code

    def __str__(self):
        return f"ConsoleOutput(text={self.text}, exit_code={self.exit_code})"

class Session(ABC):
    """Abstract base class for sandbox sessions."""
    
    def __init__(
        self,
        lang: str,
        verbose: bool = True,
        resource_limits: Optional[ResourceLimits] = None,
        strict_security: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.lang = lang
        self.verbose = verbose
        self.resource_limits = resource_limits or ResourceLimits()
        self.strict_security = strict_security
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.security_scanner = SecurityScanner()
        self.resource_monitor: Optional[ResourceMonitor] = None
        self._container = None
        
    def _log(self, message: str, level: str = 'info'):
        """Log message if verbose is enabled."""
        if self.verbose:
            getattr(self.logger, level)(message)
            
    def _setup_monitoring(self):
        """Set up resource monitoring for the container."""
        if self._container:
            self.resource_monitor = ResourceMonitor(
                self._container,
                limits=self.resource_limits
            )
            
    @abstractmethod
    def open(self):
        """Open the sandbox session."""
        raise NotImplementedError
        
    @abstractmethod
    def close(self):
        """Close the sandbox session."""
        raise NotImplementedError
        
    def run(self, code: str, libraries: Optional[List] = None) -> ExecutionResult:
        """
        Run code in the sandbox with security checks and resource monitoring.
        
        Args:
            code: The code to execute
            libraries: Optional list of libraries to install
            
        Returns:
            ExecutionResult containing execution details
            
        Raises:
            SecurityError: If code fails security checks
            ResourceError: If resource limits are exceeded
            ContainerError: If container operations fail
        """
        try:
            # Security scan
            self._log("Performing security scan...")
            security_issues = self.security_scanner.scan_code(
                code,
                strict=self.strict_security
            )
            
            if security_issues:
                self._log(
                    f"Found {len(security_issues)} security issues",
                    level='warning'
                )
            
            # Start resource monitoring
            if self.resource_monitor:
                self._log("Starting resource monitoring...")
                self.resource_monitor.start()
            
            # Execute code
            self._log("Executing code...")
            result = self._execute_code(code, libraries)
            
            # Get resource usage summary
            resource_summary = {}
            if self.resource_monitor:
                resource_summary = self.resource_monitor.get_summary()
                
            return ExecutionResult(
                exit_code=result.exit_code,
                output=result.text,
                error="",  # Add error handling
                execution_time=resource_summary.get('duration_seconds', 0),
                resource_usage=resource_summary
            )
            
        except (SecurityError, ResourceError, ContainerError) as e:
            self._log(f"Error during execution: {str(e)}", level='error')
            raise
            
    @abstractmethod
    def copy_to_runtime(self, src: str, dest: str):
        """Copy file to sandbox runtime."""
        raise NotImplementedError
        
    @abstractmethod
    def copy_from_runtime(self, src: str, dest: str):
        """Copy file from sandbox runtime."""
        raise NotImplementedError
        
    @abstractmethod
    def execute_command(self, command: str) -> ConsoleOutput:
        """Execute command in sandbox."""
        raise NotImplementedError
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
