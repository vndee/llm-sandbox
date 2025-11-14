# ruff: noqa: E501
"""Hyperlight backend implementation for LLM Sandbox.

This module provides a Hyperlight-based sandbox backend that uses micro VMs
instead of containers for code execution. Hyperlight provides fast, lightweight
virtualization using KVM (Linux) or Windows Hypervisor Platform (Windows).

Note: This is an experimental backend with the following requirements:
    - Rust toolchain installed (for compiling guest binaries)
    - KVM/mshv support on Linux or WHP on Windows
    - Hyperlight library compiled and available
    - Guest binary templates for each supported language

The Hyperlight backend operates differently from container backends:
    - Code is compiled into guest binaries rather than executed in containers
    - Each execution creates a new micro VM instance
    - No persistent environment (stateless execution)
    - Significantly faster startup time compared to containers

Architecture:
    1. User code is wrapped in a guest template
    2. Template is compiled to x86_64-unknown-none target
    3. Hyperlight creates micro VM and loads guest binary
    4. Code executes in VM and results are returned via host functions
    5. VM is destroyed after execution

Limitations:
    - Library installation requires rebuilding guest binary
    - No persistent state between runs
    - Limited language support (requires guest templates)
    - Compilation overhead on first run (can be cached)
"""

import json
import logging
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.core.config import SessionConfig
from llm_sandbox.core.session_base import BaseSession
from llm_sandbox.data import ConsoleOutput
from llm_sandbox.exceptions import (
    CommandEmptyError,
    ContainerError,
    MissingDependencyError,
    NotOpenSessionError,
)
from llm_sandbox.security import SecurityPolicy

HYPERLIGHT_GUEST_TEMPLATE = """#![no_std]
#![no_main]
extern crate alloc;

use alloc::string::ToString;
use alloc::vec::Vec;
use hyperlight_common::flatbuffer_wrappers::function_call::FunctionCall;
use hyperlight_common::flatbuffer_wrappers::function_types::{{
    ParameterType, ParameterValue, ReturnType,
}};
use hyperlight_common::flatbuffer_wrappers::guest_error::ErrorCode;
use hyperlight_common::flatbuffer_wrappers::util::get_flatbuffer_result;
use hyperlight_guest::error::{{HyperlightGuestError, Result}};
use hyperlight_guest_bin::guest_function::definition::GuestFunctionDefinition;
use hyperlight_guest_bin::guest_function::register::register_function;
use hyperlight_guest_bin::host_comm::call_host_function;

fn execute_code(function_call: &FunctionCall) -> Result<Vec<u8>> {{
    if let ParameterValue::String(code) = function_call.parameters.clone().unwrap()[0].clone() {{
        // Execute the user code here
        // This is a placeholder - actual implementation would execute the code
        let output = "Code execution not yet implemented".to_string();
        
        let result = call_host_function::<i32>(
            "HostPrint",
            Some(Vec::from(&[ParameterValue::String(output)])),
            ReturnType::Int,
        )?;
        Ok(get_flatbuffer_result(result))
    }} else {{
        Err(HyperlightGuestError::new(
            ErrorCode::GuestFunctionParameterTypeMismatch,
            "Invalid parameters".to_string(),
        ))
    }}
}}

#[no_mangle]
pub extern "C" fn hyperlight_main() {{
    let execute_def = GuestFunctionDefinition::new(
        "ExecuteCode".to_string(),
        Vec::from(&[ParameterType::String]),
        ReturnType::Int,
        execute_code as usize,
    );
    register_function(execute_def);
}}

#[no_mangle]
pub fn guest_dispatch_function(function_call: FunctionCall) -> Result<Vec<u8>> {{
    Err(HyperlightGuestError::new(
        ErrorCode::GuestFunctionNotFound,
        function_call.function_name.clone(),
    ))
}}
"""


class HyperlightContainerAPI:
    """Hyperlight implementation of the ContainerAPI protocol.

    This adapter provides a container-like API for Hyperlight micro VMs,
    making it compatible with the BaseSession interface.
    """

    def __init__(self, workdir: str = "/sandbox", verbose: bool = False) -> None:
        """Initialize Hyperlight container API.

        Args:
            workdir: Working directory (not used in Hyperlight, for API compatibility)
            verbose: Enable verbose logging
        """
        self.workdir = workdir
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        self._temp_dirs: list[Path] = []

    def create_container(self, config: Any) -> Any:
        """Create Hyperlight sandbox (prepare guest binary).

        Args:
            config: Configuration dict with guest_binary_path

        Returns:
            Guest binary path
        """
        if "guest_binary_path" not in config:
            msg = "Hyperlight requires guest_binary_path in config"
            raise ContainerError(msg)

        guest_binary = Path(config["guest_binary_path"])
        if not guest_binary.exists():
            msg = f"Guest binary not found: {guest_binary}"
            raise ContainerError(msg)

        return str(guest_binary)

    def start_container(self, container: Any) -> None:
        """Start container (no-op for Hyperlight as VM is created on demand)."""

    def stop_container(self, container: Any) -> None:
        """Stop container (no-op for Hyperlight as VM is destroyed after each call)."""

    def execute_command(self, container: Any, command: str, **kwargs: Any) -> tuple[int, Any]:
        """Execute command via Hyperlight VM.

        Args:
            container: Guest binary path
            command: Command to execute (passed as code to guest)
            **kwargs: Additional arguments (workdir, timeout, etc.)

        Returns:
            Tuple of (exit_code, output)
        """
        if not command:
            msg = "Command cannot be empty"
            raise CommandEmptyError(msg)

        guest_binary = container
        timeout = kwargs.get("timeout", 60)

        # For now, we'll use a simple approach: create a minimal host program
        # that calls the guest function with the command
        try:
            # Create temporary directory for host program
            temp_dir = Path(tempfile.mkdtemp(prefix="hyperlight_host_"))
            self._temp_dirs.append(temp_dir)

            # Create a simple Rust host program
            host_code = f"""
use hyperlight_host::{{MultiUseSandbox, UninitializedSandbox, GuestBinary}};

fn main() -> Result<(), Box<dyn std::error::Error>> {{
    let mut sandbox = UninitializedSandbox::new(
        GuestBinary::FilePath("{guest_binary}".to_string()),
        None
    )?;
    
    let mut multi_use_sandbox: MultiUseSandbox = sandbox.evolve()?;
    
    // Call guest function to execute code
    let result = multi_use_sandbox.call::<i32>(
        "ExecuteCode",
        r#"{command}"#.to_string(),
    )?;
    
    println!("{{:?}}", result);
    Ok(())
}}
"""

            host_program = temp_dir / "main.rs"
            host_program.write_text(host_code)

            # Create Cargo.toml for host
            cargo_toml = temp_dir / "Cargo.toml"
            cargo_toml.write_text(
                """[package]
name = "hyperlight-host-runner"
version = "0.1.0"
edition = "2021"

[dependencies]
hyperlight-host = { git = "https://github.com/hyperlight-dev/hyperlight" }
"""
            )

            # Build and run the host program
            result = subprocess.run(
                ["cargo", "run", "--quiet"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )

            return result.returncode, result.stdout.encode() if result.stdout else b""

        except subprocess.TimeoutExpired:
            return 124, b"Command timed out"
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Hyperlight execution error: {e}")
            return 1, str(e).encode()

    def copy_to_container(self, container: Any, src: str, dest: str, **_kwargs: Any) -> None:
        """Copy file to container (not supported in Hyperlight)."""
        msg = "File copy to Hyperlight VM not supported (stateless execution)"
        raise NotImplementedError(msg)

    def copy_from_container(self, container: Any, src: str, **_kwargs: Any) -> tuple[bytes, dict]:
        """Copy file from container (not supported in Hyperlight)."""
        msg = "File copy from Hyperlight VM not supported (stateless execution)"
        raise NotImplementedError(msg)

    def cleanup(self) -> None:
        """Clean up temporary directories."""
        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)


class SandboxHyperlightSession(BaseSession):
    """Sandbox session implemented using Hyperlight micro VMs.

    This class provides a sandboxed environment for code execution by leveraging
    Hyperlight's micro VM technology. Unlike container-based backends, Hyperlight
    creates lightweight VMs that boot instantly without a full OS.

    Key differences from container backends:
        - Stateless execution (no persistent environment)
        - Faster startup time (microseconds vs milliseconds)
        - Requires code compilation into guest binaries
        - Limited to languages with guest templates

    Note: This is an experimental backend. Many features available in container
    backends are not yet implemented or supported.
    """

    def __init__(
        self,
        image: str | None = None,
        lang: str = SupportedLanguage.PYTHON,
        keep_template: bool = False,
        verbose: bool = False,
        workdir: str = "/sandbox",
        security_policy: SecurityPolicy | None = None,
        default_timeout: float | None = None,
        execution_timeout: float | None = None,
        session_timeout: float | None = None,
        skip_environment_setup: bool = False,
        guest_binary_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Hyperlight session.

        Args:
            image: Not used (for API compatibility with other backends)
            lang: Programming language (currently only Python supported)
            keep_template: Whether to keep compiled guest binary
            verbose: Enable verbose logging
            workdir: Working directory (for API compatibility)
            security_policy: Security policy to enforce
            default_timeout: Default timeout for operations
            execution_timeout: Timeout for code execution
            session_timeout: Maximum session lifetime
            skip_environment_setup: Skip environment setup
            guest_binary_path: Path to pre-compiled guest binary
            **kwargs: Additional arguments
        """
        config = SessionConfig(
            image=image,
            dockerfile=None,
            lang=lang,
            keep_template=keep_template,
            verbose=verbose,
            workdir=workdir,
            security_policy=security_policy,
            default_timeout=default_timeout,
            execution_timeout=execution_timeout,
            session_timeout=session_timeout,
            skip_environment_setup=skip_environment_setup,
            container_id=None,
        )

        super().__init__(config, **kwargs)

        self.guest_binary_path = guest_binary_path
        self.container_api = HyperlightContainerAPI(workdir=workdir, verbose=verbose)
        self._temp_guest_dir: Path | None = None
        self._guest_binary: Path | None = None

    def _check_hyperlight_dependencies(self) -> None:
        """Check if Hyperlight dependencies are available."""
        # Check for Rust toolchain
        if not shutil.which("cargo"):
            msg = (
                "Hyperlight backend requires Rust toolchain. "
                "Install from https://www.rust-lang.org/tools/install"
            )
            raise MissingDependencyError(msg)

        # Check for hyperlight (try to import or check for binary)
        # For now, we'll check if we can access the hyperlight repo
        if not self.guest_binary_path:
            self._log(
                "No pre-compiled guest binary provided. "
                "Will attempt to compile guest on first run.",
                "warning",
            )

    def _compile_guest_binary(self) -> Path:
        """Compile guest binary for the session.

        Returns:
            Path to compiled guest binary

        Raises:
            ContainerError: If compilation fails
        """
        if self._guest_binary and self._guest_binary.exists():
            return self._guest_binary

        # Create temporary directory for guest source
        self._temp_guest_dir = Path(tempfile.mkdtemp(prefix="hyperlight_guest_"))

        # Create guest source file
        guest_src = self._temp_guest_dir / "src" / "main.rs"
        guest_src.parent.mkdir(parents=True)
        guest_src.write_text(HYPERLIGHT_GUEST_TEMPLATE)

        # Create Cargo.toml for guest
        cargo_toml = self._temp_guest_dir / "Cargo.toml"
        cargo_toml.write_text(
            """[package]
name = "llm-sandbox-hyperlight-guest"
version = "0.1.0"
edition = "2021"

[dependencies]
hyperlight-guest = { git = "https://github.com/hyperlight-dev/hyperlight" }
hyperlight-guest-bin = { git = "https://github.com/hyperlight-dev/hyperlight" }
hyperlight-common = { git = "https://github.com/hyperlight-dev/hyperlight" }

[profile.release]
panic = "abort"

[profile.dev]
panic = "abort"
"""
        )

        # Create .cargo/config.toml with build settings
        cargo_config_dir = self._temp_guest_dir / ".cargo"
        cargo_config_dir.mkdir()
        cargo_config = cargo_config_dir / "config.toml"
        cargo_config.write_text(
            """[build]
target = "x86_64-unknown-none"

[target.x86_64-unknown-none]
rustflags = [
  "-C",
  "code-model=small",
  "-C",
  "link-args=-e entrypoint",
]
linker = "rust-lld"
"""
        )

        # Add target if not already installed
        self._log("Adding x86_64-unknown-none target...", "info")
        subprocess.run(
            ["rustup", "target", "add", "x86_64-unknown-none"],
            check=False,
            capture_output=True,
        )

        # Build the guest binary
        self._log("Compiling Hyperlight guest binary (this may take a while)...", "info")
        try:
            result = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=self._temp_guest_dir,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes for compilation
                check=True,
            )

            if self.verbose:
                self._log(f"Build output: {result.stdout}", "debug")

            # Find the compiled binary
            target_dir = self._temp_guest_dir / "target" / "x86_64-unknown-none" / "release"
            binary_name = "llm-sandbox-hyperlight-guest"
            self._guest_binary = target_dir / binary_name

            if not self._guest_binary.exists():
                msg = f"Guest binary not found at {self._guest_binary}"
                raise ContainerError(msg)

            self._log(f"Guest binary compiled successfully: {self._guest_binary}", "info")
            return self._guest_binary

        except subprocess.CalledProcessError as e:
            msg = f"Failed to compile guest binary: {e.stderr}"
            raise ContainerError(msg) from e
        except subprocess.TimeoutExpired as e:
            msg = "Guest binary compilation timed out after 5 minutes"
            raise ContainerError(msg) from e

    def open(self) -> None:
        """Open the Hyperlight session.

        This prepares the guest binary for execution.
        """
        if self.is_open:
            return

        self._check_hyperlight_dependencies()

        # Compile or load guest binary
        if self.guest_binary_path:
            self._guest_binary = Path(self.guest_binary_path)
            if not self._guest_binary.exists():
                msg = f"Guest binary not found: {self.guest_binary_path}"
                raise ContainerError(msg)
        else:
            self._guest_binary = self._compile_guest_binary()

        # Create container (prepare for execution)
        self.container = self.container_api.create_container(
            {"guest_binary_path": str(self._guest_binary)}
        )

        self.is_open = True
        self._start_session_timer()
        self._log("Hyperlight session opened", "info")

    def close(self) -> None:
        """Close the Hyperlight session and clean up resources."""
        if not self.is_open:
            return

        self._stop_session_timer()

        # Clean up temporary directories
        if not self.config.keep_template:
            if self._temp_guest_dir and self._temp_guest_dir.exists():
                shutil.rmtree(self._temp_guest_dir, ignore_errors=True)
                self._log("Cleaned up guest binary directory", "debug")

        self.container_api.cleanup()
        self.is_open = False
        self._log("Hyperlight session closed", "info")

    def environment_setup(self) -> None:
        """Set up environment (no-op for Hyperlight).

        Hyperlight VMs are stateless and don't require environment setup.
        """
        if self.config.skip_environment_setup:
            return

        # Nothing to do for Hyperlight - stateless execution
        self._log("Environment setup skipped (Hyperlight is stateless)", "debug")

    def _handle_timeout(self) -> None:
        """Handle timeout cleanup for Hyperlight backend.

        Hyperlight VMs are automatically destroyed after execution,
        so minimal cleanup is needed on timeout.
        """
        if self.verbose:
            self._log("Hyperlight VM execution timed out", "warning")

        # Clean up temporary resources
        if not self.config.keep_template:
            if self._temp_guest_dir and self._temp_guest_dir.exists():
                shutil.rmtree(self._temp_guest_dir, ignore_errors=True)

    def _connect_to_existing_container(self, container_id: str) -> None:
        """Connect to existing Hyperlight container (not supported).

        Args:
            container_id: Container ID (not used for Hyperlight)

        Raises:
            NotImplementedError: Hyperlight does not support connecting to existing VMs
        """
        msg = (
            "Hyperlight backend does not support connecting to existing containers. "
            "Each session creates a new micro VM that is destroyed after execution."
        )
        raise NotImplementedError(msg)

    def install(self, libraries: list[str], **_kwargs: Any) -> ConsoleOutput:
        """Install libraries (not supported in Hyperlight).

        Args:
            libraries: List of libraries to install
            **_kwargs: Additional arguments

        Returns:
            ConsoleOutput with error message

        Note:
            Library installation in Hyperlight requires recompiling the guest
            binary with the libraries included. This is not currently supported.
        """
        msg = (
            "Library installation not supported in Hyperlight backend. "
            "Libraries must be included in the guest binary at compile time."
        )
        self._log(msg, "warning")
        return ConsoleOutput(exit_code=1, stderr=msg, stdout="")

    def __enter__(self) -> "SandboxHyperlightSession":
        """Enter context manager."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.close()
