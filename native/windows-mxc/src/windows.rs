//! The only execution path is MXC's native PSEC runner; no dispatcher/fallback.

use std::io::{self, Read};
use std::os::windows::fs::MetadataExt;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use appcontainer_common::base_container_runner::BaseContainerRunner;
use learning_mode_windows::SecurityEnvironmentApi;
use wxc_common::filesystem_object::normalize_object_conflicts;
use wxc_common::logger::{Logger, Mode};
use wxc_common::models::ExecutionRequest;
use wxc_common::sandbox_process::{SandboxBackend, StdioMode};

use crate::{PROTOCOL, Probe};

// MXC's pinned ETW dependency uses the documented Advapi32 exports. Version
// 1.2.3 leaves their import library to the application on Windows GNU as well.
#[link(name = "advapi32")]
unsafe extern "system" {}

pub fn probe() -> Probe {
    if !BaseContainerRunner::is_process_security_environment_usable() {
        return Probe::unavailable("native PSEC create/close probe failed on this Windows host");
    }
    match SecurityEnvironmentApi::load().and_then(|api| api.supports_deny_paths()) {
        Ok(deny_paths) => Probe {
            protocol: PROTOCOL,
            available: true,
            deny_paths,
            reason: None,
        },
        Err(_) => Probe::unavailable("could not query native PSEC policy support"),
    }
}

fn validate_workspace(request: &ExecutionRequest) -> Result<(), String> {
    // Every ancestor must be an ordinary directory. In particular, never
    // grant writes through a junction or symbolic link supplied by a prior job.
    for directory in [
        &request.working_directory,
        &request.policy.readwrite_paths[0],
    ] {
        for ancestor in Path::new(directory).ancestors() {
            let metadata = std::fs::symlink_metadata(ancestor)
                .map_err(|_| "workspace and cwd must already exist as directories".to_owned())?;
            if !metadata.is_dir() || metadata.file_attributes() & 0x400 != 0 {
                return Err("workspace and cwd must not traverse Windows reparse points".to_owned());
            }
        }
    }
    Ok(())
}

fn start_forwarder(
    mut stream: Box<dyn Read + Send>,
    stderr: bool,
    errors: mpsc::Sender<()>,
) -> Result<thread::JoinHandle<io::Result<()>>, String> {
    thread::Builder::new()
        .name(if stderr { "mxc-stderr" } else { "mxc-stdout" }.to_owned())
        .spawn(move || {
            let result = if stderr {
                crate::forwarding::copy_and_flush(&mut stream, &mut io::stderr())
            } else {
                crate::forwarding::copy_and_flush(&mut stream, &mut io::stdout())
            };
            if result.is_err() {
                let _ = errors.send(());
            }
            result
        })
        .map_err(|_| "could not start child output forwarding".to_owned())
}

pub fn run(mut request: ExecutionRequest) -> Result<i32, String> {
    let capabilities = probe();
    if !capabilities.available {
        return Err(capabilities
            .reason
            .unwrap_or_else(|| "PSEC is unavailable".to_owned()));
    }
    if !request.policy.denied_paths.is_empty() && !capabilities.deny_paths {
        return Err("this Windows host cannot enforce native PSEC deny paths".to_owned());
    }
    validate_workspace(&request)?;
    let mut logger = Logger::new(Mode::Buffer);
    if let Some(normalized) = normalize_object_conflicts(&request.policy, &mut logger)? {
        request.policy = normalized;
    }
    // These invariants plus the PSEC-only directional network policy ensure
    // the pinned SDK rejects execution if PSEC cannot be selected, even if a
    // future capability query disagrees with the initial successful probe.
    if request.policy.least_privilege_mode
        || request.policy.network_proxy.is_enabled()
        || request.policy.capture_denials.is_some()
        || request.policy.fallback.allow_dacl_mutation
        || request
            .policy
            .capabilities
            .iter()
            .any(|name| name != "registryRead")
    {
        return Err("request would require an unsupported MXC execution path".to_owned());
    }
    let mut child = BaseContainerRunner::new()
        .spawn(&request, &mut logger, StdioMode::Pipes)
        .map_err(|error| format!("native PSEC launch failed: {}", error.error_message))?;
    // The SDK attaches a non-inheritable kill-on-close Job Object before
    // resuming the suspended process. Killing this helper closes that job too.
    // No JSON protocol bytes or parent stdin handle reach the workload.
    drop(child.take_stdin());
    let stdout = child
        .take_stdout()
        .ok_or("PSEC did not provide a stdout pipe")?;
    let stderr = child
        .take_stderr()
        .ok_or("PSEC did not provide a stderr pipe")?;
    let (errors, receiver) = mpsc::channel();
    let stdout_thread = start_forwarder(stdout, false, errors.clone())?;
    let stderr_thread = start_forwarder(stderr, true, errors.clone())?;
    // Keep one sender alive so successful EOF cannot turn the wait loop into
    // a busy loop; foreground process exit, not output EOF, owns termination.
    let mut output_failed = false;
    loop {
        if child
            .try_wait()
            .map_err(|_| "could not poll PSEC process")?
            .is_some()
        {
            break;
        }
        if receiver.recv_timeout(Duration::from_millis(20)).is_ok() {
            output_failed = true;
            child
                .kill()
                .map_err(|_| "could not terminate PSEC process tree")?;
            break;
        }
    }
    // wait() terminates/reaps remaining descendants before tearing down PSEC.
    // That also closes their pipe writers before either forwarding thread joins.
    let status = child
        .wait()
        .map_err(|_| "could not reap PSEC process tree")?;
    let stdout_result = stdout_thread
        .join()
        .map_err(|_| "stdout forwarding failed")?;
    let stderr_result = stderr_thread
        .join()
        .map_err(|_| "stderr forwarding failed")?;
    if output_failed || stdout_result.is_err() || stderr_result.is_err() {
        return Err("could not forward child output".to_owned());
    }
    Ok(status)
}
