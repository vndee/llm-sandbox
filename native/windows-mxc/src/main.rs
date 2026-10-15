//! One bounded JSON request becomes one native PSEC process tree.

#[cfg(any(windows, test))]
mod forwarding;
mod policy;
#[cfg(windows)]
mod windows;

use std::io::{self, Read};

use serde::Serialize;

const PROTOCOL: u32 = 1;
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const HELPER_ERROR_EXIT: i32 = 125;

#[derive(Debug, Serialize)]
struct Probe {
    protocol: u32,
    available: bool,
    deny_paths: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

impl Probe {
    fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            protocol: PROTOCOL,
            available: false,
            deny_paths: false,
            reason: Some(reason.into()),
        }
    }
}

fn parse_request(reader: impl Read) -> Result<policy::RunRequest, String> {
    let mut bytes = Vec::new();
    reader
        .take((MAX_REQUEST_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| "could not read the launch request".to_owned())?;
    if bytes.len() > MAX_REQUEST_BYTES {
        return Err("launch request exceeds 1 MiB".to_owned());
    }
    serde_json::from_slice(&bytes).map_err(|_| "invalid launch request JSON or fields".to_owned())
}

fn main() {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    let outcome = match arguments.as_slice() {
        [flag] if flag == "--probe" => {
            #[cfg(windows)]
            let probe = windows::probe();
            #[cfg(not(windows))]
            let probe = Probe::unavailable("native MXC/PSEC requires Windows");
            println!(
                "{}",
                serde_json::to_string(&probe).expect("serializable probe")
            );
            Ok(0)
        }
        [flag] if flag == "--run" => parse_request(io::stdin().lock()).and_then(|request| {
            let request = policy::prepare(request)?;
            #[cfg(windows)]
            return windows::run(request);
            #[cfg(not(windows))]
            {
                drop(request);
                Err("native MXC/PSEC requires Windows".to_owned())
            }
        }),
        _ => Err("usage: llm-sandbox-mxc --probe | --run".to_owned()),
    };
    match outcome {
        Ok(code) => std::process::exit(code),
        Err(message) => {
            eprintln!("llm-sandbox-mxc: {message}");
            std::process::exit(HELPER_ERROR_EXIT);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_reader_is_bounded_before_json_parsing() {
        let bytes = vec![b' '; MAX_REQUEST_BYTES + 2];
        assert_eq!(
            parse_request(&bytes[..]).unwrap_err(),
            "launch request exceeds 1 MiB"
        );
    }

    #[test]
    fn rejects_unknown_fields_and_trailing_json() {
        let mut value = serde_json::to_value(policy::tests::request()).unwrap();
        value["allow_dacl_mutation"] = serde_json::json!(true);
        assert!(parse_request(serde_json::to_vec(&value).unwrap().as_slice()).is_err());
        let bytes = format!(
            "{} {{}}",
            serde_json::to_string(&policy::tests::request()).unwrap()
        );
        assert!(parse_request(bytes.as_bytes()).is_err());
    }
}
