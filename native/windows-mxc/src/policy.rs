//! Host-independent validation and translation of the narrow launcher protocol.

use std::collections::{BTreeMap, HashSet};
use std::net::{Ipv4Addr, Ipv6Addr};

use serde::{Deserialize, Serialize};
use wxc_common::cmdline::{CommandLineContext, cmdline_from_argv_for_context};
use wxc_common::models::{
    BaseProcessUiConfig, ContainerPolicy, ExecutionRequest, FallbackPolicy, NetworkAction,
    NetworkCidr, NetworkEgressPolicy, NetworkIngressPolicy, NetworkPeer, NetworkPolicy,
    NetworkRule, UiPolicy,
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunRequest {
    pub protocol: u32,
    pub argv: Vec<String>,
    pub cwd: String,
    pub read_only_paths: Vec<String>,
    pub writable_paths: Vec<String>,
    pub deny_paths: Vec<String>,
    pub network: Network,
    pub env: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Network {
    None,
    Host,
}

/// Accept only ordinary, absolute DOS paths. Device namespaces, UNC, ADS and
/// Win32 name aliases must never turn a textual policy into another object.
pub fn validate_path(path: &str) -> Result<(), String> {
    let bytes = path.as_bytes();
    if bytes.len() < 3
        || !bytes[0].is_ascii_alphabetic()
        || bytes[1] != b':'
        || bytes[2] != b'\\'
        || path.contains(['\0', '/', '"', '<', '>', '|', '?', '*'])
        || path[2..].contains(':')
        || path.chars().any(char::is_control)
    {
        return Err(
            "paths must be absolute DOS paths without device, UNC or ADS syntax".to_owned(),
        );
    }
    for component in path[3..].split('\\') {
        if component.is_empty() && path.len() == 3 {
            continue;
        }
        let basename = component
            .split('.')
            .next()
            .unwrap_or("")
            .trim_end_matches(' ')
            .to_ascii_uppercase();
        let reserved = matches!(
            basename.as_str(),
            "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$" | "CLOCK$"
        ) || ["COM", "LPT"].iter().any(|prefix| {
            basename.strip_prefix(prefix).is_some_and(|suffix| {
                matches!(
                    suffix,
                    "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
                )
            })
        });
        if component.is_empty() || component.ends_with([' ', '.']) || reserved {
            return Err(
                "paths must not contain ambiguous or reserved Windows components".to_owned(),
            );
        }
    }
    Ok(())
}

pub fn within(path: &str, root: &str) -> bool {
    let path = path.to_lowercase();
    let root = root.to_lowercase();
    path == root
        || path
            .strip_prefix(&root)
            .is_some_and(|tail| tail.starts_with('\\'))
}

pub fn prepare(request: RunRequest) -> Result<ExecutionRequest, String> {
    if request.protocol != crate::PROTOCOL {
        return Err("unsupported launcher protocol".to_owned());
    }
    if request.argv.is_empty() || request.argv.iter().any(|arg| arg.contains('\0')) {
        return Err("argv must contain an executable and no NUL characters".to_owned());
    }
    validate_path(&request.argv[0])?;
    if !request.argv[0].to_ascii_lowercase().ends_with(".exe") {
        return Err("argv[0] must be an absolute .exe path".to_owned());
    }
    validate_path(&request.cwd)?;
    for path in request
        .read_only_paths
        .iter()
        .chain(&request.writable_paths)
        .chain(&request.deny_paths)
    {
        validate_path(path)?;
    }
    if request.writable_paths.len() != 1
        || request.writable_paths[0].len() <= 3
        || !within(&request.cwd, &request.writable_paths[0])
    {
        return Err("exactly one private writable directory containing cwd is required".to_owned());
    }
    // An explicit, nonempty environment prevents the SDK's default-environment
    // branch. Nothing from the helper process is implicitly granted to the child.
    if request.env.is_empty() {
        return Err("an explicit nonempty child environment is required".to_owned());
    }
    let mut names = HashSet::new();
    for (name, value) in &request.env {
        if name.is_empty()
            || name.contains(['=', '\0'])
            || value.contains('\0')
            || !names.insert(name.to_uppercase())
        {
            return Err(
                "environment names must be unique without case, nonempty and free of NUL or '='"
                    .to_owned(),
            );
        }
    }
    let command =
        cmdline_from_argv_for_context(&request.argv, CommandLineContext::WindowsCreateProcess)
            .map_err(|error| error.to_string())?;
    if command.encode_utf16().count() >= 32767 {
        return Err("Windows command line exceeds 32766 UTF-16 code units".to_owned());
    }
    let enabled = request.network == Network::Host;
    let action = if enabled {
        NetworkAction::Allow
    } else {
        NetworkAction::Deny
    };
    // An explicit deny rule requires PSEC in the pinned SDK. Default-deny
    // alone does not: BaseContainerRunner could otherwise select transitional
    // SBOX if its native capability resolution ever became unavailable. Host
    // mode independently requires PSEC through unrestricted host loopback.
    let offline_denies = if enabled {
        Vec::new()
    } else {
        vec![NetworkRule {
            to: vec![
                NetworkPeer {
                    cidr: NetworkCidr {
                        address: Ipv4Addr::UNSPECIFIED.into(),
                        prefix_length: 0,
                    },
                    except: Vec::new(),
                },
                NetworkPeer {
                    cidr: NetworkCidr {
                        address: Ipv6Addr::UNSPECIFIED.into(),
                        prefix_length: 0,
                    },
                    except: Vec::new(),
                },
            ],
            ports: Vec::new(),
        }]
    };
    Ok(ExecutionRequest {
        script_code: command,
        working_directory: request.cwd,
        env: request
            .env
            .into_iter()
            .map(|(name, value)| format!("{name}={value}"))
            .collect(),
        policy: ContainerPolicy {
            capabilities: vec!["registryRead".to_owned()],
            readwrite_paths: request.writable_paths,
            readonly_paths: request.read_only_paths,
            denied_paths: request.deny_paths,
            fallback: FallbackPolicy {
                allow_dacl_mutation: false,
            },
            default_network_policy: if enabled {
                NetworkPolicy::Allow
            } else {
                NetworkPolicy::Block
            },
            allow_local_network: enabled,
            network_egress: Some(NetworkEgressPolicy {
                default: action,
                deny: offline_denies,
                ..Default::default()
            }),
            network_ingress: Some(NetworkIngressPolicy {
                default: action,
                host_loopback: action,
            }),
            network_specified: true,
            network_mode_specified: true,
            // Console tools may need Win32k during startup. Clipboard, input
            // injection and system-control rights stay denied by the defaults.
            ui: UiPolicy {
                disable: false,
                ..Default::default()
            },
            base_process_ui: BaseProcessUiConfig {
                isolation: "desktop".to_owned(),
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn request() -> RunRequest {
        RunRequest {
            protocol: 1,
            argv: vec![
                r"C:\Python\python.exe".to_owned(),
                "-c".to_owned(),
                "print('ok')".to_owned(),
            ],
            cwd: r"C:\private\workspace".to_owned(),
            read_only_paths: vec![r"C:\Python".to_owned(), r"C:\Windows".to_owned()],
            writable_paths: vec![r"C:\private\workspace".to_owned()],
            deny_paths: Vec::new(),
            network: Network::None,
            env: BTreeMap::from([("TEMP".to_owned(), r"C:\private\workspace".to_owned())]),
        }
    }

    #[test]
    fn offline_policy_blocks_all_network_directions_and_fallbacks() {
        let built = prepare(request()).unwrap();
        assert_eq!(
            built.policy.network_egress.unwrap().default,
            NetworkAction::Deny
        );
        let ingress = built.policy.network_ingress.unwrap();
        assert_eq!(ingress.default, NetworkAction::Deny);
        assert_eq!(ingress.host_loopback, NetworkAction::Deny);
        assert!(!built.policy.allow_local_network);
        assert!(!built.policy.fallback.allow_dacl_mutation);
        assert!(!built.policy.least_privilege_mode);
        assert!(!built.policy.network_proxy.is_enabled());
        assert!(built.policy.capture_denials.is_none());
        assert_eq!(built.policy.capabilities, vec!["registryRead"]);
        assert!(!built.policy.ui.injection);
    }

    #[test]
    fn every_policy_requires_psec_networking_even_if_a_capability_probe_changes() {
        for enabled in [false, true] {
            for denied_paths in [Vec::new(), vec![r"C:\private\secret".to_owned()]] {
                let mut input = request();
                input.network = if enabled {
                    Network::Host
                } else {
                    Network::None
                };
                input.deny_paths = denied_paths;
                let built = prepare(input).unwrap();
                let egress = built.policy.network_egress.unwrap();
                let ingress = built.policy.network_ingress.unwrap();
                if enabled {
                    assert_eq!(ingress.host_loopback, NetworkAction::Allow);
                } else {
                    assert_eq!(egress.default, NetworkAction::Deny);
                    assert!(egress.allow.is_empty());
                    assert_eq!(egress.deny.len(), 1);
                    let all_addresses = &egress.deny[0];
                    assert!(all_addresses.ports.is_empty());
                    assert_eq!(all_addresses.to.len(), 2);
                    assert_eq!(all_addresses.to[0].cidr.address, Ipv4Addr::UNSPECIFIED);
                    assert_eq!(all_addresses.to[1].cidr.address, Ipv6Addr::UNSPECIFIED);
                    assert!(
                        all_addresses
                            .to
                            .iter()
                            .all(|peer| peer.cidr.prefix_length == 0 && peer.except.is_empty())
                    );
                }
            }
        }
    }

    #[test]
    fn explicit_host_network_has_matching_ingress_and_egress() {
        let mut input = request();
        input.network = Network::Host;
        let built = prepare(input).unwrap();
        assert_eq!(
            built.policy.network_egress.unwrap().default,
            NetworkAction::Allow
        );
        let ingress = built.policy.network_ingress.unwrap();
        assert_eq!(ingress.default, NetworkAction::Allow);
        assert_eq!(ingress.host_loopback, NetworkAction::Allow);
    }

    #[test]
    fn rejects_path_aliases_and_namespace_escapes() {
        for path in [
            r"relative",
            r"C:relative",
            r"\Windows",
            r"\\server\share",
            r"\\?\C:\safe",
            r"\\.\NUL",
            r"C:\safe:stream",
            r"C:\safe\..\escape",
            r"C:\safe\.",
            r"C:\safe\NUL.txt",
            r"C:\safe\file.",
            r"C:\safe\file ",
            r"C:\safe\\file",
            r"C:/safe",
            r"C:\safe\COM¹",
            r"C:\safe\conin$",
            r"C:\safe\ConOut$.txt",
            r"C:\safe\CLOCK$",
            r"C:\safe\CON .txt",
            r"C:\safe\LPT1 .log",
        ] {
            assert!(validate_path(path).is_err(), "accepted {path:?}");
        }
        assert!(validate_path(r"C:\Program Files\Python").is_ok());
    }

    #[test]
    fn restricts_writes_and_cwd_to_private_workspace() {
        let mut input = request();
        input.cwd = r"c:\PRIVATE\workspace\nested".to_owned();
        assert!(prepare(input).is_ok());
        let mut input = request();
        input.cwd = r"C:\private\workspace-other".to_owned();
        assert!(prepare(input).is_err());
        let mut input = request();
        input.writable_paths.push(r"C:\extra".to_owned());
        assert!(prepare(input).is_err());
        let mut input = request();
        input.writable_paths = vec![r"C:\".to_owned()];
        assert!(prepare(input).is_err());
    }

    #[test]
    fn environment_does_not_fall_back_to_helper_secrets() {
        let mut input = request();
        input.env.clear();
        assert!(prepare(input).is_err());
        let mut input = request();
        input.env.insert("temp".to_owned(), "different".to_owned());
        assert!(prepare(input).is_err());
        let mut input = request();
        input.env.insert("=C:".to_owned(), "value".to_owned());
        assert!(prepare(input).is_err());
    }

    #[test]
    fn arguments_preserve_quotes_empty_values_and_trailing_slashes() {
        let mut input = request();
        input.argv = vec![
            r"C:\Program Files\Python\python.exe".to_owned(),
            String::new(),
            "a\"b".to_owned(),
            "ends with \\".to_owned(),
            "& whoami".to_owned(),
        ];
        let built = prepare(input).unwrap();
        assert_eq!(
            built.script_code,
            "\"C:\\Program Files\\Python\\python.exe\" \"\" \"a\\\"b\" \"ends with \\\\\" \"& whoami\""
        );
    }
}
