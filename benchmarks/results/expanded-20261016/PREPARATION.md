# Expanded-run preparation

The expanded experiment retains `main` at `d79a0e2038ba90f267a3f5109beea9810dde292a` and the existing Docker image. No library code or host security policy was changed. The prior Podman/Kubernetes failures still apply; no second supported host is attached. Micromamba remains excluded.

## Matched Python build

The original experiment used different builds of Python 3.13.5. This experiment extracts `/usr/local` from the same cached official Docker image for native backends, preserving its CPython binary, standard library, libpython and bundled ensurepip. Native operating-system shared libraries still come from the host. Binary hashes and the actual image ID/digest are recorded in `runtime-identity.json`.

Preparation from the repository root (requires the existing Docker SDK and local daemon):

```python
import subprocess
from pathlib import Path
import docker

root = Path('/workspace/scratch/benchmark-python-3135')
root.mkdir(parents=True, exist_ok=True)
client = docker.DockerClient(base_url='unix:///var/run/docker.sock')
container = client.containers.create('python:3.13.5-slim-bookworm', command=['python', '--version'])
try:
    stream, _ = container.get_archive('/usr/local')
    archive = root / 'usr-local.tar'
    with archive.open('wb') as target:
        for chunk in stream:
            target.write(chunk)
    subprocess.run(['tar', '-xf', str(archive), '-C', str(root)], check=True)
finally:
    container.remove(force=True)
    client.close()
```

The native configuration exposes only this additional runtime prefix read-only and selects its absolute Python executable. Standard native namespace/seccomp isolation remains enabled; network remains disabled. The native artifact size ceiling is explicitly raised to 32 MiB to permit the new 16 MiB transfer workload. NsJail uses the CI-pinned executable and preparation described in `../nsjail-provisioning.txt`.

## Offline setup-inclusive profile

`config-setup-offline.json` sets `skip_environment_setup=false`. Every session creates a new virtual environment using the matched Python build; readiness probes confirmed `sys.prefix != sys.base_prefix` and pip 25.1.1 for all three backends. Docker receives `PIP_NO_INDEX=1` and `PIP_DISABLE_PIP_VERSION_CHECK=1`, so its normal pip-upgrade command resolves against its installed pip without network access. Native setup is offline already. This isolates environment creation from downloads and is **not** the default online pip-update scenario.

## Validation before measurement

The separate smoke files in `/workspace/scratch` are excluded from results. They validated JSON results, 16 MiB transfer checksums and simultaneous execution/cleanup at 1, 4 and 16 workers. No setup, compilation, package installation, unit tests or bootstrap calculations ran concurrently with the authoritative suite. Exact measured Python source snapshots are in `source/`.

The saved shell runner is `../../run_expanded.sh`. It executes profiles sequentially and stops if cleanup cannot be confirmed. Runtime paths in the JSON configuration are local to this host and must be adapted elsewhere. Repeated runs should use a fresh output directory to preserve existing evidence.
