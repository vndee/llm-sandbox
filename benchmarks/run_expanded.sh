#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path('benchmarks/results/expanded-20261016')
profiles = [
    ('baseline', 'config-common.json', ['--rounds', '30', '--iterations', '10'], 20261016),
    ('setup', 'config-setup-offline.json', ['--rounds', '10', '--iterations', '1'], 20261017),
    ('workload', 'config-common.json', ['--rounds', '10', '--iterations', '10', '--transfer-iterations', '3'], 20261018),
    ('concurrency', 'config-common.json', ['--batches', '5', '--levels', '1', '4', '16', '--iterations', '10', '--workloads', 'marker', 'json'], 20261019),
]
for profile, config, options, seed in profiles:
    output = root / f'{profile}.json'
    command = [
        '.venv/bin/python', '-m', 'benchmarks.experiment_suite',
        '--profile', profile, '--backends', 'docker', 'bubblewrap', 'nsjail',
        '--config', str(root / config), '--output', str(output),
        '--warmups', '2', '--seed', str(seed), *options,
    ]
    print(f'Starting {profile}', flush=True)
    subprocess.run(command, check=True)
    document = json.loads(output.read_text())
    if document.get('stopped_early') or document.get('interrupted'):
        raise SystemExit(f'Stopping after {profile}: inspect cleanup/interruption diagnostics')
print('All expanded profiles complete.', flush=True)
PY
