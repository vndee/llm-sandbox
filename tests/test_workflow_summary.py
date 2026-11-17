"""Regression tests for the issue-summary workflow comment step."""

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "summary.yml"
STEP_NAME = "Comment with AI summary"


def _comment_step() -> dict:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = data["jobs"]["summary"]["steps"]
    return next(step for step in steps if step.get("name") == STEP_NAME)


def test_comment_step_keeps_expressions_out_of_the_script() -> None:
    """The run script must not contain expressions that Actions expands before bash runs."""
    step = _comment_step()

    assert "${{" not in step["run"]
    assert step["env"]["RESPONSE"] == "${{ steps.inference.outputs.response }}"


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="requires a POSIX bash")
def test_comment_step_passes_the_response_to_gh_literally(tmp_path: Path) -> None:
    """Gh must receive the response verbatim and its shell substitutions must not run."""
    step = _comment_step()
    marker = tmp_path / "substituted"
    args_file = tmp_path / "gh-args"
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    gh = stub_dir / "gh"
    gh.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$GH_ARGS_FILE"\n', encoding="utf-8")
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    response = f"it's fixed $(touch {marker}) `touch {marker}`"
    env = {
        "PATH": f"{stub_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        "GH_ARGS_FILE": str(args_file),
        "ISSUE_NUMBER": "42",
        "RESPONSE": response,
    }

    subprocess.run(  # noqa: S603
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],  # noqa: S607
        env=env,
        check=True,
    )

    assert args_file.read_text(encoding="utf-8").split("\0")[:-1] == ["issue", "comment", "42", "--body", response]
    assert not marker.exists()
