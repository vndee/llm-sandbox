"""Tests for security scanning functionality."""

import pytest
from llm_sandbox.security import SecurityScanner
from llm_sandbox.exceptions import SecurityError


@pytest.fixture
def scanner():
    return SecurityScanner()


def test_safe_code(scanner):
    code = """
def add(a, b):
    return a + b

result = add(1, 2)
print(result)
"""
    issues = scanner.scan_code(code, strict=True)
    assert len(issues) == 0
    assert scanner.is_safe(code)


def test_system_calls(scanner):
    code = """
import os
os.system('ls')
"""
    with pytest.raises(SecurityError) as exc_info:
        scanner.scan_code(code, strict=True)
    assert "system command execution" in str(exc_info.value).lower()
    assert not scanner.is_safe(code)


def test_code_execution(scanner):
    code = """
user_input = input()
eval(user_input)
"""
    with pytest.raises(SecurityError) as exc_info:
        scanner.scan_code(code, strict=True)
    assert "code execution" in str(exc_info.value).lower()


def test_file_operations(scanner):
    code = """
with open('file.txt', 'w') as f:
    f.write('hello')
"""
    issues = scanner.scan_code(code, strict=False)
    assert any(i.pattern == "file_operations" for i in issues)
    assert any(i.severity == "medium" for i in issues)


def test_network_access(scanner):
    code = """
import socket
s = socket.socket()
"""
    issues = scanner.scan_code(code, strict=False)
    assert any(i.pattern == "network_access" for i in issues)


def test_shell_injection(scanner):
    code = """
import subprocess
subprocess.run('ls', shell=True)
"""
    with pytest.raises(SecurityError) as exc_info:
        scanner.scan_code(code, strict=True)
    assert "shell injection" in str(exc_info.value).lower()


@pytest.mark.parametrize("dangerous_import", ["os", "sys", "subprocess", "shutil"])
def test_dangerous_imports(scanner, dangerous_import):
    code = f"import {dangerous_import}"
    issues = scanner.scan_code(code, strict=False)
    assert len(issues) == 1
    assert issues[0].pattern == "dangerous_imports"


def test_multiple_issues(scanner):
    code = """
import os
import subprocess

def dangerous_func():
    os.system('rm -rf /')
    eval(input())
    with open('file.txt', 'w') as f:
        f.write('hello')
"""
    with pytest.raises(SecurityError) as exc_info:
        scanner.scan_code(code, strict=True)
    assert "high severity" in str(exc_info.value).lower()


@pytest.mark.parametrize(
    "strict_mode,expected_exception", [(True, True), (False, False)]
)
def test_strict_mode(scanner, strict_mode, expected_exception):
    code = """
import os
os.system('ls')
"""
    if expected_exception:
        with pytest.raises(SecurityError):
            scanner.scan_code(code, strict=strict_mode)
    else:
        issues = scanner.scan_code(code, strict=strict_mode)
        assert len(issues) > 0
        assert issues[0].severity == "high"
