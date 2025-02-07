"""Security tests for the SandboxSession class."""

import os
import pytest
import socket
import tempfile
from pathlib import Path
from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import SecurityError

@pytest.fixture
def secure_session():
    return SandboxSession(
        lang="python",
        strict_security=True,
        max_cpu_percent=50.0,
        max_memory_bytes=100 * 1024 * 1024,  # 100MB
    )

class TestSessionSecurity:
    def test_system_command_execution(self, secure_session):
        """Test prevention of system command execution."""
        dangerous_codes = [
            """
            import os
            os.system('rm -rf /')
            """,
            """
            import subprocess
            subprocess.run(['cat', '/etc/passwd'])
            """,
            """
            from subprocess import Popen
            Popen('echo "malicious" > /tmp/test')
            """,
            """
            __import__('os').system('whoami')
            """,
        ]

        for code in dangerous_codes:
            with pytest.raises(SecurityError) as exc_info:
                secure_session.run(code)
            assert "security" in str(exc_info.value).lower()

    def test_network_access(self, secure_session):
        """Test prevention of unauthorized network access."""
        network_codes = [
            """
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(('google.com', 80))
            """,
            """
            import urllib.request
            urllib.request.urlopen('http://example.com')
            """,
            """
            import requests
            requests.get('https://api.example.com')
            """,
        ]

        for code in network_codes:
            with pytest.raises(SecurityError) as exc_info:
                secure_session.run(code)
            assert "network" in str(exc_info.value).lower()

    def test_file_system_access(self, secure_session):
        """Test file system access restrictions."""
        file_access_codes = [
            """
            with open('/etc/passwd', 'r') as f:
                print(f.read())
            """,
            """
            import os
            os.listdir('/')
            """,
            """
            with open('/root/.ssh/id_rsa', 'r') as f:
                print(f.read())
            """,
        ]

        for code in file_access_codes:
            with pytest.raises(SecurityError) as exc_info:
                secure_session.run(code)
            assert "file" in str(exc_info.value).lower()

    def test_module_import_restrictions(self, secure_session):
        """Test prevention of importing dangerous modules."""
        dangerous_imports = [
            "import socket",
            "import subprocess",
            "from subprocess import Popen",
            "import multiprocessing",
            "__import__('subprocess')",
        ]

        for code in dangerous_imports:
            with pytest.raises(SecurityError) as exc_info:
                secure_session.run(code)
            assert "module" in str(exc_info.value).lower()

    def test_resource_limit_enforcement(self, secure_session):
        """Test enforcement of resource limits."""
        # Test CPU limit
        cpu_intensive_code = """
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        fibonacci(100)  # Very CPU intensive
        """
        with pytest.raises(ResourceError) as exc_info:
            secure_session.run(cpu_intensive_code)
        assert "CPU" in str(exc_info.value)

        # Test memory limit
        memory_intensive_code = """
        x = [1] * (1024 * 1024 * 200)  # Try to allocate 200MB
        """
        with pytest.raises(ResourceError) as exc_info:
            secure_session.run(memory_intensive_code)
        assert "memory" in str(exc_info.value).lower()

    def test_code_injection_prevention(self, secure_session):
        """Test prevention of code injection attempts."""
        injection_attempts = [
            """
            exec('import os; os.system("rm -rf /")')
            """,
            """
            eval('__import__("os").system("whoami")')
            """,
            """
            globals()['__builtins__'].__import__('os').system('ls')
            """,
        ]

        for code in injection_attempts:
            with pytest.raises(SecurityError) as exc_info:
                secure_session.run(code)
            assert "injection" in str(exc_info.value).lower()

    def test_sandbox_escape_prevention(self, secure_session):
        """Test prevention of sandbox escape attempts."""
        escape_attempts = [
            """
            import os
            os.chroot('/')
            """,
            """
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            """,
            """
            with open('/proc/self/mem', 'wb') as f:
                f.write(b'malicious')
            """,
        ]

        for code in escape_attempts:
            with pytest.raises(SecurityError) as exc_info:
                secure_session.run(code)
            assert "security" in str(exc_info.value).lower()

    def test_environment_variable_access(self, secure_session):
        """Test environment variable access restrictions."""
        env_access_code = """
        import os
        print(os.environ)
        """
        with pytest.raises(SecurityError) as exc_info:
            secure_session.run(env_access_code)
        assert "environment" in str(exc_info.value).lower()

    def test_process_creation_prevention(self, secure_session):
        """Test prevention of new process creation."""
        process_creation_codes = [
            """
            import multiprocessing
            p = multiprocessing.Process(target=lambda: None)
            p.start()
            """,
            """
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor() as executor:
                executor.submit(lambda: None)
            """,
        ]

        for code in process_creation_codes:
            with pytest.raises(SecurityError) as exc_info:
                secure_session.run(code)
            assert "process" in str(exc_info.value).lower()

    def test_file_write_permissions(self, secure_session):
        """Test file write permission restrictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed_path = os.path.join(tmpdir, "test.txt")
            disallowed_paths = [
                "/etc/passwd",
                "/tmp/malicious",
                "~/.bashrc",
            ]

            # Test allowed path
            code = f"""
            with open('{allowed_path}', 'w') as f:
                f.write('test')
            """
            result = secure_session.run(code)
            assert result.exit_code == 0

            # Test disallowed paths
            for path in disallowed_paths:
                code = f"""
                with open('{path}', 'w') as f:
                    f.write('malicious')
                """
                with pytest.raises(SecurityError) as exc_info:
                    secure_session.run(code)
                assert "file" in str(exc_info.value).lower()

    def test_network_isolation(self, secure_session):
        """Test complete network isolation."""
        def try_connect(host, port):
            try:
                socket.create_connection((host, port), timeout=1)
                return True
            except:
                return False

        code = """
        import socket
        
        def try_connect(host, port):
            try:
                socket.create_connection((host, port), timeout=1)
                return True
            except:
                return False

        # Try various network access attempts
        results = []
        results.append(try_connect('8.8.8.8', 53))  # DNS
        results.append(try_connect('127.0.0.1', 80))  # Local HTTP
        results.append(try_connect('example.com', 443))  # HTTPS
        print(results)
        """

        with pytest.raises(SecurityError) as exc_info:
            secure_session.run(code)
        assert "network" in str(exc_info.value).lower() 