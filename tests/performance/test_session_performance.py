"""Performance tests for the SandboxSession class."""

import time
import pytest
import concurrent.futures
from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import ResourceError

def measure_execution_time(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nExecution time for {func.__name__}: {execution_time:.2f} seconds")
        return result, execution_time
    return wrapper

class TestSessionPerformance:
    @measure_execution_time
    def test_session_startup_time(self):
        """Test the time taken to start a new session."""
        with SandboxSession(
            backend="docker",
            image="python:3.9.19-bullseye",
            lang="python",
        ) as session:
            result = session.run("print('Hello, World!')")
            assert result.exit_code == 0
        return True

    @measure_execution_time
    def test_code_execution_performance(self):
        """Test the performance of code execution."""
        with SandboxSession(lang="python") as session:
            # CPU-intensive operation
            code = """
            def fibonacci(n):
                if n <= 1:
                    return n
                return fibonacci(n-1) + fibonacci(n-2)
            
            result = fibonacci(35)
            print(f'Result: {result}')
            """
            result = session.run(code)
            assert result.exit_code == 0
        return True

    def test_memory_usage_profile(self):
        """Test memory usage profile during execution."""
        with SandboxSession(
            lang="python",
            max_memory_bytes=512 * 1024 * 1024  # 512MB
        ) as session:
            memory_profile = []
            
            # Gradually increasing memory usage
            for i in range(5):
                size = (i + 1) * 20  # MB
                code = f"""
                import numpy as np
                # Allocate {size}MB array
                arr = np.zeros({size * 1024 * 1024 // 8})  # 8 bytes per float64
                print(f'Allocated {size}MB array')
                """
                result = session.run(code)
                assert result.exit_code == 0
                memory_profile.append(result.resource_usage["memory_mb"])

            # Verify memory usage increases
            assert all(memory_profile[i] <= memory_profile[i+1] 
                      for i in range(len(memory_profile)-1))

    def test_concurrent_sessions(self):
        """Test performance with multiple concurrent sessions."""
        def run_session(i):
            with SandboxSession(lang="python") as session:
                code = f"""
                import time
                time.sleep(1)  # Simulate work
                print('Session {i} completed')
                """
                result = session.run(code)
                return result.exit_code == 0

        start_time = time.time()
        num_sessions = 5

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_sessions) as executor:
            results = list(executor.map(run_session, range(num_sessions)))

        end_time = time.time()
        execution_time = end_time - start_time

        assert all(results)  # All sessions should complete successfully
        print(f"\nConcurrent execution time for {num_sessions} sessions: "
              f"{execution_time:.2f} seconds")

    @pytest.mark.parametrize("file_size_mb", [1, 10, 50])
    def test_file_transfer_performance(self, file_size_mb, tmp_path):
        """Test performance of file transfers between host and container."""
        # Create a test file of specified size
        test_file = tmp_path / f"test_{file_size_mb}mb.dat"
        with open(test_file, "wb") as f:
            f.write(b"0" * (file_size_mb * 1024 * 1024))

        with SandboxSession(lang="python") as session:
            # Measure upload time
            start_time = time.time()
            session.copy_to_runtime(str(test_file), f"/sandbox/test_{file_size_mb}mb.dat")
            upload_time = time.time() - start_time

            # Measure download time
            start_time = time.time()
            output_file = tmp_path / f"output_{file_size_mb}mb.dat"
            session.copy_from_runtime(
                f"/sandbox/test_{file_size_mb}mb.dat",
                str(output_file)
            )
            download_time = time.time() - start_time

            print(f"\nFile transfer performance ({file_size_mb}MB):")
            print(f"Upload time: {upload_time:.2f} seconds")
            print(f"Download time: {download_time:.2f} seconds")
            print(f"Transfer rate (upload): {file_size_mb/upload_time:.2f} MB/s")
            print(f"Transfer rate (download): {file_size_mb/download_time:.2f} MB/s")

    def test_resource_scaling(self):
        """Test how resource usage scales with workload."""
        with SandboxSession(lang="python") as session:
            workloads = [
                (10, "Light workload"),
                (20, "Medium workload"),
                (30, "Heavy workload")
            ]

            for n, description in workloads:
                code = f"""
                def fibonacci(n):
                    if n <= 1:
                        return n
                    return fibonacci(n-1) + fibonacci(n-2)
                
                result = fibonacci({n})
                print(f'Fibonacci({n}) = {{result}}')
                """
                start_time = time.time()
                result = session.run(code)
                execution_time = time.time() - start_time

                print(f"\n{description}:")
                print(f"Execution time: {execution_time:.2f} seconds")
                print(f"CPU usage: {result.resource_usage['cpu_percent']:.2f}%")
                print(f"Memory usage: {result.resource_usage['memory_mb']:.2f} MB")

    def test_language_startup_comparison(self):
        """Compare startup times for different programming languages."""
        languages = ["python", "javascript", "ruby"]
        startup_times = {}

        for lang in languages:
            start_time = time.time()
            with SandboxSession(lang=lang) as session:
                result = session.run("print('Hello')" if lang == "python"
                                   else "console.log('Hello')" if lang == "javascript"
                                   else "puts 'Hello'")
                assert result.exit_code == 0
            startup_times[lang] = time.time() - start_time

        print("\nLanguage startup times:")
        for lang, startup_time in startup_times.items():
            print(f"{lang}: {startup_time:.2f} seconds") 