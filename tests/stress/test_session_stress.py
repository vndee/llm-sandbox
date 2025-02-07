"""Stress tests for the SandboxSession class."""

import time
import pytest
import random
import threading
import concurrent.futures
from typing import List, Dict
from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import ResourceError

def run_heavy_computation(session: SandboxSession, n: int) -> bool:
    """Run a heavy computation in the session."""
    code = f"""
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    result = fibonacci({n})
    print(f'Fibonacci({n}) = {{result}}')
    """
    try:
        result = session.run(code)
        return result.exit_code == 0
    except ResourceError:
        return False

class TestSessionStress:
    def test_concurrent_sessions_heavy_load(self):
        """Test multiple sessions running heavy computations concurrently."""
        num_sessions = 10
        results: List[bool] = []
        start_time = time.time()

        def run_session(session_id: int):
            with SandboxSession(
                lang="python",
                max_cpu_percent=50.0,
                max_memory_bytes=256 * 1024 * 1024  # 256MB
            ) as session:
                n = random.randint(25, 35)  # Random Fibonacci number
                success = run_heavy_computation(session, n)
                return success

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_sessions) as executor:
            futures = [executor.submit(run_session, i) for i in range(num_sessions)]
            results = [f.result() for f in futures]

        end_time = time.time()
        execution_time = end_time - start_time

        success_rate = sum(results) / len(results) * 100
        print(f"\nConcurrent sessions test results:")
        print(f"Total sessions: {num_sessions}")
        print(f"Success rate: {success_rate:.2f}%")
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Average time per session: {execution_time/num_sessions:.2f} seconds")

        assert success_rate >= 80.0  # At least 80% success rate

    def test_rapid_session_creation(self):
        """Test rapid creation and destruction of sessions."""
        num_iterations = 20
        creation_times = []
        destruction_times = []

        for _ in range(num_iterations):
            # Measure session creation time
            start_time = time.time()
            session = SandboxSession(lang="python")
            session.open()
            creation_times.append(time.time() - start_time)

            # Measure session destruction time
            start_time = time.time()
            session.close()
            destruction_times.append(time.time() - start_time)

        avg_creation_time = sum(creation_times) / len(creation_times)
        avg_destruction_time = sum(destruction_times) / len(destruction_times)
        max_creation_time = max(creation_times)
        max_destruction_time = max(destruction_times)

        print(f"\nRapid session creation test results:")
        print(f"Average creation time: {avg_creation_time:.2f} seconds")
        print(f"Average destruction time: {avg_destruction_time:.2f} seconds")
        print(f"Max creation time: {max_creation_time:.2f} seconds")
        print(f"Max destruction time: {max_destruction_time:.2f} seconds")

        assert max_creation_time < 5.0  # Should create within 5 seconds
        assert max_destruction_time < 5.0  # Should destroy within 5 seconds

    def test_memory_stress(self):
        """Test behavior under memory pressure."""
        with SandboxSession(
            lang="python",
            max_memory_bytes=512 * 1024 * 1024  # 512MB
        ) as session:
            memory_allocations = []
            
            try:
                for size_mb in range(50, 600, 50):  # Try allocating increasing amounts
                    code = f"""
                    import numpy as np
                    # Allocate {size_mb}MB array
                    arr = np.zeros({size_mb * 1024 * 1024 // 8})  # 8 bytes per float64
                    print(f'Allocated {size_mb}MB array')
                    """
                    result = session.run(code)
                    if result.exit_code == 0:
                        memory_allocations.append(size_mb)
            except ResourceError as e:
                print(f"\nMemory stress test results:")
                print(f"Successfully allocated: {memory_allocations}")
                print(f"Failed at: {size_mb}MB")
                print(f"Error: {str(e)}")

            assert len(memory_allocations) > 0
            assert max(memory_allocations) <= 512  # Should not exceed limit

    def test_long_running_sessions(self):
        """Test behavior of long-running sessions."""
        session_duration = 60  # 1 minute
        check_interval = 5  # Check every 5 seconds
        
        with SandboxSession(lang="python") as session:
            start_time = time.time()
            iterations = 0
            
            while time.time() - start_time < session_duration:
                code = """
                import time
                import random
                
                # Do some work
                x = [random.random() for _ in range(1000)]
                time.sleep(1)
                print(f'Iteration completed: {sum(x)}')
                """
                
                result = session.run(code)
                assert result.exit_code == 0
                iterations += 1
                
                # Short sleep between iterations
                time.sleep(check_interval)
            
            print(f"\nLong-running session test results:")
            print(f"Total iterations: {iterations}")
            print(f"Average time per iteration: {session_duration/iterations:.2f} seconds")

    def test_parallel_file_operations(self):
        """Test parallel file operations in multiple sessions."""
        num_sessions = 5
        operations_per_session = 10
        results: Dict[int, List[bool]] = {i: [] for i in range(num_sessions)}
        
        def run_file_operations(session_id: int):
            with SandboxSession(lang="python") as session:
                for i in range(operations_per_session):
                    code = f"""
                    # Create and write to a file
                    with open(f'/tmp/test_{{session_id}}_{i}.txt', 'w') as f:
                        f.write('Test content ' * 1000)
                    
                    # Read from the file
                    with open(f'/tmp/test_{{session_id}}_{i}.txt', 'r') as f:
                        content = f.read()
                    
                    print(f'File operation {i} completed')
                    """
                    try:
                        result = session.run(code)
                        results[session_id].append(result.exit_code == 0)
                    except Exception as e:
                        results[session_id].append(False)
                        print(f"Session {session_id}, Operation {i} failed: {str(e)}")

        threads = []
        for i in range(num_sessions):
            thread = threading.Thread(target=run_file_operations, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        success_rates = {
            session_id: sum(session_results) / len(session_results) * 100
            for session_id, session_results in results.items()
        }

        print(f"\nParallel file operations test results:")
        for session_id, success_rate in success_rates.items():
            print(f"Session {session_id} success rate: {success_rate:.2f}%")

        assert all(rate >= 90.0 for rate in success_rates.values())

    def test_resource_exhaustion_recovery(self):
        """Test recovery after resource exhaustion."""
        with SandboxSession(
            lang="python",
            max_cpu_percent=50.0,
            max_memory_bytes=256 * 1024 * 1024
        ) as session:
            # Try to exhaust memory
            memory_code = """
            x = [1] * (1024 * 1024 * 300)  # Try to allocate 300MB
            """
            try:
                session.run(memory_code)
            except ResourceError:
                pass  # Expected to fail

            # Verify session is still usable
            recovery_code = """
            print('Session recovered successfully!')
            """
            result = session.run(recovery_code)
            assert result.exit_code == 0
            assert "recovered successfully" in result.output

            # Try to exhaust CPU
            cpu_code = """
            def infinite_loop():
                while True:
                    pass
            infinite_loop()
            """
            try:
                session.run(cpu_code)
            except ResourceError:
                pass  # Expected to fail

            # Verify session is still usable
            result = session.run(recovery_code)
            assert result.exit_code == 0
            assert "recovered successfully" in result.output 