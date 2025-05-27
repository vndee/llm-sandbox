"""Benchmarking and performance tests for security policy evaluation.

This module tests the performance impact of security policies and provides
benchmarks for different policy configurations.
"""

import time
import logging
from typing import List, Dict, Tuple
from statistics import mean, median, stdev

from llm_sandbox import SandboxSession
from llm_sandbox.security import (
    SecurityIssueSeverity,
    SecurityPattern,
    DangerousModule,
    SecurityPolicy,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SecurityBenchmark:
    """Benchmark suite for security policy performance."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_policy_creation(self, num_patterns: int, num_modules: int) -> float:
        """Benchmark security policy creation time.
        
        Args:
            num_patterns: Number of security patterns to create
            num_modules: Number of dangerous modules to create
            
        Returns:
            Time taken to create the policy in seconds
        """
        start_time = time.time()
        
        patterns = []
        for i in range(num_patterns):
            patterns.append(SecurityPattern(
                pattern=rf"\bpattern_{i}\s*\(",
                description=f"Test pattern {i}",
                severity=SecurityIssueSeverity.MEDIUM,
            ))
        
        modules = []
        for i in range(num_modules):
            modules.append(DangerousModule(
                name=f"module_{i}",
                description=f"Test module {i}",
                severity=SecurityIssueSeverity.MEDIUM,
            ))
        
        policy = SecurityPolicy(
            safety_level=SecurityIssueSeverity.MEDIUM,
            patterns=patterns,
            dangerous_modules=modules,
        )
        
        end_time = time.time()
        return end_time - start_time
    
    def benchmark_code_analysis(self, policy: SecurityPolicy, test_codes: List[str]) -> List[float]:
        """Benchmark code analysis performance.
        
        Args:
            policy: Security policy to use
            test_codes: List of code strings to analyze
            
        Returns:
            List of analysis times for each code sample
        """
        analysis_times = []
        
        with SandboxSession(lang="python", security_policy=policy, verbose=False) as session:
            for code in test_codes:
                start_time = time.time()
                session.is_safe(code)
                end_time = time.time()
                analysis_times.append(end_time - start_time)
        
        return analysis_times
    
    def run_policy_size_benchmark(self) -> None:
        """Benchmark performance with different policy sizes."""
        logger.info("Running Policy Size Benchmark...")
        
        policy_sizes = [(10, 5), (50, 25), (100, 50), (200, 100), (500, 250)]
        
        creation_times = []
        for num_patterns, num_modules in policy_sizes:
            times = []
            for _ in range(5):  # Run multiple times for accuracy
                creation_time = self.benchmark_policy_creation(num_patterns, num_modules)
                times.append(creation_time)
            
            avg_time = mean(times)
            creation_times.append(avg_time)
            logger.info(f"  Policy ({num_patterns}P, {num_modules}M): {avg_time:.4f}s")
        
        self.results['policy_creation'] = dict(zip(policy_sizes, creation_times))
    
    def run_code_complexity_benchmark(self) -> None:
        """Benchmark performance with different code complexities."""
        logger.info("Running Code Complexity Benchmark...")
        
        # Create a standard policy
        policy = self._create_standard_policy()
        
        # Different complexity levels of code
        test_cases = {
            "Simple": [
                "print('hello')",
                "x = 1 + 1",
                "import math",
            ],
            "Medium": [
                "\n".join([
                    "import json",
                    "data = {'key': 'value'}",
                    "for i in range(10):",
                    "    print(json.dumps(data))",
                ]),
                "\n".join([
                    "def factorial(n):",
                    "    if n <= 1:",
                    "        return 1",
                    "    return n * factorial(n-1)",
                    "print(factorial(5))",
                ]),
            ],
            "Complex": [
                "\n".join([
                    "import itertools",
                    "import functools",
                    "import collections",
                    "\n".join([f"x{i} = {i} * 2" for i in range(50)]),
                    "result = functools.reduce(lambda a, b: a + b, [x0, x1, x2, x3, x4])",
                    "print(result)",
                ]),
                "\n".join([
                    "class DataProcessor:",
                    "    def __init__(self, data):",
                    "        self.data = data",
                    "    def process(self):",
                    "        return [x**2 for x in self.data if x % 2 == 0]",
                    "processor = DataProcessor(list(range(100)))",
                    "print(len(processor.process()))",
                ]),
            ],
        }
        
        complexity_results = {}
        for complexity, codes in test_cases.items():
            analysis_times = self.benchmark_code_analysis(policy, codes)
            avg_time = mean(analysis_times)
            complexity_results[complexity] = {
                'avg_time': avg_time,
                'times': analysis_times,
                'samples': len(codes)
            }
            logger.info(f"  {complexity} Code: {avg_time:.4f}s avg ({len(codes)} samples)")
        
        self.results['code_complexity'] = complexity_results
    
    def run_pattern_matching_benchmark(self) -> None:
        """Benchmark regex pattern matching performance."""
        logger.info("Running Pattern Matching Benchmark...")
        
        # Test different numbers of patterns
        pattern_counts = [1, 5, 10, 25, 50, 100]
        test_code = "\n".join([
            "import os",
            "import sys",
            "import subprocess",
            "result = eval('2 + 2')",
            "exec('print(1)')",
            "os.system('echo test')",
        ])
        
        pattern_results = {}
        for count in pattern_counts:
            # Create policy with specified number of patterns
            patterns = []
            for i in range(count):
                if i < 6:  # Use real patterns for first few
                    real_patterns = [
                        r"\bos\.system\s*\(",
                        r"\beval\s*\(",
                        r"\bexec\s*\(",
                        r"\bsubprocess\.(run|call)\s*\(",
                        r"\b__import__\s*\(",
                        r"\bopen\s*\([^)]*['\"][wa]['\"][^)]*\)",
                    ]
                    pattern = real_patterns[i]
                else:
                    pattern = rf"\bdummy_pattern_{i}\s*\("
                
                patterns.append(SecurityPattern(
                    pattern=pattern,
                    description=f"Pattern {i}",
                    severity=SecurityIssueSeverity.MEDIUM,
                ))
            
            policy = SecurityPolicy(
                safety_level=SecurityIssueSeverity.MEDIUM,
                patterns=patterns,
                dangerous_modules=[],
            )
            
            # Benchmark analysis
            times = []
            for _ in range(10):  # Multiple runs
                analysis_times = self.benchmark_code_analysis(policy, [test_code])
                times.extend(analysis_times)
            
            avg_time = mean(times)
            pattern_results[count] = {
                'avg_time': avg_time,
                'std_dev': stdev(times) if len(times) > 1 else 0,
                'samples': len(times)
            }
            logger.info(f"  {count} patterns: {avg_time:.4f}s avg")
        
        self.results['pattern_matching'] = pattern_results
    
    def run_memory_usage_test(self) -> None:
        """Test memory usage of security policies."""
        logger.info("Running Memory Usage Test...")
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Create large policy
        large_policy = self._create_large_policy(1000, 500)
        
        after_policy_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Create multiple sessions
        sessions = []
        for _ in range(10):
            session = SandboxSession(lang="python", security_policy=large_policy, verbose=False)
            sessions.append(session)
        
        after_sessions_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Clean up
        del sessions
        del large_policy
        
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        memory_results = {
            'initial_mb': initial_memory,
            'after_policy_mb': after_policy_memory,
            'after_sessions_mb': after_sessions_memory,
            'final_mb': final_memory,
            'policy_overhead_mb': after_policy_memory - initial_memory,
            'sessions_overhead_mb': after_sessions_memory - after_policy_memory,
        }
        
        logger.info(f"  Initial Memory: {initial_memory:.2f}MB")
        logger.info(f"  After Large Policy: {after_policy_memory:.2f}MB (+{memory_results['policy_overhead_mb']:.2f}MB)")
        logger.info(f"  After 10 Sessions: {after_sessions_memory:.2f}MB (+{memory_results['sessions_overhead_mb']:.2f}MB)")
        logger.info(f"  Final Memory: {final_memory:.2f}MB")
        
        self.results['memory_usage'] = memory_results
    
    def _create_standard_policy(self) -> SecurityPolicy:
        """Create a standard security policy for benchmarking."""
        patterns = [
            SecurityPattern(
                pattern=r"\bos\.system\s*\(",
                description="System command execution",
                severity=SecurityIssueSeverity.HIGH,
            ),
            SecurityPattern(
                pattern=r"\beval\s*\(",
                description="Dynamic evaluation",
                severity=SecurityIssueSeverity.MEDIUM,
            ),
            SecurityPattern(
                pattern=r"\bexec\s*\(",
                description="Dynamic execution",
                severity=SecurityIssueSeverity.MEDIUM,
            ),
        ]
        
        modules = [
            DangerousModule(
                name="os",
                description="Operating system interface",
                severity=SecurityIssueSeverity.HIGH,
            ),
            DangerousModule(
                name="subprocess",
                description="Subprocess management",
                severity=SecurityIssueSeverity.HIGH,
            ),
        ]
        
        return SecurityPolicy(
            safety_level=SecurityIssueSeverity.MEDIUM,
            patterns=patterns,
            dangerous_modules=modules,
        )
    
    def _create_large_policy(self, num_patterns: int, num_modules: int) -> SecurityPolicy:
        """Create a large security policy for stress testing."""
        patterns = []
        for i in range(num_patterns):
            patterns.append(SecurityPattern(
                pattern=rf"\btest_pattern_{i}\s*\(",
                description=f"Test pattern {i}",
                severity=SecurityIssueSeverity.MEDIUM,
            ))
        
        modules = []
        for i in range(num_modules):
            modules.append(DangerousModule(
                name=f"test_module_{i}",
                description=f"Test module {i}",
                severity=SecurityIssueSeverity.MEDIUM,
            ))
        
        return SecurityPolicy(
            safety_level=SecurityIssueSeverity.MEDIUM,
            patterns=patterns,
            dangerous_modules=modules,
        )
    
    def generate_report(self) -> None:
        """Generate a comprehensive performance report."""
        logger.info("\n" + "="*60)
        logger.info("SECURITY POLICY PERFORMANCE REPORT")
        logger.info("="*60)
        
        if 'policy_creation' in self.results:
            logger.info("\n📊 Policy Creation Performance:")
            for (patterns, modules), time_taken in self.results['policy_creation'].items():
                logger.info(f"  {patterns}P/{modules}M: {time_taken:.4f}s")
        
        if 'code_complexity' in self.results:
            logger.info("\n📊 Code Analysis Performance by Complexity:")
            for complexity, data in self.results['code_complexity'].items():
                logger.info(f"  {complexity}: {data['avg_time']:.4f}s avg ({data['samples']} samples)")
        
        if 'pattern_matching' in self.results:
            logger.info("\n📊 Pattern Matching Performance:")
            for count, data in self.results['pattern_matching'].items():
                logger.info(f"  {count} patterns: {data['avg_time']:.4f}s ± {data['std_dev']:.4f}s")
        
        if 'memory_usage' in self.results:
            logger.info("\n📊 Memory Usage:")
            data = self.results['memory_usage']
            logger.info(f"  Policy Overhead: {data['policy_overhead_mb']:.2f}MB")
            logger.info(f"  Sessions Overhead: {data['sessions_overhead_mb']:.2f}MB")
        
        logger.info("\n✅ Performance benchmarking completed!")


def run_comprehensive_benchmarks() -> None:
    """Run all security policy benchmarks."""
    benchmark = SecurityBenchmark()
    
    try:
        benchmark.run_policy_size_benchmark()
        benchmark.run_code_complexity_benchmark()
        benchmark.run_pattern_matching_benchmark()
        
        # Only run memory test if psutil is available
        try:
            import psutil
            benchmark.run_memory_usage_test()
        except ImportError:
            logger.warning("psutil not available, skipping memory usage test")
        
        benchmark.generate_report()
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


def run_scalability_test() -> None:
    """Test scalability with extreme scenarios."""
    logger.info("\nRunning Scalability Test...")
    
    # Test with very large code files
    large_code = "\n".join([
        f"variable_{i} = {i} * 2 + 1" for i in range(1000)
    ]) + "\nprint('Large code executed')"
    
    # Test with policy containing many patterns
    patterns = []
    for i in range(100):
        patterns.append(SecurityPattern(
            pattern=rf"\btest_function_{i}\s*\(",
            description=f"Test function {i}",
            severity=SecurityIssueSeverity.LOW,
        ))
    
    policy = SecurityPolicy(
        safety_level=SecurityIssueSeverity.MEDIUM,
        patterns=patterns,
        dangerous_modules=[],
    )
    
    start_time = time.time()
    with SandboxSession(lang="python", security_policy=policy, verbose=False) as session:
        is_safe, violations = session.is_safe(large_code)
    end_time = time.time()
    
    logger.info(f"  Large code analysis: {end_time - start_time:.4f}s")
    logger.info(f"  Code size: {len(large_code)} characters")
    logger.info(f"  Policy size: {len(patterns)} patterns")
    logger.info(f"  Result: {'Safe' if is_safe else 'Blocked'}")
    logger.info(f"  Violations: {len(violations)}")


if __name__ == "__main__":
    logger.info("LLM Sandbox Security Policy Benchmarks")
    logger.info("=====================================")
    
    try:
        run_comprehensive_benchmarks()
        run_scalability_test()
        
        logger.info("\n🎉 All benchmarks completed successfully!")
        
    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}")
        raise
