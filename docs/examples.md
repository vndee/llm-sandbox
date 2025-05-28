# Examples

This section provides practical examples of using LLM Sandbox for executing LLM-generated code in real-world AI agent scenarios. These examples focus on the most common use cases where LLMs generate code that needs to be executed safely.

## LLM Framework Integrations

### LangChain Integration

```python
--8<-- "examples/langchain_tool.py"
```

### LangGraph Integration

```python
--8<-- "examples/langgraph_tool.py"
```

### LlamaIndex Integration

```python
--8<-- "examples/llamaindex_tool.py"
```

## Code Generation Patterns

### 1. Self-Correcting Code Generator

```python
from llm_sandbox import SandboxSession
import openai

class SelfCorrectingCodeGenerator:
    """Generate and iteratively improve code using LLM feedback"""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.max_iterations = 3

    def generate_and_test_code(self, task: str, test_cases: list) -> dict:
        """Generate code and iteratively improve it based on test results"""

        iteration = 0
        current_code = None

        while iteration < self.max_iterations:
            iteration += 1

            # Generate or improve code
            if current_code is None:
                prompt = f"Write Python code to: {task}\n\nInclude proper error handling and documentation."
            else:
                prompt = f"""
                The previous code failed. Here's what happened:

                Code: {current_code}
                Error: {last_error}

                Fix the issues and improve the code to: {task}
                """

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert Python developer. Write clean, efficient, well-tested code."},
                    {"role": "user", "content": prompt}
                ]
            )

            current_code = response.choices[0].message.content

            # Test the generated code
            with SandboxSession(lang="python") as session:
                # Setup test environment
                test_result = session.run(current_code)

                if test_result.exit_code == 0:
                    # Run test cases
                    all_passed = True
                    test_outputs = []

                    for test_case in test_cases:
                        test_code = f"""
# Test case: {test_case['description']}
try:
    result = {test_case['code']}
    expected = {test_case['expected']}
    passed = result == expected
    print(f"Test '{test_case['description']}': {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  Expected: {expected}, Got: {result}")
except Exception as e:
    print(f"Test '{test_case['description']}': ERROR - {e}")
    passed = False
"""
                        test_output = session.run(test_code)
                        test_outputs.append(test_output.stdout)

                        if "FAIL" in test_output.stdout or "ERROR" in test_output.stdout:
                            all_passed = False

                    if all_passed:
                        return {
                            "success": True,
                            "code": current_code,
                            "iterations": iteration,
                            "test_results": test_outputs
                        }
                    else:
                        last_error = "Some test cases failed: " + "\n".join(test_outputs)
                else:
                    last_error = test_result.stderr

        return {
            "success": False,
            "code": current_code,
            "iterations": iteration,
            "final_error": last_error
        }

# Example usage
generator = SelfCorrectingCodeGenerator("your-api-key")

test_cases = [
    {
        "description": "Basic sorting",
        "code": "sort_list([3, 1, 4, 1, 5])",
        "expected": [1, 1, 3, 4, 5]
    },
    {
        "description": "Empty list",
        "code": "sort_list([])",
        "expected": []
    },
    {
        "description": "Single element",
        "code": "sort_list([42])",
        "expected": [42]
    }
]

result = generator.generate_and_test_code(
    "Create a function called 'sort_list' that sorts a list of numbers in ascending order",
    test_cases
)

print(f"Success: {result['success']}")
print(f"Iterations: {result['iterations']}")
if result['success']:
    print("Generated code:", result['code'])
```

### 2. Multi-Language Code Translator

```python
from llm_sandbox import SandboxSession
import openai

class CodeTranslator:
    """Translate code between different programming languages"""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.supported_languages = ["python", "javascript", "java", "cpp", "go"]

    def translate_code(self, source_code: str, source_lang: str, target_lang: str) -> dict:
        """Translate code from one language to another and test it"""

        translation_prompt = f"""
        Translate this {source_lang} code to {target_lang}:

        {source_code}

        Requirements:
        1. Maintain the same functionality
        2. Use idiomatic {target_lang} patterns
        3. Include proper error handling
        4. Add comments explaining the translation choices
        5. Ensure the code is runnable and follows best practices
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are an expert in both {source_lang} and {target_lang}. Provide accurate, idiomatic translations."},
                {"role": "user", "content": translation_prompt}
            ]
        )

        translated_code = response.choices[0].message.content

        # Test both original and translated code
        original_result = self._test_code(source_code, source_lang)
        translated_result = self._test_code(translated_code, target_lang)

        return {
            "source_language": source_lang,
            "target_language": target_lang,
            "original_code": source_code,
            "translated_code": translated_code,
            "original_output": original_result,
            "translated_output": translated_result,
            "translation_successful": translated_result["success"],
            "outputs_match": self._compare_outputs(original_result, translated_result)
        }

    def _test_code(self, code: str, language: str) -> dict:
        """Test code execution in specified language"""
        try:
            with SandboxSession(lang=language) as session:
                result = session.run(code)
                return {
                    "success": result.exit_code == 0,
                    "output": result.stdout,
                    "error": result.stderr
                }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }

    def _compare_outputs(self, original: dict, translated: dict) -> bool:
        """Compare outputs to verify translation accuracy"""
        if not (original["success"] and translated["success"]):
            return False

        # Simple output comparison (can be enhanced for specific needs)
        return original["output"].strip() == translated["output"].strip()

# Example usage
translator = CodeTranslator("your-api-key")

python_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
"""

translation = translator.translate_code(python_code, "python", "javascript")

print(f"Translation successful: {translation['translation_successful']}")
print(f"Outputs match: {translation['outputs_match']}")
print("Translated code:", translation['translated_code'])
```

## Security and Monitoring

### Secure Code Execution Service

```python
from llm_sandbox import SandboxSession
from llm_sandbox.security import SecurityPolicy, RestrictedModule, SecurityIssueSeverity
import hashlib
import time
import logging

class SecureAICodeExecutor:
    """Production-ready secure execution service for AI-generated code"""

    def __init__(self):
        self.execution_log = []
        self.security_policy = self._create_security_policy()
        self.logger = logging.getLogger(__name__)

    def _create_security_policy(self) -> SecurityPolicy:
        """Create comprehensive security policy for AI-generated code"""
        return SecurityPolicy(
            severity_threshold=SecurityIssueSeverity.MEDIUM,
            restricted_modules=[
                RestrictedModule("os", "Operating system access", SecurityIssueSeverity.HIGH),
                RestrictedModule("subprocess", "Process execution", SecurityIssueSeverity.HIGH),
                RestrictedModule("socket", "Network operations", SecurityIssueSeverity.MEDIUM),
                RestrictedModule("ctypes", "Foreign function library", SecurityIssueSeverity.HIGH)
            ]
        )

    def execute_ai_code(
        self,
        code: str,
        user_id: str,
        ai_model: str,
        language: str = "python",
        timeout: int = 30
    ) -> dict:
        """Execute AI-generated code with comprehensive security and monitoring"""

        execution_id = hashlib.sha256(f"{user_id}{time.time()}{code}".encode()).hexdigest()[:16]

        # Log execution attempt
        log_entry = {
            "execution_id": execution_id,
            "user_id": user_id,
            "ai_model": ai_model,
            "language": language,
            "timestamp": time.time(),
            "code_length": len(code),
            "code_hash": hashlib.sha256(code.encode()).hexdigest()
        }

        try:
            with SandboxSession(
                lang=language,
                security_policy=self.security_policy,
                runtime_config={
                    "timeout": timeout,
                    "mem_limit": "256m",
                    "cpu_count": 1,
                    "network_mode": "none",
                    "read_only": True
                }
            ) as session:
                # Security check
                is_safe, violations = session.is_safe(code)

                if not is_safe:
                    log_entry["security_violations"] = [v.description for v in violations]
                    self.execution_log.append(log_entry)

                    return {
                        "success": False,
                        "execution_id": execution_id,
                        "error": "Security policy violations detected",
                        "violations": [
                            {"description": v.description, "severity": v.severity.name}
                            for v in violations
                        ]
                    }

                # Execute code
                result = session.run(code)

                log_entry["success"] = result.exit_code == 0
                log_entry["execution_time"] = time.time() - log_entry["timestamp"]
                self.execution_log.append(log_entry)

                return {
                    "success": result.exit_code == 0,
                    "execution_id": execution_id,
                    "output": result.stdout[:5000],  # Limit output size
                    "error": result.stderr[:1000] if result.stderr else None,
                    "execution_time": log_entry["execution_time"]
                }

        except Exception as e:
            log_entry["error"] = str(e)
            self.execution_log.append(log_entry)

            return {
                "success": False,
                "execution_id": execution_id,
                "error": f"Execution failed: {str(e)}"
            }

    def get_execution_stats(self, user_id: str = None) -> dict:
        """Get execution statistics"""
        logs = self.execution_log
        if user_id:
            logs = [log for log in logs if log["user_id"] == user_id]

        if not logs:
            return {"total": 0}

        total = len(logs)
        successful = sum(1 for log in logs if log.get("success", False))
        violations = sum(1 for log in logs if "security_violations" in log)

        return {
            "total_executions": total,
            "successful_executions": successful,
            "security_violations": violations,
            "success_rate": successful / total if total > 0 else 0,
            "violation_rate": violations / total if total > 0 else 0
        }

# Example usage
executor = SecureAICodeExecutor()

# Example AI-generated code execution
ai_code = """
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave Generated by AI')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Successfully generated sine wave plot!")
"""

result = executor.execute_ai_code(
    code=ai_code,
    user_id="ai_agent_001",
    ai_model="gpt-4",
    language="python"
)

print(f"Execution successful: {result['success']}")
if result['success']:
    print("Output:", result['output'])
else:
    print("Error:", result['error'])

# Get statistics
stats = executor.get_execution_stats()
print(f"Success rate: {stats['success_rate']:.2%}")
```

## Performance Optimization

### Parallel AI Code Processing

```python
from llm_sandbox import SandboxSession
import concurrent.futures
import time

class ParallelAICodeProcessor:
    """Process multiple AI-generated code snippets in parallel"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def process_code_batch(self, code_tasks: list) -> list:
        """Process multiple code tasks in parallel"""

        def execute_single_task(task):
            task_id, code, language = task["id"], task["code"], task.get("language", "python")

            start_time = time.time()

            try:
                with SandboxSession(
                    lang=language,
                    runtime_config={"timeout": 30, "mem_limit": "128m"}
                ) as session:
                    result = session.run(code)

                    return {
                        "task_id": task_id,
                        "success": result.exit_code == 0,
                        "output": result.stdout,
                        "error": result.stderr,
                        "execution_time": time.time() - start_time
                    }
            except Exception as e:
                return {
                    "task_id": task_id,
                    "success": False,
                    "output": "",
                    "error": str(e),
                    "execution_time": time.time() - start_time
                }

        # Execute tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(execute_single_task, task): task
                for task in code_tasks
            }

            results = []
            for future in concurrent.futures.as_completed(future_to_task):
                results.append(future.result())

        return sorted(results, key=lambda x: x["task_id"])

# Example usage
processor = ParallelAICodeProcessor(max_workers=3)

# Batch of AI-generated code tasks
code_tasks = [
    {
        "id": 1,
        "code": "print('Task 1: Hello from AI!')\nprint(sum(range(100)))",
        "language": "python"
    },
    {
        "id": 2,
        "code": "import math\nprint(f'Task 2: Pi = {math.pi:.6f}')",
        "language": "python"
    },
    {
        "id": 3,
        "code": "console.log('Task 3: JavaScript execution')\nconsole.log(Array.from({length: 10}, (_, i) => i * 2))",
        "language": "javascript"
    }
]

results = processor.process_code_batch(code_tasks)

for result in results:
    print(f"Task {result['task_id']}: {'✓' if result['success'] else '✗'}")
    print(f"  Execution time: {result['execution_time']:.3f}s")
    if result['success']:
        print(f"  Output: {result['output'][:100]}...")
    else:
        print(f"  Error: {result['error']}")
```

## Best Practices

### 1. Code Validation Pipeline

```python
from llm_sandbox import SandboxSession
import openai

class AICodeValidator:
    """Validate AI-generated code before execution"""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def validate_and_improve_code(self, code: str, requirements: str) -> dict:
        """Validate code and suggest improvements"""

        validation_prompt = f"""
        Review this code for:
        1. Syntax errors
        2. Logic issues
        3. Security concerns
        4. Performance problems
        5. Best practices compliance

        Requirements: {requirements}
        Code: {code}

        Provide:
        - Issues found (if any)
        - Improved version of the code
        - Explanation of changes
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a senior code reviewer. Identify issues and provide improved code."},
                {"role": "user", "content": validation_prompt}
            ]
        )

        review_result = response.choices[0].message.content

        # Test both original and improved code
        original_test = self._test_code(code)

        # Extract improved code from review (simplified extraction)
        improved_code = self._extract_improved_code(review_result)
        improved_test = self._test_code(improved_code) if improved_code else None

        return {
            "original_code": code,
            "review_feedback": review_result,
            "improved_code": improved_code,
            "original_test_result": original_test,
            "improved_test_result": improved_test,
            "improvement_successful": improved_test and improved_test["success"]
        }

    def _test_code(self, code: str) -> dict:
        """Test code execution"""
        try:
            with SandboxSession(lang="python") as session:
                result = session.run(code)
                return {
                    "success": result.exit_code == 0,
                    "output": result.stdout,
                    "error": result.stderr
                }
        except Exception as e:
            return {"success": False, "output": "", "error": str(e)}

    def _extract_improved_code(self, review_text: str) -> str:
        """Extract improved code from review text"""
        # Simple extraction - look for code blocks
        import re
        code_blocks = re.findall(r'```python\n(.*?)\n```', review_text, re.DOTALL)
        return code_blocks[-1] if code_blocks else None

# Example usage
validator = AICodeValidator("your-api-key")

ai_generated_code = """
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

numbers = [1, 2, 3, 4, 5]
print(calculate_average(numbers))
"""

validation = validator.validate_and_improve_code(
    code=ai_generated_code,
    requirements="Function should handle edge cases like empty lists and non-numeric inputs"
)

print("Validation Results:")
print("Original successful:", validation["original_test_result"]["success"])
print("Improvement successful:", validation["improvement_successful"])
print("Review feedback:", validation["review_feedback"])
```

## Next Steps

- Learn about [Security Best Practices](security.md) for AI-generated code
- Explore [Configuration Options](configuration.md) for production deployments
- Check [API Reference](api-reference.md) for advanced features
- Read about [Contributing](contributing.md) to extend functionality

For more examples and use cases, visit our [GitHub repository](https://github.com/vndee/llm-sandbox/tree/main/examples).
