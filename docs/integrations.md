# LLM Framework Integrations

LLM Sandbox seamlessly integrates with popular LLM frameworks to provide secure code execution capabilities. This guide covers integration patterns and examples.

## LangChain Integration

### Basic Tool Integration

```python
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from llm_sandbox import SandboxSession

@tool
def execute_code(code: str, language: str = "python") -> str:
    """
    Execute code in a secure sandbox environment.

    Args:
        code: The code to execute
        language: Programming language (python, javascript, java, cpp, go, ruby)

    Returns:
        The execution output
    """
    with SandboxSession(lang=language, verbose=False) as session:
        result = session.run(code)
        if result.exit_code != 0:
            return f"Error: {result.stderr}"
        return result.stdout

# Create agent with code execution tool
llm = OpenAI(temperature=0)
tools = [execute_code]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use the agent
response = agent.run(
    "Write and execute Python code to calculate the factorial of 10"
)
print(response)
```

### Advanced Tool with Libraries

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Optional

class CodeExecutionInput(BaseModel):
    """Input for code execution tool"""
    code: str = Field(description="The code to execute")
    language: str = Field(default="python", description="Programming language")
    libraries: Optional[List[str]] = Field(
        default=None,
        description="Libraries to install before execution"
    )

def execute_code_with_libs(
    code: str,
    language: str = "python",
    libraries: Optional[List[str]] = None
) -> str:
    """Execute code with optional library installation"""
    try:
        with SandboxSession(lang=language) as session:
            result = session.run(code, libraries=libraries)
            return f"Exit code: {result.exit_code}\n{result.stdout}"
    except Exception as e:
        return f"Execution error: {str(e)}"

# Create structured tool
code_tool = StructuredTool.from_function(
    func=execute_code_with_libs,
    name="CodeExecutor",
    description="Execute code with optional library installation",
    args_schema=CodeExecutionInput
)

# Use in agent
agent = initialize_agent(
    [code_tool],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

response = agent.run(
    "Use numpy to create a 5x5 matrix of random numbers and calculate its determinant"
)
```

### Chain with Code Execution

```python
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

# Chain 1: Generate code
code_prompt = PromptTemplate(
    input_variables=["task"],
    template="""Write Python code to accomplish this task: {task}

Provide only the code, no explanations."""
)
code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")

# Chain 2: Execute code
def execute_code_chain(inputs: dict) -> dict:
    code = inputs["code"]
    with SandboxSession(lang="python") as session:
        result = session.run(code)
        return {"output": result.stdout, "error": result.stderr}

# Combine chains
from langchain.chains import TransformChain

execute_chain = TransformChain(
    input_variables=["code"],
    output_variables=["output", "error"],
    transform=execute_code_chain
)

overall_chain = SequentialChain(
    chains=[code_chain, execute_chain],
    input_variables=["task"],
    output_variables=["code", "output", "error"],
    verbose=True
)

# Run the chain
result = overall_chain({"task": "Generate the first 20 Fibonacci numbers"})
print(f"Generated code:\n{result['code']}")
print(f"\nOutput:\n{result['output']}")
```

### Memory and State Management

```python
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

class CodeExecutionMemory:
    """Custom memory for code execution history"""

    def __init__(self):
        self.executions = []
        self.session = None

    def start_session(self, **kwargs):
        self.session = SandboxSession(**kwargs)
        self.session.open()

    def execute(self, code: str, libraries: List[str] = None):
        if not self.session:
            self.start_session(lang="python")

        result = self.session.run(code, libraries)
        self.executions.append({
            "code": code,
            "output": result.stdout,
            "error": result.stderr,
            "exit_code": result.exit_code
        })
        return result

    def close_session(self):
        if self.session:
            self.session.close()
            self.session = None

# Use with conversation chain
memory = ConversationSummaryMemory(llm=llm)
code_memory = CodeExecutionMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Interactive code development
code_memory.start_session(lang="python")

# First execution
result1 = code_memory.execute("data = [1, 2, 3, 4, 5]")
result2 = code_memory.execute("print(f'Sum: {sum(data)}')")

code_memory.close_session()
```

## LangGraph Integration

### Stateful Code Execution Workflow

```python
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import Graph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
import operator

class CodeState(TypedDict):
    """State for code execution workflow"""
    task: str
    code: str
    output: str
    error: str
    iterations: int
    messages: Annotated[Sequence[str], operator.add]

# Define nodes
def generate_code(state: CodeState) -> CodeState:
    """Generate code based on task"""
    # Use LLM to generate code
    prompt = f"Write Python code for: {state['task']}"
    code = llm.predict(prompt)

    return {
        "code": code,
        "messages": [f"Generated code for task: {state['task']}"]
    }

def execute_code(state: CodeState) -> CodeState:
    """Execute the generated code"""
    with SandboxSession(lang="python") as session:
        result = session.run(state["code"])

        return {
            "output": result.stdout,
            "error": result.stderr,
            "messages": [f"Executed code with exit code: {result.exit_code}"]
        }

def check_output(state: CodeState) -> str:
    """Check if output is satisfactory"""
    if state["error"]:
        return "fix_code"
    elif state["iterations"] >= 3:
        return "end"
    else:
        return "end"

def fix_code(state: CodeState) -> CodeState:
    """Fix code based on error"""
    prompt = f"""Fix this Python code that has an error:

Code:
{state['code']}

Error:
{state['error']}
"""
    fixed_code = llm.predict(prompt)

    return {
        "code": fixed_code,
        "iterations": state["iterations"] + 1,
        "messages": [f"Attempted to fix code, iteration {state['iterations'] + 1}"]
    }

# Build graph
workflow = Graph()

# Add nodes
workflow.add_node("generate", generate_code)
workflow.add_node("execute", execute_code)
workflow.add_node("fix", fix_code)

# Add edges
workflow.add_edge("generate", "execute")
workflow.add_conditional_edges(
    "execute",
    check_output,
    {
        "fix_code": "fix",
        "end": END
    }
)
workflow.add_edge("fix", "execute")

# Set entry point
workflow.set_entry_point("generate")

# Compile
app = workflow.compile()

# Run workflow
initial_state = {
    "task": "Calculate the prime numbers between 1 and 50",
    "code": "",
    "output": "",
    "error": "",
    "iterations": 0,
    "messages": []
}

result = app.invoke(initial_state)
print(f"Final output: {result['output']}")
```

### Tool-Based Graph

```python
from langgraph.prebuilt import create_react_agent

# Create code execution tool
@tool
def run_python_code(code: str, libraries: List[str] = None) -> str:
    """Run Python code with optional libraries"""
    with SandboxSession(lang="python") as session:
        result = session.run(code, libraries=libraries)
        return result.stdout if result.exit_code == 0 else f"Error: {result.stderr}"

@tool
def run_data_analysis(
    csv_data: str,
    analysis_type: str = "summary"
) -> str:
    """Run data analysis on CSV data"""
    code = f"""
import pandas as pd
import io

data = '''{csv_data}'''
df = pd.read_csv(io.StringIO(data))

if "{analysis_type}" == "summary":
    print(df.describe())
elif "{analysis_type}" == "correlation":
    print(df.corr())
else:
    print(df.head())
"""

    with SandboxSession(lang="python") as session:
        result = session.run(code, libraries=["pandas"])
        return result.stdout

# Create agent
tools = [run_python_code, run_data_analysis]
agent = create_react_agent(llm, tools)

# Use agent
response = agent.invoke({
    "messages": [
        ("user", "Generate sample sales data and analyze it")
    ]
})
```

## LlamaIndex Integration

### Function Tool Integration

```python
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

def execute_code_with_context(
    code: str,
    language: str = "python",
    context: dict = None
) -> str:
    """
    Execute code with optional context variables.

    Args:
        code: Code to execute
        language: Programming language
        context: Dictionary of variables to inject

    Returns:
        Execution output
    """
    with SandboxSession(lang=language) as session:
        # Inject context if provided
        if context and language == "python":
            context_code = "\n".join([
                f"{key} = {repr(value)}"
                for key, value in context.items()
            ])
            full_code = f"{context_code}\n\n{code}"
        else:
            full_code = code

        result = session.run(full_code)
        return result.stdout

# Create LlamaIndex tool
code_tool = FunctionTool.from_defaults(
    fn=execute_code_with_context,
    name="code_executor",
    description="Execute code in a sandboxed environment with optional context"
)

# Create agent
llm = OpenAI(model="gpt-4", temperature=0)
agent = ReActAgent.from_tools(
    [code_tool],
    llm=llm,
    verbose=True
)

# Use agent
response = agent.chat(
    "Calculate the compound interest for $10,000 at 5% annual rate over 10 years"
)
print(response)
```

### Query Engine Integration

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.response_synthesizers import BaseSynthesizer

class CodeExecutionQueryEngine(CustomQueryEngine):
    """Query engine that executes code to answer questions"""

    def __init__(self, llm, security_policy=None):
        self.llm = llm
        self.security_policy = security_policy

    def custom_query(self, query_str: str) -> str:
        # Generate code to answer the query
        code_prompt = f"""
Write Python code to answer this question: {query_str}

The code should print the answer clearly.
"""
        code = self.llm.complete(code_prompt).text

        # Execute code safely
        with SandboxSession(
            lang="python",
            security_policy=self.security_policy
        ) as session:
            # Check if code is safe
            if self.security_policy:
                is_safe, violations = session.is_safe(code)
                if not is_safe:
                    return f"Code failed security check: {violations}"

            result = session.run(code)

            if result.exit_code != 0:
                return f"Execution error: {result.stderr}"

            return result.stdout

# Use the query engine
from llm_sandbox.security import get_security_policy

query_engine = CodeExecutionQueryEngine(
    llm=llm,
    security_policy=get_security_policy("data_science")
)

response = query_engine.query(
    "What is the correlation between height and weight in a generated dataset of 100 people?"
)
print(response)
```

## OpenAI Function Calling

### Direct Integration

```python
import openai
import json
from typing import List, Dict

def create_code_execution_function() -> Dict:
    """Create OpenAI function specification for code execution"""
    return {
        "name": "execute_code",
        "description": "Execute code in a secure sandbox environment",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to execute"
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "java", "cpp", "go", "ruby"],
                    "description": "Programming language"
                },
                "libraries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Libraries to install"
                }
            },
            "required": ["code", "language"]
        }
    }

def handle_function_call(function_call) -> str:
    """Handle the function call from OpenAI"""
    args = json.loads(function_call.arguments)

    with SandboxSession(
        lang=args["language"],
        verbose=False
    ) as session:
        result = session.run(
            args["code"],
            libraries=args.get("libraries")
        )

        return json.dumps({
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code
        })

# Use with OpenAI
client = openai.OpenAI()

messages = [
    {"role": "user", "content": "Calculate the first 10 prime numbers"}
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    functions=[create_code_execution_function()],
    function_call="auto"
)

# Handle function call if present
if response.choices[0].message.function_call:
    function_result = handle_function_call(
        response.choices[0].message.function_call
    )

    # Add function result to conversation
    messages.append(response.choices[0].message)
    messages.append({
        "role": "function",
        "name": "execute_code",
        "content": function_result
    })

    # Get final response
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    print(final_response.choices[0].message.content)
```

## Custom Framework Integration

### Generic Integration Pattern

```python
--8<-- "examples/async_example.py"
```

### Middleware Pattern

```python
from typing import Callable
import functools
import time

class CodeExecutionMiddleware:
    """Middleware for code execution with logging, caching, etc."""

    def __init__(self):
        self.cache = {}
        self.execution_log = []

    def with_logging(self, func: Callable) -> Callable:
        """Log all executions"""
        @functools.wraps(func)
        def wrapper(code: str, **kwargs):
            start_time = time.time()
            result = func(code, **kwargs)

            self.execution_log.append({
                "timestamp": time.time(),
                "code": code,
                "language": kwargs.get("language", "python"),
                "duration": time.time() - start_time,
                "success": result.get("success", False)
            })

            return result
        return wrapper

    def with_caching(self, func: Callable) -> Callable:
        """Cache execution results"""
        @functools.wraps(func)
        def wrapper(code: str, **kwargs):
            cache_key = f"{code}:{kwargs}"

            if cache_key in self.cache:
                return self.cache[cache_key]

            result = func(code, **kwargs)
            self.cache[cache_key] = result

            return result
        return wrapper

    def with_retry(self, max_attempts: int = 3) -> Callable:
        """Retry on failure"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(code: str, **kwargs):
                for attempt in range(max_attempts):
                    result = func(code, **kwargs)
                    if result.get("success"):
                        return result

                    if attempt < max_attempts - 1:
                        time.sleep(1)  # Wait before retry

                return result
            return wrapper
        return decorator

# Apply middleware
middleware = CodeExecutionMiddleware()
executor = SandboxCodeExecutor()

# Wrap with middleware
execute_with_features = middleware.with_logging(
    middleware.with_caching(
        middleware.with_retry(max_attempts=2)(
            executor.execute
        )
    )
)

# Use enhanced executor
result = execute_with_features(
    "print('Hello with middleware!')",
    language="python"
)
```

## Integration Best Practices

### 1. Error Handling

```python
class RobustCodeExecutor:
    """Robust code executor with comprehensive error handling"""

    def execute_safely(self, code: str, **kwargs):
        try:
            # Pre-execution validation
            if not code or not code.strip():
                return {"error": "Empty code provided"}

            # Security check
            with SandboxSession(**kwargs) as session:
                is_safe, violations = session.is_safe(code)
                if not is_safe:
                    return {
                        "error": "Security violation",
                        "violations": [
                            v.description for v in violations
                        ]
                    }

                # Execute
                result = session.run(code)

                # Post-execution validation
                if result.exit_code != 0:
                    return {
                        "error": "Execution failed",
                        "stderr": result.stderr,
                        "exit_code": result.exit_code
                    }

                return {
                    "success": True,
                    "output": result.stdout
                }

        except TimeoutError:
            return {"error": "Execution timeout"}
        except MemoryError:
            return {"error": "Memory limit exceeded"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
```

### 2. Resource Management

```python
from contextlib import contextmanager
import threading
import queue

class ResourceManagedExecutor:
    """Executor with resource management"""

    def __init__(self, max_concurrent=5):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.execution_queue = queue.Queue()

    @contextmanager
    def acquire_resources(self):
        """Acquire execution resources"""
        self.semaphore.acquire()
        try:
            yield
        finally:
            self.semaphore.release()

    def execute(self, code: str, **kwargs):
        """Execute with resource management"""
        with self.acquire_resources():
            # Configure resource limits
            runtime_configs = kwargs.get("runtime_configs", {})
            runtime_configs.update({
                "cpu_count": 1,
                "mem_limit": "256m",
            })
            kwargs["runtime_configs"] = runtime_configs

            with SandboxSession(**kwargs) as session:
                return session.run(code)
```

### 3. Monitoring and Metrics

```python
import time
from dataclasses import dataclass
from typing import List

@dataclass
class ExecutionMetrics:
    """Metrics for code execution"""
    timestamp: float
    duration: float
    language: str
    success: bool
    code_length: int
    memory_used: int = 0
    cpu_time: float = 0.0

class MonitoredExecutor:
    """Executor with monitoring capabilities"""

    def __init__(self):
        self.metrics: List[ExecutionMetrics] = []

    def execute_with_monitoring(self, code: str, **kwargs):
        """Execute code with monitoring"""
        start_time = time.time()

        try:
            with SandboxSession(**kwargs) as session:
                result = session.run(code)

                # Collect metrics
                metric = ExecutionMetrics(
                    timestamp=start_time,
                    duration=time.time() - start_time,
                    language=kwargs.get("lang", "python"),
                    success=result.exit_code == 0,
                    code_length=len(code)
                )

                self.metrics.append(metric)

                return result

        except Exception as e:
            # Record failure
            metric = ExecutionMetrics(
                timestamp=start_time,
                duration=time.time() - start_time,
                language=kwargs.get("lang", "python"),
                success=False,
                code_length=len(code)
            )
            self.metrics.append(metric)
            raise

    def get_statistics(self):
        """Get execution statistics"""
        if not self.metrics:
            return {}

        success_rate = sum(
            1 for m in self.metrics if m.success
        ) / len(self.metrics)

        avg_duration = sum(
            m.duration for m in self.metrics
        ) / len(self.metrics)

        return {
            "total_executions": len(self.metrics),
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "languages": list(set(m.language for m in self.metrics))
        }
```

## Next Steps

- See practical [Examples](examples.md)
- Learn about [Security Policies](security.md)
- Explore [Backend Options](backends.md)
- Read the [API Reference](api-reference.md)
