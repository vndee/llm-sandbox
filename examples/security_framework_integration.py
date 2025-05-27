"""Integration examples for security policies with popular frameworks.

This module demonstrates how to integrate LLM Sandbox security policies
with various frameworks and use cases like LangChain, FastAPI, Flask, etc.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from llm_sandbox import SandboxSession
from llm_sandbox.security import (
    SecurityIssueSeverity,
    SecurityPattern,
    DangerousModule,
    SecurityPolicy,
)
from llm_sandbox.data import ConsoleOutput, ExecutionResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SecureCodeExecutor:
    """Secure code executor with comprehensive security features."""
    
    def __init__(self, security_policy: SecurityPolicy):
        self.security_policy = security_policy
        self.execution_history = []
        self.security_violations = []
    
    def execute_code(
        self, 
        code: str, 
        libraries: List[str] = None,
        timeout: int = 30,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Execute code with security checks and logging.
        
        Args:
            code: Code to execute
            libraries: Libraries to install
            timeout: Execution timeout in seconds
            user_id: User identifier for auditing
            
        Returns:
            Dictionary with execution results and security info
        """
        execution_id = f"exec_{datetime.now().isoformat()}_{hash(code) % 10000}"
        
        result = {
            'execution_id': execution_id,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'code_hash': hash(code),
            'security_check': None,
            'execution_result': None,
            'error': None,
            'libraries': libraries or [],
        }
        
        try:
            with SandboxSession(
                lang="python", 
                security_policy=self.security_policy,
                verbose=True
            ) as session:
                # Security check
                is_safe, violations = session.is_safe(code)
                result['security_check'] = {
                    'is_safe': is_safe,
                    'violations': [
                        {
                            'pattern': v.pattern,
                            'description': v.description,
                            'severity': v.severity.name
                        } for v in violations
                    ]
                }
                
                if not is_safe:
                    # Log security violation
                    violation_record = {
                        'execution_id': execution_id,
                        'user_id': user_id,
                        'timestamp': datetime.now().isoformat(),
                        'violations': result['security_check']['violations'],
                        'code_sample': code[:100] + '...' if len(code) > 100 else code
                    }
                    self.security_violations.append(violation_record)
                    
                    result['error'] = f"Security policy violation: {len(violations)} issues found"
                    logger.warning(f"Security violation blocked execution {execution_id}")
                else:
                    # Execute code
                    execution_result = session.run(code, libraries=libraries)
                    
                    if isinstance(execution_result, (ConsoleOutput, ExecutionResult)):
                        result['execution_result'] = {
                            'exit_code': execution_result.exit_code,
                            'stdout': execution_result.stdout,
                            'stderr': execution_result.stderr,
                        }
                        
                        # Add plot data if available
                        if hasattr(execution_result, 'plots') and execution_result.plots:
                            result['execution_result']['plots'] = [
                                {
                                    'format': plot.format.value,
                                    'size': len(plot.content_base64)
                                } for plot in execution_result.plots
                            ]
                    
                    logger.info(f"Code execution completed successfully: {execution_id}")
        
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Code execution failed: {execution_id} - {e}")
        
        # Store execution record
        self.execution_history.append(result)
        return result
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report."""
        return {
            'total_executions': len(self.execution_history),
            'total_violations': len(self.security_violations),
            'violation_rate': len(self.security_violations) / len(self.execution_history) if self.execution_history else 0,
            'recent_violations': self.security_violations[-10:],  # Last 10 violations
            'successful_executions': len([e for e in self.execution_history if e.get('execution_result')]),
        }


# FastAPI Integration Example
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from pydantic import BaseModel
    
    class CodeExecutionRequest(BaseModel):
        code: str
        libraries: Optional[List[str]] = None
        security_level: Optional[str] = "production"
    
    class CodeExecutionResponse(BaseModel):
        execution_id: str
        is_safe: bool
        result: Optional[Dict[str, Any]] = None
        violations: Optional[List[Dict[str, str]]] = None
        error: Optional[str] = None
    
    def create_fastapi_app() -> FastAPI:
        """Create FastAPI app with secure code execution endpoint."""
        app = FastAPI(title="Secure Code Execution API")
        
        # Different security policies for different endpoints
        security_policies = {
            "development": create_development_policy(),
            "production": create_production_policy(),
            "educational": create_educational_policy(),
        }
        
        executors = {
            name: SecureCodeExecutor(policy) 
            for name, policy in security_policies.items()
        }
        
        def create_development_policy() -> SecurityPolicy:
            return SecurityPolicy(
                safety_level=SecurityIssueSeverity.HIGH,
                patterns=[
                    SecurityPattern(
                        pattern=r"\bos\.system\s*\(",
                        description="System command execution",
                        severity=SecurityIssueSeverity.HIGH,
                    )
                ],
                dangerous_modules=[
                    DangerousModule(
                        name="os",
                        description="Operating system interface",
                        severity=SecurityIssueSeverity.HIGH,
                    )
                ]
            )
        
        def create_production_policy() -> SecurityPolicy:
            return SecurityPolicy(
                safety_level=SecurityIssueSeverity.LOW,
                patterns=[
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
                ],
                dangerous_modules=[
                    DangerousModule("os", "OS interface", SecurityIssueSeverity.HIGH),
                    DangerousModule("subprocess", "Subprocess", SecurityIssueSeverity.HIGH),
                    DangerousModule("socket", "Network", SecurityIssueSeverity.MEDIUM),
                ]
            )
        
        def create_educational_policy() -> SecurityPolicy:
            return SecurityPolicy(
                safety_level=SecurityIssueSeverity.MEDIUM,
                patterns=[
                    SecurityPattern(
                        pattern=r"\bos\.system\s*\(",
                        description="System command execution",
                        severity=SecurityIssueSeverity.HIGH,
                    )
                ],
                dangerous_modules=[
                    DangerousModule("os", "OS interface", SecurityIssueSeverity.HIGH),
                ]
            )
        
        @app.post("/execute", response_model=CodeExecutionResponse)
        async def execute_code(request: CodeExecutionRequest):
            """Execute code with security checks."""
            security_level = request.security_level or "production"
            
            if security_level not in executors:
                raise HTTPException(status_code=400, detail=f"Invalid security level: {security_level}")
            
            executor = executors[security_level]
            
            try:
                result = executor.execute_code(
                    code=request.code,
                    libraries=request.libraries,
                    user_id="api_user"  # In real app, get from auth
                )
                
                return CodeExecutionResponse(
                    execution_id=result['execution_id'],
                    is_safe=result['security_check']['is_safe'],
                    result=result.get('execution_result'),
                    violations=result['security_check']['violations'],
                    error=result.get('error')
                )
            
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/security-report")
        async def get_security_report(security_level: str = "production"):
            """Get security report for a specific security level."""
            if security_level not in executors:
                raise HTTPException(status_code=400, detail=f"Invalid security level: {security_level}")
            
            return executors[security_level].get_security_report()
        
        return app

except ImportError:
    logger.warning("FastAPI not available, skipping FastAPI integration example")
    
    def create_fastapi_app():
        return None


# Flask Integration Example
try:
    from flask import Flask, request, jsonify
    
    def create_flask_app() -> Flask:
        """Create Flask app with secure code execution."""
        app = Flask(__name__)
        
        # Security policy for Flask app
        policy = SecurityPolicy(
            safety_level=SecurityIssueSeverity.MEDIUM,
            patterns=[
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
            ],
            dangerous_modules=[
                DangerousModule("os", "OS interface", SecurityIssueSeverity.HIGH),
                DangerousModule("subprocess", "Subprocess", SecurityIssueSeverity.HIGH),
            ]
        )
        
        executor = SecureCodeExecutor(policy)
        
        @app.route('/execute', methods=['POST'])
        def execute_code():
            """Execute code endpoint."""
            data = request.get_json()
            
            if not data or 'code' not in data:
                return jsonify({'error': 'Code is required'}), 400
            
            try:
                result = executor.execute_code(
                    code=data['code'],
                    libraries=data.get('libraries', []),
                    user_id=request.remote_addr  # Simple user identification
                )
                return jsonify(result)
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @app.route('/security-report', methods=['GET'])
        def security_report():
            """Get security report."""
            return jsonify(executor.get_security_report())
        
        return app

except ImportError:
    logger.warning("Flask not available, skipping Flask integration example")
    
    def create_flask_app():
        return None


# LangChain Tool Integration
try:
    from langchain.tools import BaseTool
    from langchain.pydantic_v1 import BaseModel, Field
    
    class CodeExecutionInput(BaseModel):
        """Input for code execution tool."""
        code: str = Field(description="Python code to execute")
        libraries: List[str] = Field(default=[], description="Libraries to install")
    
    class SecureCodeExecutionTool(BaseTool):
        """LangChain tool for secure code execution."""
        name: str = "secure_code_executor"
        description: str = "Execute Python code in a secure sandbox environment with security checks"
        args_schema = CodeExecutionInput
        
        def __init__(self, security_policy: SecurityPolicy = None):
            super().__init__()
            self.security_policy = security_policy or self._create_default_policy()
            self.executor = SecureCodeExecutor(self.security_policy)
        
        def _create_default_policy(self) -> SecurityPolicy:
            """Create default security policy for LangChain integration."""
            return SecurityPolicy(
                safety_level=SecurityIssueSeverity.MEDIUM,
                patterns=[
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
                ],
                dangerous_modules=[
                    DangerousModule("os", "OS interface", SecurityIssueSeverity.HIGH),
                    DangerousModule("subprocess", "Subprocess", SecurityIssueSeverity.HIGH),
                    DangerousModule("socket", "Network", SecurityIssueSeverity.MEDIUM),
                ]
            )
        
        def _run(self, code: str, libraries: List[str] = None) -> str:
            """Execute code and return result."""
            try:
                result = self.executor.execute_code(
                    code=code,
                    libraries=libraries or [],
                    user_id="langchain_agent"
                )
                
                if result.get('error'):
                    return f"Error: {result['error']}"
                
                if not result['security_check']['is_safe']:
                    violations = result['security_check']['violations']
                    return f"Security violation: Code blocked due to {len(violations)} security issues"
                
                exec_result = result.get('execution_result', {})
                output = exec_result.get('stdout', '')
                
                if exec_result.get('stderr'):
                    output += f"\nErrors: {exec_result['stderr']}"
                
                return output or "Code executed successfully (no output)"
            
            except Exception as e:
                return f"Execution failed: {str(e)}"
        
        async def _arun(self, code: str, libraries: List[str] = None) -> str:
            """Async version of code execution."""
            # For now, just run synchronously
            return self._run(code, libraries)

except ImportError:
    logger.warning("LangChain not available, skipping LangChain integration example")
    
    class SecureCodeExecutionTool:
        def __init__(self, *args, **kwargs):
            pass


# Jupyter Notebook Integration Example
class JupyterSecurityExtension:
    """Security extension for Jupyter-like environments."""
    
    def __init__(self, security_policy: SecurityPolicy = None):
        self.security_policy = security_policy or self._create_jupyter_policy()
        self.executor = SecureCodeExecutor(self.security_policy)
        self.cell_history = []
    
    def _create_jupyter_policy(self) -> SecurityPolicy:
        """Create security policy suitable for Jupyter notebooks."""
        return SecurityPolicy(
            safety_level=SecurityIssueSeverity.MEDIUM,
            patterns=[
                SecurityPattern(
                    pattern=r"\bos\.system\s*\(",
                    description="System command execution",
                    severity=SecurityIssueSeverity.HIGH,
                ),
                SecurityPattern(
                    pattern=r"\b!\w+",  # Shell commands in Jupyter
                    description="Shell command execution",
                    severity=SecurityIssueSeverity.MEDIUM,
                ),
                SecurityPattern(
                    pattern=r"%%bash",
                    description="Bash cell magic",
                    severity=SecurityIssueSeverity.MEDIUM,
                ),
            ],
            dangerous_modules=[
                DangerousModule("os", "OS interface", SecurityIssueSeverity.HIGH),
                DangerousModule("subprocess", "Subprocess", SecurityIssueSeverity.HIGH),
            ]
        )
    
    def execute_cell(self, cell_code: str, cell_id: str = None) -> Dict[str, Any]:
        """Execute a Jupyter cell with security checks."""
        cell_id = cell_id or f"cell_{len(self.cell_history)}"
        
        result = self.executor.execute_code(
            code=cell_code,
            user_id=f"jupyter_user_{cell_id}"
        )
        
        # Add cell-specific information
        result['cell_id'] = cell_id
        result['cell_number'] = len(self.cell_history) + 1
        
        self.cell_history.append(result)
        return result
    
    def get_notebook_security_summary(self) -> Dict[str, Any]:
        """Get security summary for the entire notebook session."""
        violations = [cell for cell in self.cell_history 
                     if not cell['security_check']['is_safe']]
        
        return {
            'total_cells': len(self.cell_history),
            'cells_with_violations': len(violations),
            'violation_rate': len(violations) / len(self.cell_history) if self.cell_history else 0,
            'most_common_violations': self._get_common_violations(),
        }
    
    def _get_common_violations(self) -> List[Dict[str, Any]]:
        """Get most common security violations."""
        violation_counts = {}
        
        for cell in self.cell_history:
            for violation in cell['security_check']['violations']:
                desc = violation['description']
                violation_counts[desc] = violation_counts.get(desc, 0) + 1
        
        return [
            {'description': desc, 'count': count}
            for desc, count in sorted(violation_counts.items(), 
                                    key=lambda x: x[1], reverse=True)
        ]


def demo_framework_integrations():
    """Demonstrate various framework integrations."""
    logger.info("🔧 Framework Integration Demonstrations")
    logger.info("=" * 50)
    
    # Test SecureCodeExecutor
    logger.info("\n1. Testing SecureCodeExecutor:")
    policy = SecurityPolicy(
        safety_level=SecurityIssueSeverity.MEDIUM,
        patterns=[
            SecurityPattern(
                pattern=r"\bos\.system\s*\(",
                description="System command execution",
                severity=SecurityIssueSeverity.HIGH,
            )
        ],
        dangerous_modules=[
            DangerousModule("os", "OS interface", SecurityIssueSeverity.HIGH)
        ]
    )
    
    executor = SecureCodeExecutor(policy)
    
    test_codes = [
        "print('Hello, World!')",  # Safe
        "import os\nos.system('ls')",  # Should be blocked
        "import math\nprint(math.sqrt(16))",  # Safe
    ]
    
    for i, code in enumerate(test_codes, 1):
        result = executor.execute_code(code, user_id=f"demo_user_{i}")
        logger.info(f"   Test {i}: {'✅ EXECUTED' if result.get('execution_result') else '❌ BLOCKED'}")
        logger.info(f"     Code: {code[:30]}...")
        if result.get('error'):
            logger.info(f"     Error: {result['error']}")
    
    # Test Jupyter Integration
    logger.info("\n2. Testing Jupyter Integration:")
    jupyter_ext = JupyterSecurityExtension()
    
    jupyter_cells = [
        "import pandas as pd\ndf = pd.DataFrame({'A': [1, 2, 3]})\nprint(df)",
        "import os\nos.system('whoami')",  # Should be blocked
        "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.show()",
    ]
    
    for i, cell in enumerate(jupyter_cells, 1):
        result = jupyter_ext.execute_cell(cell, f"cell_{i}")
        logger.info(f"   Cell {i}: {'✅ EXECUTED' if result.get('execution_result') else '❌ BLOCKED'}")
        if result.get('error'):
            logger.info(f"     Error: {result['error']}")
    
    summary = jupyter_ext.get_notebook_security_summary()
    logger.info(f"   Notebook Summary: {summary['total_cells']} cells, {summary['cells_with_violations']} violations")
    
    # Test LangChain Tool
    logger.info("\n3. Testing LangChain Tool Integration:")
    try:
        tool = SecureCodeExecutionTool()
        result = tool._run("print('Hello from LangChain!')")
        logger.info(f"   LangChain Tool Result: {result[:50]}...")
    except Exception as e:
        logger.info(f"   LangChain Tool Error: {e}")
    
    logger.info("\n✅ Framework integration demonstrations completed!")


if __name__ == "__main__":
    logger.info("LLM Sandbox Security Framework Integration")
    logger.info("=========================================")
    
    try:
        demo_framework_integrations()
        
        # Show how to create and run the web apps
        logger.info("\n🌐 Web Application Examples:")
        logger.info("   FastAPI app: Run with `uvicorn app:app --reload`")
        logger.info("   Flask app: Run with `flask run`")
        logger.info("   See the code for complete implementation details.")
        
    except Exception as e:
        logger.error(f"Framework integration demo failed: {e}")
        raise
