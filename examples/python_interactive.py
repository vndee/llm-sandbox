from llm_sandbox import PythonInteractiveSandboxSession


with PythonInteractiveSandboxSession(verbose=True, keep_template=True) as session:
    out = session.run_cell("print('Hello, World!')")