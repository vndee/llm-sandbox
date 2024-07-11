# Reference: https://python.langchain.com/v0.1/docs/modules/agents/quick_start/
# Reference: https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/

from typing import Optional, List
from llm_sandbox import SandboxSession

from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgentWorker


import nest_asyncio

nest_asyncio.apply()


def run_code(lang: str, code: str, libraries: Optional[List] = None) -> str:
    """
    Run code in a sandboxed environment.
    :param lang: The language of the code, must be one of ['python', 'java', 'javascript', 'cpp', 'go', 'ruby'].
    :param code: The code to run.
    :param libraries: The libraries to use, it is optional.
    :return: The output of the code.
    """
    with SandboxSession(lang=lang, verbose=False) as session:  # type: ignore[attr-defined]
        return session.run(code, libraries).text


if __name__ == "__main__":
    llm = OpenAI(model="gpt-4o", temperature=0)
    code_execution_tool = FunctionTool.from_defaults(fn=run_code)

    agent_worker = FunctionCallingAgentWorker.from_tools(
        [code_execution_tool],
        llm=llm,
        verbose=True,
        allow_parallel_tool_calls=False,
    )
    agent = agent_worker.as_agent()

    response = agent.chat(
        "Write python code to calculate Pi number by Monte Carlo method then run it."
    )
    print(response)

    response = agent.chat(
        "Write python code to calculate the factorial of a number then run it."
    )
    print(response)

    response = agent.chat(
        "Write python code to calculate the Fibonacci sequence then run it."
    )
    print(response)

    response = agent.chat("Calculate the sum of the first 10000 numbers.")
    print(response)
