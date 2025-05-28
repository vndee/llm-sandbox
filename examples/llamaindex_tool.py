# ruff: noqa: E501

# Reference: https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/

import logging

import nest_asyncio
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

from llm_sandbox import SandboxSession

nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_code(lang: str, code: str, libraries: list | None = None) -> str:
    """Run code in a sandboxed environment.

    :param lang: The language of the code, must be one of ['python', 'java', 'javascript', 'cpp', 'go', 'ruby'].
    :param code: The code to run.
    :param libraries: The libraries to use, it is optional.
    :return: The output of the code.
    """
    with SandboxSession(lang=lang, verbose=False) as session:
        return session.run(code, libraries).stdout


if __name__ == "__main__":
    llm = OpenAI(model="gpt-4.1-nano", temperature=0)
    code_execution_tool = FunctionTool.from_defaults(fn=run_code)

    agent_worker = FunctionCallingAgentWorker.from_tools(
        [code_execution_tool],
        llm=llm,
        verbose=True,
        allow_parallel_tool_calls=False,
    )
    agent = agent_worker.as_agent()

    response = agent.chat("Write python code to calculate Pi number by Monte Carlo method then run it.")
    logger.info(response)

    response = agent.chat("Write python code to calculate the factorial of a number then run it.")
    logger.info(response)

    response = agent.chat("Write python code to calculate the Fibonacci sequence then run it.")
    logger.info(response)

    response = agent.chat("Calculate the sum of the first 10000 numbers.")
    logger.info(response)
