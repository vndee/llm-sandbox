# ruff: noqa: E501

# Reference: https://python.langchain.com/docs/how_to/custom_tools/

import logging

from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from llm_sandbox import SandboxSession

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@tool
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
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    prompt = hub.pull("hwchase17/openai-functions-agent")
    tools = [run_code]

    agent = create_tool_calling_agent(llm, tools, prompt)  # type: ignore[arg-type]
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # type: ignore[arg-type]
    output = agent_executor.invoke({
        "input": "Write python code to calculate Pi number by Monte Carlo method then run it."
    })
    logger.info("Agent: %s", output)

    output = agent_executor.invoke({"input": "Write python code to calculate the factorial of a number then run it."})
    logger.info("Agent: %s", output)

    output = agent_executor.invoke({"input": "Write python code to calculate the Fibonacci sequence then run it."})
    logger.info("Agent: %s", output)

    output = agent_executor.invoke({"input": "Calculate the sum of the first 10000 numbers."})
    logger.info("Agent: %s", output)
