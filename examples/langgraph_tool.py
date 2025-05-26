import logging

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

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
    agent = create_react_agent(model="openai:gpt-4.1-nano", tools=[run_code])
    logger.info(
        "Agent: %s",
        agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": "Write python code to calculate Pi number by Monte Carlo method then run it.",
                }
            ]
        }),
    )
    logger.info(
        "Agent: %s",
        agent.invoke({
            "messages": [
                {"role": "user", "content": "Write python code to calculate the factorial of a number then run it."}
            ]
        }),
    )
    logger.info(
        "Agent: %s",
        agent.invoke({
            "messages": [
                {"role": "user", "content": "Write python code to calculate the Fibonacci sequence then run it."}
            ]
        }),
    )
    logger.info(
        "Agent: %s",
        agent.invoke({"messages": [{"role": "user", "content": "Calculate the sum of the first 10000 numbers."}]}),
    )
