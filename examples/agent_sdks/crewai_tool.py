# ruff: noqa: T201, INP001
"""LLM Sandbox as a tool for CrewAI.

    pip install 'llm-sandbox[docker]' crewai

Verified against crewai 1.15.10. The `@tool` decorator takes the tool name as
its argument; the docstring becomes the description the model sees.
"""

from _sandbox import run_python
from crewai import Agent, Crew, Task
from crewai.tools import tool


@tool("Execute Python")
def execute_python(code: str) -> str:
    """Execute Python code in a secure sandboxed container and return its stdout.

    Use this for calculations and data manipulation. Always print the result.
    """
    return run_python(code)


analyst = Agent(
    role="Data Analyst",
    goal="Answer quantitative questions by writing and running Python",
    backstory="You prefer computing an answer over estimating one.",
    tools=[execute_python],
)

task = Task(
    description="Compute the mean and standard deviation of the first 100 primes.",
    expected_output="The two numbers, with the code used to obtain them.",
    agent=analyst,
)

if __name__ == "__main__":
    print(Crew(agents=[analyst], tasks=[task]).kickoff())
