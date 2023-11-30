from typing import Any, List, Tuple, Union

from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain.utilities import SerpAPIWrapper

import os

os.environ["SERPAPI_API_KEY"] = "<your_serp_api_key>"

search = SerpAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="In case you need to answer questions about current events",
        return_direct=True
    )
]


class FakeAgent(BaseSingleActionAgent):

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        return AgentAction(tool="Search", tool_input=kwargs["input"], log="")

    async def aplan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any) -> Union[
        AgentAction, AgentFinish]:
        return AgentAction(
            tool="Search",
            tool_input=kwargs["input"],
            log=""
        )


agent = FakeAgent()
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

agent_executor.run("What is the current population of Uganda?")