from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter

server_params = StdioServerParameters(
    command="uv",
    args= [
        "run",
        "--with",
        "mcp[cli]==2.0.0",
        "mcp",
        "run",
        r"F:\MyBatches\Edureka-AgenticAI-04-Jul\mcp-demos\first-demo\server.py"
    ]
)

mcp_adapter = MCPServerAdapter(server_params)

@CrewBase
class McpCrew():
    """McpCrew crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            tools=mcp_adapter.tools,
            verbose=True
        )


    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )


    @crew
    def crew(self) -> Crew:
        """Creates the McpCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
