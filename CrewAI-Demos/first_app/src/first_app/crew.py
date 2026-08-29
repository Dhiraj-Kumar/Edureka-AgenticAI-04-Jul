from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class FirstApp():
    """FirstApp crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def transcript_cleaner(self) -> Agent:
        return Agent(
            config=self.agents_config['transcript_cleaner'], # type: ignore[index]
        )

    @agent
    def meeting_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['meeting_summarizer'], # type: ignore[index]
        )

    @agent
    def action_item_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['action_item_extractor'], # type: ignore[index]
        )

    @task
    def clean_transcript(self) -> Task:
        return Task(
            config=self.tasks_config['clean_transcript'],
        )

    @task
    def generate_summary(self) -> Task:
        return Task(
            config=self.tasks_config['generate_summary'],
        )
    @task
    def extract_action_items(self) -> Task:
        return Task(
            config=self.tasks_config['extract_action_items'],
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the FirstApp crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
