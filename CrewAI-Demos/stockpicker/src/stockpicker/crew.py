from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from pydantic import BaseModel, Field
from typing import List
from crewai_tools import SerperDevTool
from langchain_community.tools import DuckDuckGoSearchRun

class TrendingCompany(BaseModel):
    """ A company that is in the new and attracting attention """
    name: str = Field(description="Company Name")
    ticker: str = Field(description="Stock ticker symbol")
    reason: str = Field(description="Reason this company is trending in the news")

class TrendingCompanyList(BaseModel):
    """List of multiple trending companies that are in the news"""
    companies: List[TrendingCompany]=Field(description="List of companies trending in the news")

class TrendingCompanyResearch(BaseModel):
    """Detailed research on a company"""
    name: str = Field(description="Company Name")
    market_position: str = Field(description="Current market position and competitive analysis")
    future_outlook: str = Field(description="Future outlook and growth prospects")
    investment_potential: str = Field(description="Investment potential and suitability for investment")

class TrendingCompanyResearchList(BaseModel):
    """A list of detailed research on all the companies"""
    research_list: List[TrendingCompanyResearch] = Field(description="Comprehensive research on all trending companies")

# web_search = DuckDuckGoSearchRun()

@CrewBase
class Stockpicker():
    """Stockpicker crew"""

    agents: list[BaseAgent]
    tasks: list[Task]


    @agent
    def trending_company_finder(self) -> Agent:
        return Agent(
            config=self.agents_config['trending_company_finder'],
            verbose=True,
            # tools=[web_search]
            tools=[SerperDevTool()]
        )

    @agent
    def financial_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_researcher'],
            verbose=True,
            tools=[SerperDevTool()]
        )

    @agent
    def stock_picker(self) -> Agent:
        return Agent(
            config=self.agents_config['stock_picker'],
            verbose=True,
        )

    @task
    def find_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config['find_trending_companies'],
            output_pydantic=TrendingCompanyList
        )

    @task
    def research_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config['research_trending_companies'],
            output_pydantic=TrendingCompanyResearchList
        )
        
    @task
    def pick_best_company(self) -> Task:
        return Task(
            config=self.tasks_config['pick_best_company']
        )

    

    @crew
    def crew(self) -> Crew:
        """Creates the Stockpicker crew"""

        manager = Agent(
            config = self.agents_config['manager'],
            verbose=True
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=manager,
            verbose=True,
        )
