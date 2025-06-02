from crewai import Agent, Task, Crew, Process
from crewai.project import agent, task, crew
from crewai.project import CrewBase
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun

# DuckDuckGo 검색 도구
# websearch.py 또는 crew.py 안에

from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun

class DuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGo Web Search"
    description: str = "Search the web for public information."

    def _run(self, query: str) -> str:
        return DuckDuckGoSearchRun().run(query)


@CrewBase
class HiCrewBase: 
    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def policy_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['policy_researcher'],
            tools=[DuckDuckGoTool()],
            verbose=True
        )

    @agent
    def issue_sorter(self) -> Agent:
        return Agent(
            config=self.agents_config['issue_sorter'],
            verbose=True
        )

    @agent
    def neutral_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['neutral_analyzer'],
            verbose=True
        )

    @agent
    def sentiment_critic(self) -> Agent:
        return Agent(
            config=self.agents_config['sentiment_critic'],
            verbose=True
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['report_writer'],
            verbose=True
        )

    @task
    def gather_policy_data(self) -> Task:
        return Task(config=self.tasks_config['gather_policy_data'])

    @task
    def sort_by_issue(self) -> Task:
        return Task(config=self.tasks_config['sort_by_issue'])

    @task
    def compare_policies(self) -> Task:
        return Task(config=self.tasks_config['compare_policies'])

    @task
    def analyze_sentiment(self) -> Task:
        return Task(config=self.tasks_config['analyze_sentiment'])

    @task
    def generate_report(self) -> Task:
        return Task(
            config=self.tasks_config['generate_report'],
            output_file="final_report.md"
        )

# 🔄 순차 구조
class HiCrewaiSequential(HiCrewBase):
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )

# 🧠 계층 구조
class HiCrewaiHierarchy(HiCrewBase):
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            verbose=True
        )
