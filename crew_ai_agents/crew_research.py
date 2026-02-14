from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import TavilySearchTool
from crewai import LLM
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


@CrewBase
class ResearchCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        self.search_tool = TavilySearchTool(max_results=10)
        self.llm = LLM(
            model="gemini-1.5-flash",
            max_tokens=4000,
        )

    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["research_agent"],
            tools=[self.search_tool],
            llm=self.llm,
        )

    @agent
    def summarization_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["summarization_agent"],
            llm=self.llm,
        )

    @agent
    def fact_checker_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["fact_checker_agent"],
            tools=[self.search_tool],
            llm=self.llm,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            tools=[self.search_tool],
            llm=self.llm,
        )

    @task
    def summarization_task(self) -> Task:
        return Task(config=self.tasks_config["summarization_task"], llm=self.llm)

    @task
    def fact_checking_task(self) -> Task:
        return Task(
            config=self.tasks_config["fact_checking_task"],
            tools=[self.search_tool],
            llm=self.llm,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
        )
