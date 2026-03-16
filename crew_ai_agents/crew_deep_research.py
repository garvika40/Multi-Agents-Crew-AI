from typing import Type
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Task, Process, LLM
from crewai.tools import BaseTool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


class TavilyInput(BaseModel):
    query: str = Field(..., description="The research query to search for.")


class CustomTavilySearchTool(BaseTool):
    name: str = "Tavily Search"
    description: str = (
        "Performs web search using Tavily. Use for research queries and finding credible sources."
    )
    args_schema: Type[BaseModel] = TavilyInput

    def _run(self, query: str) -> str:
        results = TavilySearch(max_results=10).invoke(query)
        return str(results)


class ResearchCrew:
    def __init__(self):
        self.search_tool = CustomTavilySearchTool()
        self.llm = LLM(
            model="gpt-4o",
            temperature=0.7,
        )

    def research_agent(self) -> Agent:
        return Agent(
            role="Internet Researcher",
            goal="Find the most relevant and up-to-date information on a given topic. You MUST use the Tavily Search tool to search for information.",
            backstory="You are a skilled researcher with expertise in retrieving credible, real-time information from online sources. Always use the search tool to find facts.",
            tools=[self.search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def summarization_agent(self) -> Agent:
        return Agent(
            role="Content Summarizer",
            goal="Condense research findings into an easy-to-read summary.",
            backstory="You are an expert in breaking down complex information into clear, structured insights.",
            llm=self.llm,
            verbose=True,
        )

    def fact_checker_agent(self) -> Agent:
        return Agent(
            role="Fact-Checking Specialist",
            goal="Verify research findings and ensure factual accuracy. You MUST use the Tavily Search tool to verify claims.",
            backstory="You specialize in detecting misinformation and validating claims using credible sources. Always use the search tool to fact-check.",
            tools=[self.search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def research_task(self) -> Task:
        return Task(
            description="Your task is to search for information on the given topic: {topic}. Use the Tavily Search tool to find credible sources and extract key facts. Report back with specific findings and sources.",
            expected_output="A comprehensive summary of relevant research findings with URLs and source names.",
            agent=self.research_agent(),
            tools=[self.search_tool],
        )

    def summarization_task(self) -> Task:
        return Task(
            description="Summarize the research findings into clear, concise key points.",
            expected_output="A structured summary of the main findings.",
            agent=self.summarization_agent(),
        )

    def fact_checking_task(self) -> Task:
        return Task(
            description="Verify the accuracy of the summarized findings using the Tavily Search tool. Identify any inconsistencies and confirm key claims with evidence.",
            expected_output="A fact-check report confirming the accuracy of key claims with verification sources.",
            agent=self.fact_checker_agent(),
            tools=[self.search_tool],
        )

    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.research_agent(),
                self.summarization_agent(),
                self.fact_checker_agent(),
            ],
            tasks=[
                self.research_task(),
                self.summarization_task(),
                self.fact_checking_task(),
            ],
            process=Process.sequential,
        )
