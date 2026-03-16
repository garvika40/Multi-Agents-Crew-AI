import os
import yaml
from typing import Type
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process, LLM
from crewai.flow.flow import Flow, listen, start
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from crewai.tools import BaseTool

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["FIRECRAWL_API_KEY"] = os.getenv("FIRECRAWL_API_KEY")

llm = LLM(model="gpt-4o")

MAX_SCRAPED_CONTEXT_CHARS = 6000

# --------------- LOAD CONFIG ------------------

with open("crew_ai_agents/config/linkdin_agents.yaml", "r", encoding="utf-8") as f:
    agents_config = yaml.safe_load(f)

with open("crew_ai_agents/config/linkdin_tasks.yaml", "r", encoding="utf-8") as f:
    tasks_config = yaml.safe_load(f)


# --------------- LOAD EXAMPLE CONTENT ------------------


def _list_available_style_files() -> list[str]:
    """List all .txt files in assets folder for style selection."""
    try:
        import glob

        files = glob.glob("crew_ai_agents/assets/*.txt")
        return [os.path.basename(f).replace(".txt", "") for f in files]
    except Exception:
        return []


def _load_style_from_file(style_name: str) -> str:
    """Load style notes from a specific file in assets."""
    if not style_name:
        return ""
    try:
        filepath = f"crew_ai_agents/assets/{style_name}.txt"
        with open(filepath, "r", encoding="utf-8") as style_file:
            return style_file.read()
    except FileNotFoundError:
        return ""


def _load_example_linkedin_content() -> str:
    """Load example LinkedIn posts from assets for writer reference."""
    return _load_style_from_file("example_linkedin")


EXAMPLE_LINKEDIN_CONTENT = _load_example_linkedin_content()


# --------------- TAVILY TOOL ------------------


class TavilyInput(BaseModel):
    query: str = Field(..., description="The research query to search for.")


class TavilySearchTool(BaseTool):
    name: str = "Tavily Search"
    description: str = (
        "Performs in-depth web search. Use for research queries, market analysis, and finding credible sources."
    )
    args_schema: Type[BaseModel] = TavilyInput

    def _run(self, query: str) -> str:
        results = TavilySearch(max_results=10).invoke(query)
        return str(results)


# --------------- FIRECRAWL TOOL ------------------


class FirecrawlInput(BaseModel):
    url: str = Field(..., description="The full URL of the page to scrape.")


class FirecrawlScrapeTool(BaseTool):
    name: str = "Firecrawl Scrape"
    description: str = (
        "Scrapes the full content of a URL and returns it as markdown. Use when you have a specific source URL to extract content from."
    )
    args_schema: Type[BaseModel] = FirecrawlInput

    def _run(self, url: str) -> str:
        app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
        try:
            result = app.scrape(url, formats=["markdown", "html"])
            markdown = getattr(result, "markdown", None) or result.get("markdown", "")
            return markdown or getattr(result, "text", "No content extracted.")
        except Exception as e:
            return f"Scrape failed for {url}: {str(e)}"


# --------------- INSTANTIATE ------------------

search_tools = [TavilySearchTool(), FirecrawlScrapeTool()]

# --------------- OUTPUT MODELS ------------------


class PostDraft(BaseModel):
    content: str


class CriticReport(BaseModel):
    status: str
    violations: list


class VisualBrief(BaseModel):
    visual_type: str
    brief: dict


class PipelineState(BaseModel):
    topic: str = ""
    source_url: str = ""
    scraped_context: str = ""
    draft_content: str = ""
    critic_status: str = ""
    visual_brief: dict = {}
    style_notes: str = ""  # Custom style notes provided by user


class ResearchSource(BaseModel):
    sub_question: str
    url: str
    institution: str
    extract: str  # what the researcher pulled out
    full_content: str  # full markdown from firecrawl/tavily
    type: str  # "data" or "analysis"


class ResearchOutput(BaseModel):
    sources: list[ResearchSource]


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars]


# --------------- DEFINE CREW AGENT AND TASK ------------------


topic_analyst = Agent(
    config=agents_config["topic_analyst"],
    llm=llm,
    verbose=True,
)

researcher = Agent(
    config=agents_config["researcher"],
    tools=search_tools,  # Tavily for search, Firecrawl for deep URL scraping
    llm=llm,
    output_pydantic=ResearchOutput,  # <-- add this
    verbose=True,
)

editorial_filter = Agent(
    config=agents_config["editorial_filter"],
    llm=llm,
    verbose=True,
)

writer = Agent(
    config=agents_config["writer"],
    llm=llm,
    verbose=True,
)

critic = Agent(
    config=agents_config["critic"],
    llm=llm,
    verbose=True,
)


# --------------- DEFINE TASKS ------------------

sharpen_topic_task = Task(
    config=tasks_config["sharpen_topic_task"],
    agent=topic_analyst,
)

research_task = Task(
    config=tasks_config["research_task"],
    agent=researcher,
    context=[sharpen_topic_task],
)

filter_task = Task(
    config=tasks_config["filter_task"],
    agent=editorial_filter,
    context=[sharpen_topic_task, research_task],
)

write_task = Task(
    config=tasks_config["write_task"],
    agent=writer,
    context=[sharpen_topic_task, filter_task],
    output_pydantic=PostDraft,
    description=(
        tasks_config["write_task"].get("description", "")
        + (
            "\n\nSTYLE REFERENCE:\nRefer to the provided examples in your agent backstory for the tone, structure, and formatting style. Match the professional, direct tone and multi-paragraph structure used in those examples."
            if EXAMPLE_LINKEDIN_CONTENT
            else ""
        )
    ),
)

critic_task = Task(
    config=tasks_config["critic_task"],
    agent=critic,
    context=[write_task],
    output_pydantic=CriticReport,
)


# --------------- FLOW ------------------


class ContentPipeline(Flow[PipelineState]):

    @start()
    def receive_input(self):
        """
        Entry point. If a source_url is provided, Firecrawl scrapes it
        and stores the markdown in state as additional context.
        The scraped content is appended to the topic string passed into
        the crew so the Topic Analyst and Researcher both benefit from it.
        If no URL is provided the pipeline runs on the raw topic string alone.
        """
        print(f"\n# Pipeline started for topic: {self.state.topic}\n")

        if self.state.source_url:
            print(f"# Scraping source URL for context: {self.state.source_url}\n")
            app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

            try:
                scrape_results = app.scrape(
                    self.state.source_url, formats=["markdown", "html"]
                )
                markdown_text = getattr(scrape_results, "markdown", None)
                if markdown_text is None:
                    markdown_text = scrape_results.get("markdown", "")
                if not markdown_text:
                    markdown_text = getattr(scrape_results, "text", "")

                scraped_content = markdown_text or ""
                self.state.scraped_context = _truncate_text(
                    scraped_content,
                    MAX_SCRAPED_CONTEXT_CHARS,
                )
                print(
                    f"# Scraped {len(scraped_content)} characters; using {len(self.state.scraped_context)} characters as context.\n"
                )

            except Exception as e:
                print(
                    f"# Firecrawl scrape failed: {e}. Continuing without URL context.\n"
                )
                self.state.scraped_context = ""

        return self.state

    @listen(receive_input)
    def run_content_crew(self):
        print("# Running content pipeline crew...\n")

        # Append scraped content to topic input if available
        topic_input = self.state.topic
        if self.state.scraped_context:
            topic_input = (
                f"{self.state.topic}\n\n"
                f"Additional context scraped from source URL:\n"
                f"{self.state.scraped_context}"
            )

        # Build write task with custom or default style notes
        style_description = ""
        if self.state.style_notes:
            style_description = f"\n\nSTYLE REFERENCE:\n{self.state.style_notes}"
        elif EXAMPLE_LINKEDIN_CONTENT:
            style_description = "\n\nSTYLE REFERENCE:\nRefer to the provided examples in your agent backstory for the tone, structure, and formatting style. Match the professional, direct tone and multi-paragraph structure used in those examples."

        custom_write_task = Task(
            config=tasks_config["write_task"],
            agent=writer,
            context=[sharpen_topic_task, filter_task],
            output_pydantic=PostDraft,
            description=tasks_config["write_task"].get("description", "")
            + style_description,
        )

        content_crew = Crew(
            agents=[
                topic_analyst,
                researcher,
                editorial_filter,
                writer,
                critic,
            ],
            tasks=[
                sharpen_topic_task,
                research_task,
                filter_task,
                custom_write_task,
                critic_task,
            ],
            process=Process.sequential,
            memory=False,
            verbose=True,
        )

        result = content_crew.kickoff(inputs={"topic": topic_input})
        return result

    @listen(run_content_crew)
    def handle_output(self, result):
        print("\n# Pipeline complete.\n")

        for task_output in result.tasks_output:

            if hasattr(task_output, "pydantic") and isinstance(
                task_output.pydantic, PostDraft
            ):
                print("---- FINAL DRAFT ----")
                print(task_output.pydantic.content)
                self.state.draft_content = task_output.pydantic.content

            if hasattr(task_output, "pydantic") and isinstance(
                task_output.pydantic, CriticReport
            ):
                print(f"\n---- CRITIC STATUS: {task_output.pydantic.status} ----")
                self.state.critic_status = task_output.pydantic.status
                if task_output.pydantic.violations:
                    for v in task_output.pydantic.violations:
                        print(f"  Violation: {v}")

            if hasattr(task_output, "pydantic") and isinstance(
                task_output.pydantic, VisualBrief
            ):
                print(f"\n---- VISUAL BRIEF: {task_output.pydantic.visual_type} ----")
                print(task_output.pydantic.brief)
                self.state.visual_brief = task_output.pydantic.brief

        return self.state


# --------------- ENTRY POINT ------------------

if __name__ == "__main__":
    topic = input("Enter your topic: ").strip()
    source_url = input(
        "Enter a source URL to scrape for context (or press Enter to skip): "
    ).strip()
