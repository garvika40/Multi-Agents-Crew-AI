# import packages
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


# Tool A: DuckDuckGo


@tool(parse_docstring=True)
def duckduckgo_search_tool(query: str) -> str:
    """
    Perform a quick web search using DuckDuckGo.

    This tool is useful for simple, factual queries or checking
    current events where deep analysis is not required.

    Args:
        query: The search query string.

    Returns:
        A text-based summary of the DuckDuckGo search results.
    """
    return DuckDuckGoSearchRun().run(query)


# Tool B: Tavily
@tool(parse_docstring=True)
def tavily_search_tool(
    query: str,
    max_results: int = 15,
    include_raw_content: bool = True,
    include_links: bool = True,
) -> str:
    """
    Perform an in-depth web search using Tavily.

    This tool is intended for complex or multi-hop research tasks,
    such as market analysis, technical deep dives, or report generation.

    Args:
        query: The research query to search for.
        max_results: Maximum number of search results to retrieve.
        include_raw_content: Whether to include raw page content.
        include_links: Whether to include source links in the results.

    Returns:
        A formatted and summarized string of high-quality search results.
    """
    search_results = TavilySearch(
        max_results=max_results,
        include_raw_content=include_raw_content,
        include_links=include_links,
    ).invoke(query)

    # summarized_results = process_search_results(search_results)
    # print(summarized_results)
    # return format_search_output(summarized_results)
    print(search_results)
    return search_results


# Tool C: Reflection / Thinking
@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """
    Record a strategic reflection on the research process.

    This tool is used to deliberately pause and evaluate research
    progress, identify gaps, and decide whether further searching
    is required or if a final answer can be produced.

    When to use:
    - After reviewing search results
    - Before deciding on next research steps
    - When evaluating completeness and quality of gathered information

    Args:
        reflection: A detailed reflection covering findings, gaps,
            quality of evidence, and next steps.

    Returns:
        A confirmation message indicating the reflection was recorded.
    """
    return f"Reflection recorded: {reflection}"


search_tools = [duckduckgo_search_tool, tavily_search_tool, think_tool]


# # response_search = tavily_search_tool("What is a sky??")
# import json

# with open("response.json", "w") as f:
#     json.dump(response_search, f, indent=4)
#     f.close()
