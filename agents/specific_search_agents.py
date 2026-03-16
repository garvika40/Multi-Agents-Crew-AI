from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, List
from langchain.chat_models import init_chat_model
from tools.research_tools import think_tool, tavily_search_tool, duckduckgo_search_tool
from pydantic import BaseModel,Field
from langgraph.graph import StateGraph, START, END
from typing import Optional
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    filter_messages,
    AIMessage
)
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

import os, sys
#########################-----------API AND MODEL CONFIG--------------#######################################

api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     print("ERROR: OPENAI_API_KEY not set!")
#     sys.exit(1)
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_PROJECT"] = "research-agent"



tools = [think_tool, tavily_search_tool, duckduckgo_search_tool]
tools_by_name = {tool.name: tool for tool in tools}

# Initialize models
compress_model = init_chat_model(
    model="openai:gpt-4.1", max_tokens=32000
)
summarization_model = init_chat_model(model="openai:gpt-4.1-mini")

model = init_chat_model(model="openai:gpt-4o-mini", )
model_with_tools = model.bind_tools(tools=[duckduckgo_search_tool])
model_with_advanced_tool = model.bind_tools(tools=[tavily_search_tool])

#---------UTIL ----------

def tool_node_presales(response):
    """Execute all tool calls from the previous LLM response.
    Executes all tool calls from the previous LLM responses.
    Returns updated state with tool execution results.
    """

    tool_calls = response.tool_calls
    observations = []

    for tool_call in tool_calls:

        tool_name = tool_call["name"]
        tool_call_args = tool_call["args"]

        result = tools_by_name[tool_name].invoke(tool_call_args)
        observations.append(result)

    tool_outputs = [
        ToolMessage(content=o, name=tc["name"], tool_call_id=tc["id"])
        for o, tc in zip(observations, tool_calls)
    ]
    return tool_outputs


# ---- STATE ----
class ResearchState(TypedDict):
    researcher_messages: str
    company: str
    company_description: str
    jobs_data: Optional[Dict]
    revenue_data: Optional[Dict]
    news: Optional[List[str]]
    mission_vision: Optional[str]
    ai_research: Optional[Dict]
    recommended_lens: Optional[List[str]]
    sales_assets: Optional[List[str]]
    final_report: Optional[str]


class JobPosting(BaseModel):
    job_title: str = Field(description="Title of the job posting")
    job_description: str = Field(description="Short summary of the role")
    job_link: str = Field(description="Direct URL to the job posting")
    
class JobsAtCompany(BaseModel):
    jobs_at_company: List[JobPosting]
    
class StructuredResponse(BaseModel):
    company_name: str
    company_description: str = Field(
        description="Short company description clarifying which company is being referenced based on name."
    )


class RevenueSource(BaseModel):
    url: str
    title: str 
    summary: str


class RevenueIntelligence(BaseModel):
    revenue_summary: str
    latest_revenue_summary: str
    industry: str
    estimated_it_spend_summary: str
    confidence_level: str
    listed_company: str
    key_financial_sources: List[RevenueSource]



class NewsItem(BaseModel):
    headline: str
    summary: str
    source: str
    date: Optional[str]
    category: str = Field(
        description="Transformation, Financial, Technology, Workforce/Leadership_Changes, Regulatory, Risk, ESG, Aquisition"
    )
    presales_implication: str


class CompanyNewsIntel(BaseModel):
    company: str
    key_news: List[NewsItem]
    overall_signal: str


class ExecutiveInfo(BaseModel):
    name: str
    role: str



class CompanyWebsiteIntel(BaseModel):
    company_name: str
    about_company: str = Field(description="A very detailed about info of the company")
    mission_statement: str
    vision_statement: str
    purpose_statement: str
    areas_of_operations: List[str]
    industries_served: List[str]
    geographic_presence: List[str]
    key_products_services: List[str]
    leadership_team: List[ExecutiveInfo] = Field(description="Top 10 leaders including the IT and ERP")
    strategic_priorities: List[str]
    corporate_values: List[str]
    source_pages: List[str]


class AIInitiative(BaseModel):
    title: str
    link: str
    summary: str

class AIOutlookIntel(BaseModel):
    company: str
    ai_related_news: List[AIInitiative]
    ai_partnerships: List[AIInitiative]
    overall_take_on_ai: str



# ---- NODES ----

def resolve_company(state):
    user_query = model.with_structured_output(StructuredResponse).invoke(state["researcher_messages"])
    print(user_query)
    return {"company": user_query.company_name, "company_description": user_query.company_description}

from langchain_core.messages import SystemMessage, HumanMessage

def jobs_research(state):
    messages = [
        SystemMessage(
            content="""
    You analyze job market signals related to Oracle Fusion.

    Rules:
    - Only include jobs explicitly mentioning Oracle Fusion ERP.
    - If none are found, return:
    - fusion_jobs_found = false
    - jobs_at_company = empty list
    - Never fabricate job postings.
    """
        ),
        HumanMessage(
            content=f"Find Oracle Fusion ERP job openings for {state['company']}"
        )
    ]
    # Step 1 — Model decides tool usage
    response = model_with_advanced_tool.invoke(messages)

    # Step 2 — Execute tool node
    tool_output = tool_node_presales(response)

    # Step 3 — Give tool output BACK to model
    synthesis_messages = messages + [response] + tool_output
    structured = model.with_structured_output(JobsAtCompany).invoke(synthesis_messages)
    print(structured)


    return {"jobs_data": structured}


def revenue_research(state):

    messages = [
        SystemMessage(
            content="""
            You are a financial research analyst.

            Rules:
            - Prefer official financial disclosures.
            - If unavailable, use industry benchmarks.
            - Provide revenue estimate with confidence level.
            - Estimate IT budget based on industry norms. Typically 5% of Revenue.
            - Never fabricate financial figures.
            """
                    ),
                    HumanMessage(
                        content=f"""
            Research revenue intelligence for:

            Company: {state['company']}

            Return:
            Output in structured format 
            """
        )
    ]

    # Step 1 — Model decides tool usage
    response = model_with_advanced_tool.invoke(messages)

    # Step 2 — Execute tool node
    tool_output = tool_node_presales(response)

    # Step 3 — Give tool output BACK to model
    synthesis_messages = messages + [response] + tool_output
    structured = model.with_structured_output(RevenueIntelligence).invoke(synthesis_messages)
    print(structured)
    return {"revenue_data":structured }


def news_research(state):

    messages = [
        SystemMessage(
            content="""
                You are conducting presales intelligence research for mentined company.

                Find recent news specifically relevant for enterprise technology sales.

                Focus ONLY on news that indicates:

                - Digital transformation or technology initiatives
                - Financial performance or funding events
                - Mergers, acquisitions, partnerships, or divestitures
                - Workforce changes (hiring, layoffs, leadership changes)
                - Regulatory, compliance, cybersecurity, or risk events
                - AI adoption, automation, or back-office modernization
                - Sustainability or CSR initiatives affecting operations

                For each relevant news item provide strcutured output

                Rules:

                - Ignore stock price chatter unless it signals business change.
                - Ignore purely promotional PR unless it affects strategy.
                - Do not fabricate news.
                - Prioritize credible sources and recent developments."""
        ),
        HumanMessage(
            content=f"""
            News Research intelligence for:

            Company: {state['company']}

            Return:
            Output in structured format 
            """)
    ]

    # Step 1 — Model decides tool usage
    response = model_with_advanced_tool.invoke(messages)

    # Step 2 — Execute tool node
    tool_output = tool_node_presales(response)

    # Step 3 — Give tool output BACK to model
    synthesis_messages = messages + [response] + tool_output
    structured = model.with_structured_output(CompanyNewsIntel).invoke(synthesis_messages)
    print(structured)
    return {"news":structured }



def about_company_research(state):

    messages = [
        SystemMessage(
            content="""
                You are conducting corporate intelligence research for presales preparation.
                Company: {company}

                Extract structured information strictly from the company’s official website content.

                Focus on identifying:

                1. Company overview (clear summary of what the company does)
                2. Mission statement
                3. Vision statement
                4. Purpose statement (if distinct from mission/vision)
                5. Areas of operation (industries, markets, regions)
                6. Geographic presence (countries or regions mentioned)
                7. Key products or services
                8. Leadership team 
                9. Stated strategic priorities or transformation themes
                10. Corporate values or guiding principles

                Rules:
                - Only use information explicitly found on official company pages.
                - Do not infer or fabricate details.
                - If a section is not available, return null.
                - Do not include marketing fluff beyond what is clearly stated.
                - Keep descriptions factual and concise.

                Return the output in structured format matching the predefined schema"""
        ),
        HumanMessage(
            content=f"""
            About Company Research intelligence for:

            Company: {state['company']}

            Return:
            Output in structured format 
            """)
    ]

    # Step 1 — Model decides tool usage
    response = model_with_advanced_tool.invoke(messages)

    # Step 2 — Execute tool node
    tool_output = tool_node_presales(response)

    # Step 3 — Give tool output BACK to model
    synthesis_messages = messages + [response] + tool_output
    structured = model.with_structured_output(CompanyWebsiteIntel).invoke(synthesis_messages)
    print(structured)
    return {"mission_vision":structured }



def ai_research(state):

    messages = [
        SystemMessage(
            content="""
                You are conducting presales intelligence research focused on artificial intelligence adoption.

                Assess the company's AI outlook based on publicly available information such as:
                - AI related news 
                - AI related leadership statements
                - AI partnerships
                - AI product announcements
                - AI related hiring signals
                - AI technology investments

                Rules:
                - Base conclusions only on explicit signals.
                - If evidence is weak or absent, state low confidence.
                - Do not fabricate initiatives.
                - Prioritize strategic signals over marketing claims.

                Return structured output matching the predefined schema."""
        ),
        HumanMessage(
            content=f"""
            AI intelligence for:

            Company: {state['company']}

            Return:
            Output in structured format 
            """)
    ]

    # Step 1 — Model decides tool usage
    response = model_with_advanced_tool.invoke(messages)

    # Step 2 — Execute tool node
    tool_output = tool_node_presales(response)

    # Step 3 — Give tool output BACK to model
    synthesis_messages = messages + [response] + tool_output
    structured = model.with_structured_output(AIOutlookIntel).invoke(synthesis_messages)
    print(structured)
    return {"ai_research":structured }



def generate_html_report(state):

    # -----------------------------
    # Convert Pydantic models to dict (if present)
    # -----------------------------
    context = {
        "company": state.get("company"),
        "company_description": state.get("company_description"),
        "jobs_data": state.get("jobs_data").dict() if state.get("jobs_data") else None,
        "revenue_data": state.get("revenue_data").dict() if state.get("revenue_data") else None,
        "news": state.get("news").dict() if state.get("news") else None,
        "mission_vision": state.get("mission_vision").dict() if state.get("mission_vision") else None,
        "ai_research": state.get("ai_research").dict() if state.get("ai_research") else None,
    }

    # -----------------------------
    # Setup Jinja environment
    # -----------------------------
    template_dir = Path("templates")
    template_dir.mkdir(exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=True
    )

    template = env.get_template("presales_report.html.jinja")

    # -----------------------------
    # Render HTML
    # -----------------------------
    rendered_html = template.render(**context)

    # -----------------------------
    # Save file
    # -----------------------------
    output_path = Path("presales_report.html")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    # -----------------------------
    # Return updated state
    # -----------------------------
    return {
        "final_report": str(output_path)
    }
    
    
    
    
agent_builder = StateGraph(ResearchState)

agent_builder.add_node("resolve_company", resolve_company)
agent_builder.add_node("jobs_research", jobs_research)
agent_builder.add_node("revenue_research", revenue_research)
agent_builder.add_node("news_research", news_research)
agent_builder.add_node("about_company_research", about_company_research)
agent_builder.add_node("ai_research", ai_research)
agent_builder.add_node("generate_report", generate_html_report)

agent_builder.add_edge(START, "resolve_company")

agent_builder.add_edge("resolve_company", "jobs_research")
agent_builder.add_edge("resolve_company", "revenue_research")
agent_builder.add_edge("resolve_company", "news_research")
agent_builder.add_edge("resolve_company", "about_company_research")
agent_builder.add_edge("resolve_company", "ai_research")

agent_builder.add_edge("jobs_research", "generate_report")
agent_builder.add_edge("revenue_research", "generate_report")
agent_builder.add_edge("news_research", "generate_report")
agent_builder.add_edge("about_company_research", "generate_report")
agent_builder.add_edge("ai_research", "generate_report")

agent_builder.add_edge("generate_report", END)


# ---- compile ----
researcher_agent = agent_builder.compile()
state = {"researcher_messages": "I want to research about a potential client for a presales pitch. The company I want to research is JAMF"}
result = researcher_agent.invoke(state)
result["company"]
result["company_description"]