import json
import time
import uuid
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import (
    filter_messages,
    SystemMessage,
    ToolMessage,
    HumanMessage,
    AIMessage,
)
from typing import Literal
from utils.prompt import research_agent_prompt
from graph_orchestration.define_state import ResearcherState, ResearcherOutputState
from tools.research_tools import think_tool, tavily_search_tool, duckduckgo_search_tool
from utils.token_usage import update_token_metrics
from utils.prompt import (
    compress_research_system_prompt,
    compress_research_human_message,
)
from utils.message_formatting import get_today_str, format_messages


# 3. Agent Construction
tools = [think_tool, tavily_search_tool, duckduckgo_search_tool]
tools_by_name = {tool.name: tool for tool in tools}
model = init_chat_model(model="openai:gpt-4o")
model_with_tools = model.bind_tools(tools)
summarization_model = init_chat_model(model="openai:gpt-4.1-mini")
compress_model = init_chat_model(
    model="openai:gpt-4.1", max_tokens=32000
)  # model="anthropic:claude-sonnet-4-20250514", max_tokens=64000


def init_state(user_query: str) -> ResearcherState:
    return {
        "researcher_messages": [HumanMessage(content=user_query)],
        "tool_metrics": {"total_calls": 0, "by_tool": {}, "calls": []},
        "token_metrics": {"input": 0, "output": 0, "total": 0},
        "loop_count": 0,
        "run_id": str(uuid.uuid4()),
        "start_time": time.time(),
    }


def llm_call(state: ResearcherState):
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    state["loop_count"] += 1

    messages = [SystemMessage(content=research_agent_prompt)] + state[
        "researcher_messages"
    ]

    resp = model_with_tools.invoke(messages)
    update_token_metrics(state, resp)

    return {"researcher_messages": [resp]}


def tool_node(state: ResearcherState):
    """Execute all tool calls from the previous LLM response.

    Executes all tool calls from the previous LLM responses.
    Returns updated state with tool execution results.
    """

    tool_calls = state["researcher_messages"][-1].tool_calls
    observations = []

    for tool_call in tool_calls:

        name = tool_call["name"]
        args = tool_call["args"]

        result = tools_by_name[name].invoke(args)
        observations.append(result)

        # ---- metrics ----

        state["tool_metrics"]["total_calls"] += 1

        state["tool_metrics"]["by_tool"].setdefault(name, 0)
        state["tool_metrics"]["by_tool"][name] += 1

        state["tool_metrics"]["calls"].append({"tool": name, "args": args})

    tool_outputs = [
        ToolMessage(content=o, name=tc["name"], tool_call_id=tc["id"])
        for o, tc in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}


# -------------------------
# ROUTER
# -------------------------

MAX_TOOL_CALLS = 3
MAX_LOOPS = 3


def should_continue(
    state: ResearcherState,
) -> Literal["tool_node", "compress_research"]:

    # guardrails
    if state["tool_metrics"]["total_calls"] >= MAX_TOOL_CALLS:
        return "compress_research"

    if state["loop_count"] >= MAX_LOOPS:
        return "compress_research"

    last = state["researcher_messages"][-1]

    if last.tool_calls:
        return "tool_node"

    return "compress_research"


# -------------------------
# COMPRESS NODE
# -------------------------


def compress_research(state: ResearcherState):

    system_msg = compress_research_system_prompt.format(date=get_today_str())

    messages = (
        [SystemMessage(content=system_msg)]
        + state["researcher_messages"]
        + [HumanMessage(content=compress_research_human_message)]
    )
    clean_messages = []

    for m in messages:
        # Skip tool response messages
        if isinstance(m, ToolMessage):
            continue

        # Skip assistant tool-call messages
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            continue

        clean_messages.append(m)

    resp = compress_model.invoke(clean_messages)

    update_token_metrics(state, resp)

    raw_notes = [
        str(m.content)
        for m in filter_messages(
            state["researcher_messages"],
            include_types=["tool", "ai"],
        )
    ]

    return {
        "compressed_research": str(resp.content),
        "raw_notes": ["\n".join(raw_notes)],
    }


# -------------------------
# FINALIZE LOG
# -------------------------


def finalize_run_log(state: ResearcherState):

    duration = time.time() - state["start_time"]

    log = {
        "run_id": state["run_id"],
        "loops": state["loop_count"],
        "tool_metrics": state["tool_metrics"],
        "token_metrics": state["token_metrics"],
        "duration_sec": round(duration, 2),
    }

    print("\n===== AGENT RUN LOG =====")
    print(json.dumps(log, indent=2))

    # optional: persist
    # redis.set(f"agent_log:{state['run_id']}", json.dumps(log))

    return {}


# ===== GRAPH CONSTRUCTION =====

agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# ---- nodes ----
agent_builder.add_node("init_state", init_state)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)
agent_builder.add_node("finalize_run_log", finalize_run_log)

# ---- edges ----

agent_builder.add_edge(START, "llm_call")

agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        "compress_research": "compress_research",
    },
)

agent_builder.add_edge("tool_node", "llm_call")

agent_builder.add_edge("compress_research", "finalize_run_log")
agent_builder.add_edge("finalize_run_log", END)

# ---- compile ----

researcher_agent = agent_builder.compile()


# Example brief
research_brief = """I want to research about a potential client for a presales pitch. The company I want to research is JAMF, I want everything from tge market it serves to the work it does, to its clients, excecutives what they post on linkdin and any thing else that must be required to understand to design  a presales pitch"""
state = init_state(research_brief)

result = researcher_agent.invoke(state)
format_messages(result["researcher_messages"])
