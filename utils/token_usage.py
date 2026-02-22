from graph_orchestration.define_state import ResearcherState
from langchain_core.messages import HumanMessage
import uuid
import time


# -------------------------
# INIT STATE HELPER
# -------------------------


def init_state(user_query: str) -> ResearcherState:
    return {
        "researcher_messages": [HumanMessage(content=user_query)],
        "tool_metrics": {"total_calls": 0, "by_tool": {}, "calls": []},
        "token_metrics": {"input": 0, "output": 0, "total": 0},
        "loop_count": 0,
        "run_id": str(uuid.uuid4()),
        "start_time": time.time(),
    }


# -------------------------
# TOKEN LOGGER
# -------------------------


def update_token_metrics(state: ResearcherState, resp):
    usage = {}

    if hasattr(resp, "response_metadata"):
        usage = resp.response_metadata.get("token_usage", {})

    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)

    state["token_metrics"]["input"] += inp
    state["token_metrics"]["output"] += out
    state["token_metrics"]["total"] += inp + out
