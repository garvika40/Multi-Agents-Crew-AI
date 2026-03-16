import asyncio
import io
from contextlib import redirect_stdout
from typing import Any

import streamlit as st

DEEP_RESEARCH_WORKFLOW = "Deep Research"
LINKEDIN_WORKFLOW = "LinkedIn Content Creation"


def _extract_text_from_result(result: Any) -> str:
    if result is None:
        return "No content returned."
    if isinstance(result, str):
        return result

    raw_value = getattr(result, "raw", None)
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value

    tasks_output = getattr(result, "tasks_output", None)
    if tasks_output:
        for task_output in reversed(tasks_output):
            task_raw = getattr(task_output, "raw", None)
            if isinstance(task_raw, str) and task_raw.strip():
                return task_raw

    return str(result)


def _run_deep_research(topic: str) -> str:
    if not topic.strip():
        raise ValueError("Topic is required for Deep Research.")

    from crew_ai_agents.crew_deep_research import ResearchCrew

    with redirect_stdout(io.StringIO()):
        crew = ResearchCrew().crew()
        result = crew.kickoff(inputs={"topic": topic.strip()})

    return _extract_text_from_result(result)


def _run_ticket_creation() -> str:
    from crew_ai_agents.crew_ticket_creation import TicketRoutingFlow

    with redirect_stdout(io.StringIO()):
        result = asyncio.run(TicketRoutingFlow().kickoff_async())

    return _extract_text_from_result(result)


def _run_linkedin_content(
    topic: str, source_url: str, style_notes: str = ""
) -> dict[str, str]:
    if not topic.strip():
        raise ValueError("Topic is required for LinkedIn Content Creation.")

    from crew_ai_agents.linkdin_content_creation import ContentPipeline

    with redirect_stdout(io.StringIO()):
        flow = ContentPipeline()
        flow.state.topic = topic.strip()
        flow.state.source_url = source_url.strip()
        flow.state.style_notes = style_notes.strip()  # Pass custom style notes
        flow.kickoff()

    return {
        "draft": flow.state.draft_content or "No draft content returned.",
        "critic_status": flow.state.critic_status or "unknown",
    }


def _render_deep_research_output(content: str) -> None:
    st.subheader("Research Output")
    st.markdown(content)


def _render_linkedin_output(content: dict[str, str]) -> None:
    critic_status = content.get("critic_status", "unknown")
    status_label = critic_status.replace("_", " ").title()

    st.subheader("LinkedIn Draft")
    st.markdown(content.get("draft", "No draft content returned."))
    st.caption(f"Critic Status: {status_label}")


def main() -> None:
    st.set_page_config(page_title="Crew AI Workbench", layout="wide")
    st.title("Crew AI Workbench")
    st.write("Pick a workflow, provide inputs, and run it from one place.")

    workflow = st.selectbox(
        "Choose a workflow",
        (
            # DEEP_RESEARCH_WORKFLOW,
            LINKEDIN_WORKFLOW,
        ),
    )

    topic = ""
    source_url = ""
    style_notes = ""

    # if workflow == DEEP_RESEARCH_WORKFLOW:
    #     topic = st.text_area("Topic", placeholder="Enter the research topic")

    if workflow == LINKEDIN_WORKFLOW:
        topic = st.text_area("Topic", placeholder="Enter the post topic")
        source_url = st.text_input(
            "Source URL (optional)",
            placeholder="https://example.com/article",
        )

        # Load available style files from assets
        from crew_ai_agents.linkdin_content_creation import (
            _list_available_style_files,
            _load_style_from_file,
        )

        available_styles = _list_available_style_files()
        style_choice = st.radio(
            "Content Style",
            ["Custom Style"] + ["System Default"],
            horizontal=True,
        )

        if style_choice != "Custom Style":
            style_notes = _load_style_from_file(style_choice)
            st.info(f"Loaded style from: {style_choice}")

        # Allow custom style input
        custom_style = st.text_area(
            "Or paste custom style notes (optional)",
            placeholder="Describe the tone, format, and style you want...",
            height=100,
        )
        if custom_style.strip():
            style_notes = custom_style

    run_clicked = st.button("Run", type="primary")

    if run_clicked:
        with st.spinner("Running workflow..."):
            if workflow == DEEP_RESEARCH_WORKFLOW:
                result = _run_deep_research(topic)
            # elif workflow == "Ticket Creation":
            #     result, logs = _run_ticket_creation()
            else:
                result = _run_linkedin_content(topic, source_url, style_notes)

            st.success("Workflow finished.")
            if workflow == DEEP_RESEARCH_WORKFLOW:
                _render_deep_research_output(result)
            else:
                _render_linkedin_output(result)


if __name__ == "__main__":
    main()
