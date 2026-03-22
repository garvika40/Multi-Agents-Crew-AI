# Multi-Agents-Crew-AI (Crew AI Workbench) AND Multi Agent Deep Research (Langchain and Langgraph)

A multi-agent AI workbench built with **CrewAI** + **Streamlit** for running agent-based workflows from a single UI.  
This repository currently focuses on a **LinkedIn Content Creation** pipeline that researches a topic, optionally scrapes a source URL for context, drafts a post, and runs a critic step to validate the output.

## Key Features

- **Streamlit UI** to run workflows interactively (`streamlit_app.py`)
- **Multi-agent orchestration with CrewAI**
  - Topic Analyst (refines/sharpens topic)
  - Researcher (web research + optional deep URL scraping)
  - Editorial Filter (filters/organizes research)
  - Writer (generates a LinkedIn post draft)
  - Critic (checks draft and returns status/violations)
- **Research tooling**
  - **Tavily Search** for web search
  - **Firecrawl** for scraping a provided URL into markdown context
- **Config-driven agents and tasks**
  - YAML-based configs under `crew_ai_agents/config/`
- **Style control**
  - Use **custom style notes** in the UI
  - Or load style reference content from `crew_ai_agents/assets/*.txt`

## Project Structure

- `streamlit_app.py` — Streamlit “Crew AI Workbench” UI
- `crew_ai_agents/`
  - `linkdin_content_creation.py` — LinkedIn content pipeline (Crew + Flow)
  - `crew_deep_research.py` — Deep research crew (present, UI currently commented)
  - `crew_ticket_creation.py` — Ticket routing flow (present, UI currently commented)
  - `config/` — YAML files for agent/task definitions
  - `assets/` — style/reference `.txt` files
- `agents/` — extra search agent utilities
- `tools/` — research tools helpers
- `utils/` — prompt utilities, formatting, schemas, token tracking
- `pyproject.toml` / `uv.lock` — Python project and dependency lock

## Requirements

- Python **3.12+** (see `.python-version`)
- API keys for external services (see Environment Variables below)

## Environment Variables

Create a `.env` file in the project root (or export these in your shell):

```bash
OPENAI_API_KEY=...
TAVILY_API_KEY=...
FIRECRAWL_API_KEY=...
```

Notes:
- `OPENAI_API_KEY` is used by CrewAI’s LLM config (this project uses `gpt-4o` in code).
- `TAVILY_API_KEY` powers the web search tool.
- `FIRECRAWL_API_KEY` is used to scrape a provided source URL into markdown context.

## Installation

This repo uses a `pyproject.toml` setup. You can install dependencies with any modern Python workflow.

### Option A: Using `uv` (recommended if you already use it)
```bash
uv sync
```

### Option B: Using `pip`
```bash
pip install -r requirements.txt
```

> If you don’t have a `requirements.txt`, you can install from `pyproject.toml` using a tool like `pip install .`
> or use `uv`/`poetry`. (This repo includes `uv.lock`, so `uv sync` is usually easiest.)

## Run the App (Streamlit)

```bash
streamlit run streamlit_app.py
```

Then open the local URL Streamlit prints in your terminal.

## How the LinkedIn Workflow Works

1. **Input**: topic (+ optional source URL + optional style notes)
2. **(Optional)** If a source URL is provided, the app scrapes it (Firecrawl) and truncates context.
3. **Crew runs sequentially**:
   - sharpen topic → research → filter → write draft → critic
4. **Output**:
   - A LinkedIn post draft
   - Critic status (shown in the UI)

## Example Usage

- Topic: `AI agents for enterprise customer support`
- Source URL (optional): an article you want the post grounded in
- Style notes: “Professional tone, concise paragraphs, include a short hook and 3 bullet takeaways.”

## Roadmap / Ideas

- Enable additional workflows in the Streamlit UI:
  - Deep Research workflow
- Add export options (Markdown / PDF / JSON)
- Add citations/links formatting for research sources
- Improve critic feedback loop (auto-rewrite if violations found)

## Troubleshooting

- **Empty output or errors**: verify your `.env` keys are set correctly.
- **Firecrawl scrape fails**: try a different URL or run without a source URL.
- **Model access**: ensure your OpenAI account has access to the configured model.

