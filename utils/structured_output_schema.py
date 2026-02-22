from pydantic import BaseModel, Field


# ===== STRUCTURED OUTPUT SCHEMAS ======================================
class ResearchQuestion(BaseModel):
    """Schema for research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


class Summary(BaseModel):
    """Schema for webpage content summarization."""

    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(
        description="Important quotes and excerpts from the content"
    )
