"""
PubMed Tools for LangChain Agents

This module provides ready-to-use LangChain tools for integrating PubMed research
into your clinical decision support agents.
"""

from typing import Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from blackwell.pubmed import TreatmentResearcher


# Initialize a global researcher instance
_researcher: Optional[TreatmentResearcher] = None


def initialize_pubmed_tools(email: Optional[str] = None, api_key: Optional[str] = None):
    """
    Initialize the PubMed researcher with optional credentials.
    
    Args:
        email: Your email (recommended by NCBI)
        api_key: NCBI API key for higher rate limits (optional)
    """
    global _researcher
    _researcher = TreatmentResearcher(email=email, api_key=api_key)


def get_researcher() -> TreatmentResearcher:
    """Get or create the global researcher instance."""
    global _researcher
    if _researcher is None:
        _researcher = TreatmentResearcher()
    return _researcher


# Pydantic models for tool arguments
class ResearchTreatmentOptionsInput(BaseModel):
    """Input schema for research_treatment_options tool."""
    diagnosis: str = Field(description="The medical diagnosis or condition to research")
    max_results: int = Field(default=10, description="Maximum number of articles to retrieve", ge=1, le=20)


class ResearchSpecificTreatmentInput(BaseModel):
    """Input schema for research_specific_treatment_efficacy tool."""
    diagnosis: str = Field(description="The medical diagnosis or condition")
    treatment: str = Field(description="The specific treatment to research")
    max_results: int = Field(default=8, description="Maximum number of articles to retrieve", ge=1, le=20)


class GetTreatmentGuidelinesInput(BaseModel):
    """Input schema for get_treatment_guidelines tool."""
    diagnosis: str = Field(description="The medical diagnosis or condition")
    max_results: int = Field(default=5, description="Maximum number of guidelines to retrieve", ge=1, le=10)


# Tool functions (internal implementations)
def _research_treatment_options_func(diagnosis: str, max_results: int = 10) -> str:
    """Research treatment options for a given diagnosis from PubMed."""
    researcher = get_researcher()
    results = researcher.research_treatment(
        diagnosis=diagnosis,
        max_results=max_results,
        years_back=5,
        include_reviews=True,
        include_clinical_trials=True
    )
    return researcher.format_results_for_llm(results)


def _research_specific_treatment_efficacy_func(diagnosis: str, treatment: str, max_results: int = 8) -> str:
    """Research the efficacy of a specific treatment for a given diagnosis."""
    researcher = get_researcher()
    results = researcher.research_specific_treatment(
        diagnosis=diagnosis,
        treatment=treatment,
        max_results=max_results,
        years_back=5
    )
    return researcher.format_results_for_llm(results)


def _get_treatment_guidelines_func(diagnosis: str, max_results: int = 5) -> str:
    """Find clinical practice guidelines and recommendations for treating a diagnosis."""
    researcher = get_researcher()
    results = researcher.get_treatment_guidelines(
        diagnosis=diagnosis,
        max_results=max_results,
        years_back=3
    )
    return researcher.format_results_for_llm(results)


# Create structured tools
research_treatment_options = StructuredTool.from_function(
    func=_research_treatment_options_func,
    name="research_treatment_options",
    description=(
        "Research treatment options for a given diagnosis from PubMed. "
        "Searches for recent clinical trials, systematic reviews, and meta-analyses. "
        "Returns a formatted summary of relevant articles with titles, abstracts, and URLs."
    ),
    args_schema=ResearchTreatmentOptionsInput
)

research_specific_treatment_efficacy = StructuredTool.from_function(
    func=_research_specific_treatment_efficacy_func,
    name="research_specific_treatment_efficacy",
    description=(
        "Research the efficacy of a specific treatment for a given diagnosis. "
        "Searches PubMed for research on treatment effectiveness. "
        "Useful when you have a treatment hypothesis and want evidence."
    ),
    args_schema=ResearchSpecificTreatmentInput
)

get_treatment_guidelines = StructuredTool.from_function(
    func=_get_treatment_guidelines_func,
    name="get_treatment_guidelines",
    description=(
        "Find clinical practice guidelines and recommendations for treating a diagnosis. "
        "Searches for official treatment guidelines and consensus statements from medical societies."
    ),
    args_schema=GetTreatmentGuidelinesInput
)

# List of all tools for easy import
PUBMED_TOOLS = [
    research_treatment_options,
    research_specific_treatment_efficacy,
    get_treatment_guidelines
]
