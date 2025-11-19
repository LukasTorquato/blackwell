from typing import List, TypedDict, Dict
import os

# LangChain imports
from langsmith import traceable
from langchain.agents import create_agent
from langchain_core.messages import (
    HumanMessage,
    AnyMessage
)

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from blackwell.config import *
from blackwell.prompts import *
from blackwell.utils import format_references
from blackwell.document_processer import build_retriever
from blackwell.pubmed_tools import PUBMED_TOOLS, initialize_pubmed_tools
from blackwell.rag_tools import RAG_TOOLS, initialize_rag_tools


##################### Graph Compiling Script #####################
# This script compiles the LangGraph graph, the sub-agents and their tools.
class GraphState(TypedDict):
    # Type for the state of the retrieval and query graph
    next_node: str
    anamnesis_report: AnyMessage
    query: AnyMessage  # Improved query for vector similarity search
    reports: Dict[str, AnyMessage]
    final_report: str
    references: List[Dict]  # Track all references from RAG and PubMed


@traceable(run_type="llm")
def analyze_query(state: GraphState) -> GraphState:
    # Analyze user query to improve research
    try:
        if state["reports"].get("hypothesis_report") is None:
            print("Proposing Hypothesis query...")
            state["query"] = fast_model.invoke([hypothesis_rag_prompt, 
                                                state["anamnesis_report"]])
        else:
            print("Proposing Treatment query...")
            state["query"] = fast_model.invoke([treatment_rag_prompt, 
                                                state["reports"]["hypothesis_report"], 
                                                state["anamnesis_report"]])
        return state
    
    except Exception as e:
        logger.error(f"Error in query analysis: {e}")
        logger.error(f"Result:\n{state['query']}")
        state["next_node"] = "analyze"  # Retry analysis on error
        return state

@traceable(run_type="llm")
def rag_research(state: GraphState) -> GraphState:
    # Use RAG agent to retrieve documents and crawl web if needed
    print("RAG Agent researching...")
    try:
        if state["query"] is None or state["query"].content == "":
            raise Exception("No query returned from analysis.")
        
        if state["reports"].get("hypothesis_report") is None:
            # Diagnostic Pipeline
            result = rag_agent_diagnosis.invoke({"messages": [{"role": "user", "content": state["query"].content}]})
            state["next_node"] = "hypothesis"
        else:
            # Therapeutic Pipeline
            result = rag_agent_treatment.invoke({"messages": [{"role": "user", "content": state["query"].content}]})
            state["next_node"] = "pubmed"
        
        if len(result['messages']) < 3:
            # Retry if insufficient messages
            raise Exception("RAG Agent returned insufficient messages, retrying once...")
        
        research_content = result['messages'][-1].content
        print(f"RAG Agent completed research with {len(result['messages'])} messages")
        
        # Hardwiring fix for list response
        if type(research_content) == list:
            research_content = research_content[0]["text"]

        state["reports"]["research_report"] = HumanMessage(content=research_content)

        if ("**References:**" in research_content):
            references = research_content.split("**References:**")[1]
            # Extract RAG references from tool calls in final report
            for ref in references.split("\n"):
                #print(f"RAG Reference found: {ref}")
                if ref.strip() != "":
                    if ref != "---":
                        state["references"].append({
                            "type": "RAG",
                            "reference": ref.strip("* ")
                        })
        else:
            raise Exception("No references found in RAG research report:\n", research_content)
 
    except Exception as e:
        logger.error(f"Error in RAG research: {e}")
        state["next_node"] = "analyze"  # Retry analysis on error

    return state

@traceable(run_type="llm")
def pubmed_search(state: GraphState) -> GraphState:
    # Placeholder for PubMed search node
    print("Performing PubMed search for additional context...")
    try:
        result = pubmed_agent.invoke({"messages": [{"role": "user", "content": state["query"].content}]})
        pubmed_research = result['messages'][-1].content

        # Hardwiring fix for list response
        if type(pubmed_research) == list:
            pubmed_research = pubmed_research[0]["text"]

        # Append PubMed research to existing RAG research report    
        combined_report = state["reports"]["research_report"].content + "\n\n" + pubmed_research
        state["reports"]["research_report"] = HumanMessage(content=combined_report)

        if ("**References:**" in pubmed_research):
            references = pubmed_research.split("**References:**")[1]
            # Extract PubMed references
            for ref in references.split("\n"):
                if ref.strip() != "":
                    if ref != "---":
                        state["references"].append({
                            "type": "PubMed",
                            "reference": ref.strip("* ")
                        })

        return state

    except Exception as e:
        logger.error(f"Error in PubMed research: {e}")
        if 'result' in locals():
            logger.error(f"Result:\n{result}")
        return state

@traceable(run_type="llm")
def generate_hypothesis(state: GraphState) -> GraphState:
    # Generate a hypothesis using retrieved context
    print("Assessing clinical certainty expert...")
    try:
        state["reports"]["certainty_report"] = fast_model.invoke(
                                    [clinical_certainty_prompt] + 
                                    [state["anamnesis_report"]] + 
                                    [state["reports"]["research_report"]])
        
        print("Running investigator expert...")
        state["reports"]["investigator_report"] = fast_model.invoke(
                                    [investigative_workup_prompt] + 
                                    [state["anamnesis_report"]] + 
                                    [state["reports"]["research_report"]])
        
        print("Generating hypothesis report...")
        state["reports"]["hypothesis_report"] = fast_model.invoke(
                                    [hypothesis_synthesis_prompt] + 
                                    [state["anamnesis_report"]] + 
                                    [state["reports"]["certainty_report"]] + 
                                    [state["reports"]["investigator_report"]] +
                                    [state["reports"]["research_report"]])

        return state
    except Exception as e:
        logger.error(f"Error in hypothesis node: {e}")
        logger.error(f"Reports:\n{state['reports']}")
        return state

@traceable(run_type="llm")
def generate_treatment(state: GraphState) -> GraphState:
    # Generate a treatment plan using retrieved context
    print("Generating treatment plan...")
    try:
        state["reports"]["treatment_report"] = fast_model.invoke([treatment_eval_prompt] + 
                                                    [state["reports"]["hypothesis_report"]] + 
                                                    [state["anamnesis_report"]] + 
                                                    [state["reports"]["research_report"]])
        print("Generating final report...")    
        result = pro_model.invoke([final_report_prompt] + 
                                [state["anamnesis_report"]] + 
                                [state["reports"]["hypothesis_report"]] + 
                                [state["reports"]["treatment_report"]] + 
                                [state["reports"]["research_report"]])

        references_text = format_references(state["references"])
        if type(result) == list:
            state["final_report"] = result[0].content + f"\n\n{references_text}"
        else:
            state["final_report"] = result.content + f"\n\n{references_text}"

        return state
    
    except Exception as e:
        logger.error(f"Error in treatment node: {e}")
        logger.error(f"Reports:\n{state['reports']}")
        if 'result' in locals() and result is not None:
            logger.error(f"Final report result:\n{result}")
        return state

def router(state: GraphState):
    return state["next_node"]


# Build the vector store
print("Building vector store for RAG...")
vector_store = build_retriever(add_new_docs=False)

# Initialize RAG tools with the vector store
print("Initializing RAG tools...")
initialize_rag_tools(vector_store)
print("Creating RAG agents...")
quoted_d_prompt = diagnostic_rag_prompt.content.format(quota=QUOTA_LIMIT)
quoted_t_prompt = therapeutic_rag_prompt.content.format(quota=QUOTA_LIMIT)
rag_agent_diagnosis = create_agent(
    model=agent_model,
    tools=RAG_TOOLS,
    system_prompt=quoted_d_prompt
)
rag_agent_treatment = create_agent(
    model=agent_model,
    tools=RAG_TOOLS,
    system_prompt=quoted_t_prompt
)
# Initialize PubMed tools
print("Initializing PubMed tools...")
initialize_pubmed_tools(api_key=os.getenv("PUBMED_API_KEY"))
pubmed_agent = create_agent(
    model=agent_model, 
    tools=PUBMED_TOOLS,
    system_prompt=pubmed_research_agent_prompt.content
)

# Create the graph
print("Compiling Evaluator Agent...")
workflow = StateGraph(GraphState)
memory = MemorySaver()

# Add nodes
workflow.add_node("analyze", analyze_query)
workflow.add_node("rag_research", rag_research)
workflow.add_node("hypothesis", generate_hypothesis)
workflow.add_node("treatment", generate_treatment)
workflow.add_node("pubmed", pubmed_search)

# Create edges
workflow.add_edge("analyze", "rag_research")
workflow.add_edge("hypothesis", "analyze")
workflow.add_conditional_edges("rag_research", router)
workflow.add_edge("pubmed", "treatment")
workflow.add_edge("treatment", END)

# Set the entry point
workflow.set_entry_point("analyze")

# Compile the graph
EvaluatorAgent = workflow.compile(checkpointer=memory)
