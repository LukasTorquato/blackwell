from typing import List, TypedDict, Dict
import os

# LangChain imports
from langsmith import traceable
from langchain.agents import create_agent
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
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

    #context: List[Document]
    t_run: int
    anamnesis_report: AnyMessage
    query: AnyMessage  # Improved query for vector similarity search
    reports: Dict[str, AnyMessage]
    final_report: str
    references: List[Dict]  # Track all references from RAG and PubMed


@traceable(run_type="llm")
def analyze_query(state: GraphState) -> GraphState:
    # Analyze user query to improve research
    state["t_run"] += 1
    if state["t_run"] == 1:
        print("Proposing Hypothesis query...")
        state["query"] = fast_model.invoke([hypothesis_rag_prompt, 
                                            state["anamnesis_report"]])
    else:
        print("Proposing Treatment query...")
        state["query"] = fast_model.invoke([treatment_rag_prompt, 
                                            state["reports"]["hypothesis_report"], 
                                            state["anamnesis_report"]])
    return state

@traceable(run_type="llm")
def rag_research(state: GraphState) -> GraphState:
    # Use RAG agent to retrieve documents and crawl web if needed
    print("RAG Agent researching...")
    try:
        if state["query"].content == "" or state["query"] is None:
            raise Exception("No query returned from analysis.")
        
        if state["reports"].get("hypothesis_report") is None:
            # Invoke the RAG agent with the query
            result = rag_agent_diagnosis.invoke({"messages": [{"role": "user", "content": state["query"].content}]})
        else:
            result = rag_agent_treatment.invoke({"messages": [{"role": "user", "content": state["query"].content}]})
        
        # Extract the research report from the agent's response
        #print(result)
        research_content = result['messages'][-1].content
        print(f"RAG Agent completed research with {len(result['messages'])} messages")
        
        # Hardwiring fix for list response
        if type(research_content) == list:
            research_content = research_content[0]["text"]

        state["reports"]["research_report"] = HumanMessage(content=research_content)

        references = research_content.split("**References:**")[1]
        # Extract RAG references from tool calls in messages
        for ref in references.split("\n"):
            #print(f"RAG Reference found: {ref}")
            if ref.strip() != "":
                if ref != "---":
                    state["references"].append({
                        "type": "RAG",
                        "reference": ref.strip("* ")
                    })
            
        
    except Exception as e:
        print(f"Error in RAG research: {e}")
        state["research_report"] = f"Error during research: {str(e)}"
    
    return state

@traceable(run_type="llm")
def pubmed_search(state: GraphState) -> GraphState:
    # Placeholder for PubMed search node
    print("Performing PubMed search for additional context...")
    result = pubmed_agent.invoke({"messages": [{"role": "user", "content": state["query"].content}]})
    research_content = result['messages'][-1].content

    # Hardwiring fix for list response
    if type(research_content) == list:
        research_content = research_content[0]["text"]

    combined_report = state["reports"]["research_report"].content + "\n\n" + research_content
    state["reports"]["research_report"] = HumanMessage(content=combined_report)

    references = research_content.split("**References:**")[1]
    # Extract PubMed references
    for ref in references.split("\n"):
        if ref.strip() != "":
            if ref != "---":
                state["references"].append({
                    "type": "PubMed",
                    "reference": ref.strip("* ")
                })
   

    return state

@traceable(run_type="llm")
def generate_hypothesis(state: GraphState) -> GraphState:
    # Generate a hypothesis using retrieved context
    print("Assessing clinical certainty expert...")
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

@traceable(run_type="llm")
def generate_treatment(state: GraphState) -> GraphState:
    # Generate a treatment plan using retrieved context
    print("Generating treatment plan...")
    state["reports"]["treatment_report"] = fast_model.invoke([treatment_eval_prompt] + 
                                                  [state["reports"]["hypothesis_report"]] + 
                                                  [state["anamnesis_report"]] + 
                                                  [state["reports"]["research_report"]])
    print("Generating final report...")    
    result = pro_model.invoke([final_report_prompt] + 
                                              [state["anamnesis_report"]] + 
                                              [state["reports"]["hypothesis_report"]] + 
                                              [state["reports"]["treatment_report"]] + 
                                              [state["reports"]["research_report"]] )

    references_text = format_references(state["references"])
    state["final_report"] = result.content + f"\n\n{references_text}"

    return state

def run_count(state: GraphState):
    if state["t_run"] == 1:
        return "hypothesis"
    else:
        return "pubmed"


# Build the vector store
print("Building vector store for RAG...")
vector_store = build_retriever(add_new_docs=False)

# Initialize RAG tools with the vector store
print("Initializing RAG tools...")
initialize_rag_tools(vector_store)
rag_agent_diagnosis = create_agent(
    model=light_model,
    tools=RAG_TOOLS,
    system_prompt=diagnostic_rag_prompt.content
)

rag_agent_treatment = create_agent(
    model=light_model,
    tools=RAG_TOOLS,
    system_prompt=therapeutic_rag_prompt.content
)
# Initialize PubMed tools
print("Initializing PubMed tools...")
initialize_pubmed_tools(api_key=os.getenv("PUBMED_API_KEY"))
pubmed_agent = create_agent(
    model=light_model, 
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
workflow.add_conditional_edges("rag_research", run_count)
workflow.add_edge("pubmed", "treatment")
workflow.add_edge("treatment", END)

# Set the entry point
workflow.set_entry_point("analyze")

# Compile the graph
EvaluatorAgent = workflow.compile(checkpointer=memory)
