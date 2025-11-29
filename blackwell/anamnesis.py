from typing import List, TypedDict, Annotated

# LangChain imports
from langchain_core.messages import AnyMessage, HumanMessage

# LangGraph imports
from langgraph.graph import StateGraph, add_messages, START, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from blackwell.config import fast_model
from blackwell.prompts import anamnesis_prompt, document_analysis_prompt
from blackwell.utils import get_available_docs
from blackwell.document_processer import load_documents


##################### Graph Compiling Script #####################
# This script compiles the LangGraph graph for the Anamnesis Agent.
class AnamnesisState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]  # Built-in MessagesState
    documents_report: AnyMessage
    final_report: str
    function: str

def router(state: AnamnesisState):
    # Route to examination proposal or tests based on state
    if state["final_report"] is not None:
        return END
    elif state["function"] == "chat":
        return "anamnesis"
    elif state["function"] == "document_analysis":
        return "document"
    else:
        return "report"

def anamnesis(state: AnamnesisState) -> AnamnesisState:
    # Generate an answer using messages
    if state["documents_report"] is not None:
        state["messages"] = [fast_model.invoke([anamnesis_prompt] + [state["documents_report"]] + state["messages"])]
    else:
        state["messages"] = [fast_model.invoke([anamnesis_prompt] + state["messages"])]

    return state

def check_anamnesis_completion(state: AnamnesisState):
    # Check if anamnesis is complete
    if "ANAMNESIS REPORT" in state["messages"][-1].content:
        return "report"
    else:
        return END

def document_analysis(state: AnamnesisState) -> AnamnesisState:
    # Load additional tests documents
    docs = get_available_docs(folder_path="./tmp", extensions=["pdf", "txt", "csv"])
    if len(docs) == 0:
        state["documents_report"] = HumanMessage(content="No additional documents provided.")
        return state
    if state["documents_report"] is not None:
        past_processing = state["documents_report"].content
        documents = "Past Document Analysis Report:\n" + past_processing + "\n\n[NEW DOCUMENT CONTENT]:\n" + "\n\n".join(doc.page_content for doc in load_documents(docs))
    else:
        documents = "[DOCUMENT CONTENT]:\n" + "\n\n".join(doc.page_content for doc in load_documents(docs))

    docs_content = HumanMessage(content=documents)
    state["documents_report"] = fast_model.invoke([document_analysis_prompt] + state["messages"] + [docs_content])
    state["messages"] = [HumanMessage(content="New documents received and processed.")]
    
    return state

def final_report(state: AnamnesisState) -> AnamnesisState:
    # Propose examinations based on anamnesis

    document_analysis = state["documents_report"].content if state["documents_report"] is not None else "No additional documents provided."
    document_analysis = document_analysis.replace("[DOCUMENT_ANALYSIS_REPORT]", "**Objective Lab Findings:**").strip()
    anamnesis_report = "[ANAMNESIS REPORT]:\n" + state["messages"][-1].content.split("[ANAMNESIS REPORT]:")[-1]
    state["final_report"] = anamnesis_report +"\n\n"+document_analysis
    state["messages"] = state["messages"][:-1]
    
    return state


print("Compiling Anamnesis Agent...")
workflow = StateGraph(AnamnesisState)
memory = MemorySaver()

# Add nodes
workflow.add_node("anamnesis", anamnesis)
workflow.add_node("document", document_analysis)
workflow.add_node("report", final_report)

# Create edges
workflow.add_conditional_edges("anamnesis", check_anamnesis_completion)
workflow.add_conditional_edges(START, router)
workflow.add_edge("document", "anamnesis")
workflow.add_edge("report", END)

# Compile the graph
AnamnesisAgent = workflow.compile(checkpointer=memory)
