from typing import List, TypedDict, Annotated

# LangChain imports
from langchain_core.messages import AnyMessage, SystemMessage

# LangGraph imports
from langgraph.graph import StateGraph, add_messages, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from blackwell.config import fast_model
from blackwell.prompts import anamnesis_prompt


##################### Graph Compiling Script #####################
# This script compiles the LangGraph graph for the Anamnesis Agent.
class PatientState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]  # Built-in MessagesState
    profile: str

def generate_message(state: PatientState) -> PatientState:
    # Generate an answer using messages
    patient_profile = SystemMessage(content=state["profile"])
    state["messages"] = [fast_model.invoke([patient_profile] + state["messages"])]

    return state


print("Compiling AI Patient Agent...")
workflow = StateGraph(PatientState)
memory = MemorySaver()

# Add nodes
workflow.add_node("generate_message", generate_message)

# Create edges
workflow.add_edge("generate_message", END)

# Set the entry point
workflow.set_entry_point("generate_message")

# Compile the graph
AiPatient = workflow.compile(checkpointer=memory)
