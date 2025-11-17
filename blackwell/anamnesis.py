from typing import List, TypedDict, Annotated

# LangChain imports
from langchain_core.messages import AnyMessage

# LangGraph imports
from langgraph.graph import StateGraph, add_messages, END
from langgraph.checkpoint.memory import MemorySaver
#from langsmith import traceable

# Local imports
from blackwell.config import fast_model
from blackwell.prompts import anamnesis_prompt


##################### Graph Compiling Script #####################
# This script compiles the LangGraph graph for the Anamnesis Agent.
class AnamnesisState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]  # Built-in MessagesState

#@traceable(run_type="llm")
def anamnesis(state: AnamnesisState) -> AnamnesisState:
    # Generate an answer using messages
    state["messages"] = [fast_model.invoke([anamnesis_prompt] + state["messages"])]

    return state


print("Compiling Anamnesis Agent...")
workflow = StateGraph(AnamnesisState)
memory = MemorySaver()

# Add nodes
workflow.add_node("anamnesis", anamnesis)

# Create edges
workflow.add_edge("anamnesis", END)

# Set the entry point
workflow.set_entry_point("anamnesis")

# Compile the graph
AnamnesisAgent = workflow.compile(checkpointer=memory)
