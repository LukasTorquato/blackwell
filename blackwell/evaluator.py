from typing import List, TypedDict, Annotated

# LangChain imports
from langchain.schema import Document
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AnyMessage,
    RemoveMessage,
)

# LangGraph imports
from langgraph.graph import StateGraph, add_messages, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from blackwell.config import *
from blackwell.prompts import evaluator_prompt, analysis_prompt, reflect_prompt
from blackwell.document_processer import build_retriever


##################### Graph Compiling Script #####################
# This script compiles the LangGraph graph and the vector store for the RAG pipeline.
class GraphState(TypedDict):
    # Type for the state of the retrieval and query graph

    context: List[Document]
    anamnesis_report: AnyMessage
    query: AnyMessage  # Improved query for vector similarity search
    final_report: AnyMessage


# Define the nodes in the graph
def analyze_query(state: GraphState) -> GraphState:
    # Analyze user query to improve vector similarity search
    print("Analyzing query...")

    state["query"] = evaluator_llm.invoke([analysis_prompt, state["anamnesis_report"]])
    return state


def retrieve_documents(state: GraphState) -> GraphState:
    # Retrieve relevant documents for the latest query
    print("Retrieving documents...")

    # print(f"Improved query: {state['query'].content}")
    # TODO: Check if similarity search can be done with HumanMessage instead of str
    # TODO: https://smith.langchain.com/hub/zulqarnain/multi-query-retriever-similarity
    try:
        if state["query"].content == "" or state["query"] is None:
            raise Exception("No query returned from analysis.")

        retrieved_docs = vector_store.similarity_search(state["query"].content, k=DOCS_RETRIEVED)
        if retrieved_docs == [] or retrieved_docs is None:
            raise Exception("No documents found for the query.")

        state["context"] = retrieved_docs
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        state["context"] = None

    return state


def generate_answer(state: GraphState) -> GraphState:
    # Generate an answer using retrieved context
    print("Generating answer...")

    documents = state["context"]

    # Format context from documents
    if documents == [] or documents is None:
        context_str = "No relevant context found."
    else:
        context_str = "[RAG_CONTEXT]:\n" + "\n\n".join(doc.page_content for doc in documents)
    context = SystemMessage(content=context_str)

    state["final_report"] = [evaluator_llm.invoke([evaluator_prompt] + [state["anamnesis_report"]] + [context])]

    return state




# Build the vector store
vector_store = build_retriever()

# Create the graph
print("Compiling Evaluator Agent...")
workflow = StateGraph(GraphState)
memory = MemorySaver()

# Add nodes
workflow.add_node("analyze", analyze_query)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("response", generate_answer)

# Create edges

workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "response")
workflow.add_edge("response", END)

# Set the entry point
workflow.set_entry_point("analyze")

# Compile the graph
EvaluatorAgent = workflow.compile(checkpointer=memory)
