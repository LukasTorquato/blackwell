from typing import List, TypedDict
import os
# LangChain imports
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AnyMessage,
    RemoveMessage,
)

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from blackwell.config import *
from blackwell.prompts import *
from blackwell.document_processer import build_retriever
from blackwell.utils import fetch_medical_website_content
from blackwell.pubmed_tools import PUBMED_TOOLS, initialize_pubmed_tools


##################### Graph Compiling Script #####################
# This script compiles the LangGraph graph and the vector store for the RAG pipeline.
class GraphState(TypedDict):
    # Type for the state of the retrieval and query graph

    context: List[Document]
    t_run: int
    anamnesis_report: AnyMessage
    research_report: str
    query: AnyMessage  # Improved query for vector similarity search
    hypothesis_report: AnyMessage
    treatment_report: AnyMessage
    final_report: AnyMessage


# Define the nodes in the graph
def analyze_query(state: GraphState) -> GraphState:
    # Analyze user query to improve vector similarity search
    state["t_run"] += 1
    if state["t_run"] == 1:
        print("Proposing Hypothesis query...")
        state["query"] = evaluator_llm.invoke([hypothesis_rag_prompt, state["anamnesis_report"]])
    else:
        print("Proposing Treatment query...")
        state["query"] = evaluator_llm.invoke([treatment_rag_prompt, state["hypothesis_report"], state["anamnesis_report"]])
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

def web_crawl(state: GraphState) -> GraphState:
    print("Web crawling for additional context...")
    documents = state["context"]
    context_str = "[RAG_CONTEXT]:\n" + "\n\n".join(doc.page_content for doc in documents)
    result = evaluator_llm.invoke([web_crawl_prompt] + [state["query"]] + [context_str])
    print(result.content)
    links = result.content.split(',')
    for link in links:
        context_str = context_str.replace(link, fetch_medical_website_content(link)["content"])

    state["research_report"] = context_str
    return state

def pubmed_search(state: GraphState) -> GraphState:
    # Placeholder for PubMed search node
    print("Performing PubMed search for additional context...")
    result = pubmed_agent.invoke({"messages": [{"role": "user", "content": state["query"].content}]})
    print(result['messages'][-1].content)
    state["research_report"] += result['messages'][-1].content

    return state

def generate_hypothesis(state: GraphState) -> GraphState:
    # Generate a hypothesis using retrieved context
    research = HumanMessage(content=state["research_report"])
    state["hypothesis_report"] = evaluator_llm.invoke([hypothesis_eval_prompt] + [state["anamnesis_report"]] + [research])

    return state

def generate_treatment(state: GraphState) -> GraphState:
    # Generate a treatment plan using retrieved context
    research = HumanMessage(content=state["research_report"])
    print("Generating treatment plan...")
    state["treatment_report"] = evaluator_llm.invoke([treatment_eval_prompt] + [state["hypothesis_report"]] + [state["anamnesis_report"]] + [research])
    print("Generating final report...")
    state["final_report"] = [evaluator_llm.invoke([final_report_prompt] + [state["anamnesis_report"]] + [state["hypothesis_report"]] + [state["treatment_report"]] + [research])]

    return state

def run_count(state: GraphState):
    if state["t_run"] == 1:
        return "hypothesis"
    else:
        return "pubmed"


# Build the vector store
vector_store = build_retriever()
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
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("crawl", web_crawl)
workflow.add_node("hypothesis", generate_hypothesis)
workflow.add_node("treatment", generate_treatment)
workflow.add_node("pubmed", pubmed_search)  # Placeholder for PubMed search node

# Create edges
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "crawl")
workflow.add_edge("hypothesis", "analyze")
workflow.add_conditional_edges("crawl", run_count)
workflow.add_edge("pubmed", "treatment")
workflow.add_edge("treatment", END)

# Set the entry point
workflow.set_entry_point("analyze")

# Compile the graph
EvaluatorAgent = workflow.compile(checkpointer=memory)
