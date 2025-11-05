"""
RAG Tools for LangChain Agents

This module provides ready-to-use LangChain tools for integrating vector store retrieval
and web crawling into your clinical decision support agents.
"""

from typing import Optional, List
from langchain_core.tools import StructuredTool
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from blackwell.utils import fetch_medical_website_content


# Global vector store instance
_vector_store = None


def initialize_rag_tools(vector_store):
    """
    Initialize the RAG tools with a vector store.
    
    Args:
        vector_store: The ChromaDB vector store instance
    """
    global _vector_store
    _vector_store = vector_store


def get_vector_store():
    """Get the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        raise ValueError("Vector store not initialized. Call initialize_rag_tools() first.")
    return _vector_store


# Pydantic models for tool arguments
class RetrieveDocumentsInput(BaseModel):
    """Input schema for retrieve_documents tool."""
    query: str = Field(description="The search query to find relevant medical documents in the vector database")
    k: int = Field(default=10, description="Number of documents to retrieve", ge=1, le=20)


class WebCrawlMedlineInput(BaseModel):
    """Input schema for web_crawl_medline tool."""
    urls: str = Field(description="Comma-separated list of medical website URLs to crawl (MedlinePlus, Mayo Clinic, CDC, etc.)")


# Tool functions
def _retrieve_documents_func(query: str, k: int = 10) -> str:
    """
    Retrieve relevant medical documents from the vector database using similarity search.
    
    This tool searches through the local knowledge base of medical documents (PDFs, texts, etc.)
    that have been indexed in the vector database. Use this to find relevant information from
    your curated medical literature collection.
    
    Args:
        query: The search query describing the medical information needed
        k: Number of most relevant documents to retrieve (default: 10)
        
    Returns:
        Formatted string containing the retrieved document contents with source metadata
    """
    try:
        vector_store = get_vector_store()
        
        if not query or query.strip() == "":
            return "Error: Query cannot be empty. Please provide a specific search query."
        
        # Perform similarity search
        retrieved_docs: List[Document] = vector_store.similarity_search(query, k=k)
        
        if not retrieved_docs:
            return f"No documents found for query: '{query}'. Try rephrasing or broadening your search."
        
        # Format results
        formatted_results = []
        formatted_results.append(f"Retrieved {len(retrieved_docs)} documents for query: '{query}'\n")
        formatted_results.append("=" * 80)
        
        for idx, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get('source', 'Unknown source')
            page = doc.metadata.get('page', 'N/A')
            
            formatted_results.append(f"\n[Document {idx}/{len(retrieved_docs)}]")
            formatted_results.append(f"Source: {source} (Page: {page})")
            formatted_results.append("-" * 80)
            formatted_results.append(doc.page_content)
            formatted_results.append("=" * 80)
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"


def _web_crawl_medline_func(urls: str) -> str:
    """
    Crawl medical websites (MedlinePlus, Mayo Clinic, CDC, etc.) to extract relevant health information.
    
    This tool fetches and extracts content from trusted medical information websites. It intelligently
    identifies and extracts the main medical content while filtering out navigation, ads, and other
    non-essential elements.
    
    Supports:
    - MedlinePlus (medlineplus.gov)
    - Mayo Clinic (mayoclinic.org)
    - CDC (cdc.gov)
    - NIH (nih.gov)
    - WebMD (webmd.com)
    - FamilyDoctor (familydoctor.org)
    
    Args:
        urls: Comma-separated list of URLs to crawl (e.g., "https://medlineplus.gov/..., https://www.mayoclinic.org/...")
        
    Returns:
        Formatted string containing the extracted content from each website with metadata
    """
    try:
        # Parse the comma-separated URLs
        url_list = [url.strip() for url in urls.split(',') if url.strip()]
        
        if not url_list:
            return "Error: No valid URLs provided. Please provide at least one URL."
        
        formatted_results = []
        formatted_results.append(f"Web Crawling {len(url_list)} medical websites...\n")
        formatted_results.append("=" * 80)
        
        for idx, url in enumerate(url_list, 1):
            formatted_results.append(f"\n[Website {idx}/{len(url_list)}]")
            formatted_results.append(f"URL: {url}")
            formatted_results.append("-" * 80)
            
            # Fetch the website content
            result = fetch_medical_website_content(url)
            
            if result.get('success', False):
                formatted_results.append(f"Title: {result.get('title', 'N/A')}")
                formatted_results.append(f"Source: {result.get('source', 'N/A')}")
                formatted_results.append(f"\nContent:")
                formatted_results.append(result.get('content', 'No content extracted'))
            else:
                error_msg = result.get('error', 'Unknown error')
                formatted_results.append(f"Error: Failed to fetch content - {error_msg}")
            
            formatted_results.append("=" * 80)
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error during web crawling: {str(e)}"


# Create structured tools
retrieve_documents = StructuredTool.from_function(
    func=_retrieve_documents_func,
    name="retrieve_documents",
    description=(
        "Retrieve relevant medical documents from the local vector database using similarity search. "
        "Use this tool to search through your curated collection of medical literature (PDFs, texts, etc.). "
        "Provide a specific query describing the medical information you need, and optionally specify "
        "the number of documents to retrieve (default: 10, max: 20)."
    ),
    args_schema=RetrieveDocumentsInput,
    return_direct=False
)

web_crawl_medline = StructuredTool.from_function(
    func=_web_crawl_medline_func,
    name="web_crawl_medline",
    description=(
        "Crawl trusted medical websites to extract health information. "
        "Supports MedlinePlus, Mayo Clinic, CDC, NIH, WebMD, and FamilyDoctor. "
        "Provide a comma-separated list of URLs to crawl. The tool will extract the main medical content "
        "while filtering out navigation, ads, and other non-essential elements. "
        "Example: 'https://medlineplus.gov/..., https://www.mayoclinic.org/...'"
    ),
    args_schema=WebCrawlMedlineInput,
    return_direct=False
)


# Export the tools list
RAG_TOOLS = [retrieve_documents, web_crawl_medline]
