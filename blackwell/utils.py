from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict
import json
import os
import requests
from bs4 import BeautifulSoup

from blackwell.pubmed import PubMedClient


def export_json(data: dict, filename: str) -> None:
    """
    Export data to a JSON file.

    Args:
        data: Data to export
        filename: Name of the file to save the data
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def get_available_docs(folder_path, extensions) -> list:
    """
    Search for documents in the specified folder and return their names and paths.

    Args:
        folder_path: Path to the folder to search (defaults to 'data' in project root)
        extensions: List of file extensions to include (e.g., ['pdf', 'txt'])
                  If None, includes all files

    Returns:
        List of dictionaries containing document name and path
    """

    # Check if directory exists
    if not os.path.isdir(folder_path):
        print(f"Warning: Directory not found at {folder_path}")
        return []

    documents = []

    # Walk through directory and subdirectories
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if file has one of the specified extensions
            if extensions is None or any(
                file.lower().endswith(f".{ext.lower()}") for ext in extensions
            ):
                documents.append(os.path.join(root, file))
                # documents.append({"name": file, "path": os.path.join(root, file)})

    return documents


def messages_to_json(messages: list) -> list:

    msgs = []
    for msg in messages:
        if not isinstance(msg, (HumanMessage, AIMessage)):
            raise ValueError("Invalid message type. Must be HumanMessage or AIMessage.")
        if isinstance(msg, HumanMessage):
            msgs.append([msg.content, {"role": "user"}])
        else:
            msgs.append([msg.content, {"role": "assistant"}])
    msgs_json = json.dumps(msgs)
    return msgs_json


def json_to_messages(json_str: str) -> list:
    msgs = json.loads(json_str)
    messages = []
    for msg in msgs:
        if msg[1]["role"] == "user":
            messages.append(
                HumanMessage(content=msg[0], additional_kwargs={"role": "user"})
            )
        else:
            messages.append(
                AIMessage(content=msg[0], additional_kwargs={"role": "assistant"})
            )
    return messages


def print_state_messages(state, context=False, metadata=False):
    if context:
        print("\n" + "=" * 25 + " CONTEXT " + "=" * 25)
        for c in state["context"]:
            print(f"Document: {c.metadata['source']}")
            print(c.page_content)
            print("=" * 50)

    print("\n" + "=" * 25 + " MESSAGES " + "=" * 25)
    for message in state["messages"]:
        message.pretty_print()

    if metadata:
        print("\n" + "=" * 25 + " USAGE " + "=" * 25)
        print(state["messages"][-1].usage_metadata)


def format_references(references: List[Dict]) -> str:
    """Format references into a structured text format for the final report."""
    if not references:
        return """### References 
        No references were consulted during this analysis."""
    
    formatted = ["### References\n"]
    
    # Group by type
    rag_refs = [r for r in references if r.get("type") == "RAG"]
    pubmed_refs = [r for r in references if r.get("type") == "PubMed"]
    
    # Format RAG references (local knowledge base)
    if rag_refs:
        formatted.append("**Knowledge Base References:**")
        # Remove duplicates by source+page
        seen = set()
        for ref in rag_refs:
            if ref['reference'] not in seen:
                seen.add(ref['reference'])
                formatted.append(f"- {ref['reference']}")
        formatted.append("")
    
    # Format PubMed references
    if pubmed_refs:
        formatted.append("**PubMed Articles:**")
        # Remove duplicates by PMID
        seen = set()
        for ref in pubmed_refs:
            if ref['reference'] not in seen:
                seen.add(ref['reference'])
                formatted.append(f"- {ref['reference']}")
        formatted.append("")
    
    return "\n".join(formatted)


def fetch_medical_website_content(url: str, max_chars: int = 15000) -> dict:
    """
    Fetch and extract content from various medical websites (MedlinePlus, Mayo Clinic, CDC, etc.).
    This function intelligently identifies the main content area across different website structures.
    
    Supports:
        - MedlinePlus (medlineplus.gov)
        - Mayo Clinic (mayoclinic.org)
        - FamilyDoctor (familydoctor.org)
        - CDC (cdc.gov)
        - NIH (nih.gov)
        - WebMD (webmd.com)
        - And other medical information websites
    
    Args:
        url: The URL to fetch
        max_chars: Maximum characters to return (default 15000)
    
    Returns:
        dict with keys:
            - success (bool): Whether the fetch was successful
            - content (str): The extracted text content
            - title (str): The page title
            - url (str): The original URL
            - source (str): The website domain
            - error (str, optional): Error message if success is False
            
    Example:
        >>> result = fetch_medical_website_content("https://www.mayoclinic.org/diseases-conditions/...")
        >>> if result['success']:
        >>>     print(f"Source: {result['source']}")
        >>>     print(result['title'])
        >>>     print(result['content'][:500])
    """
    try:
        # Set comprehensive headers to mimic a real browser and avoid bot detection
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        
        # Create a session for better connection handling
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(url, timeout=15, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract domain for identification
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else "Untitled"
        
        # Remove unwanted elements
        unwanted_tags = ['script', 'style', 'nav', 'header', 'footer', 'aside', 
                        'iframe', 'noscript', 'button', 'form']
        for element in soup(unwanted_tags):
            element.decompose()
        
        # Remove elements by class/id that typically contain navigation/ads
        unwanted_selectors = [
            '.navigation', '.nav', '.menu', '.sidebar', '.advertisement', 
            '.ad', '.social-share', '.related-links', '.breadcrumb',
            '#navigation', '#sidebar', '#comments', '.comments'
        ]
        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()
        
        # Define content selectors for different websites
        content_selectors = {
            'medlineplus.gov': [
                '#mplus-content', 
                '#topic-summary', 
                '.main-content',
                'article',
                '#article'
            ],
            'mayoclinic.org': [
                '.content',
                'article',
                '.main-content',
                '[role="main"]',
                '#main-content'
            ],
            'familydoctor.org': [
                '.content-area',
                'article',
                '.post-content',
                '.entry-content',
                'main'
            ],
            'cdc.gov': [
                '#content',
                'article',
                '.syndicate',
                '.content-area',
                'main'
            ],
            'nih.gov': [
                '#content',
                'article',
                '.main-content',
                '[role="main"]',
                'main'
            ],
            'webmd.com': [
                'article',
                '.article-body',
                '.content',
                'main'
            ],
            'kidshealth.org': [
                'article',
                '.article-body',
                '#content',
                'main'
            ],
            'healthline.com': [
                'article',
                '.article-body',
                '#article-content',
                'main'
            ]
        }
        
        # Find appropriate selectors for this domain
        selectors = []
        for site_domain, site_selectors in content_selectors.items():
            if site_domain in domain:
                selectors = site_selectors
                break
        
        # If no specific selectors found, use generic ones
        if not selectors:
            selectors = ['article', 'main', '[role="main"]', '.content', '#content', '.main-content']
        
        # Try to find main content area
        main_content = None
        for selector in selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # Use full body if no main content found
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        clean_text = '\n'.join(lines)
        
        # Remove duplicate lines (common in web scraping)
        seen_lines = set()
        unique_lines = []
        for line in lines:
            if line not in seen_lines or len(line) > 100:  # Keep long lines even if duplicate
                unique_lines.append(line)
                seen_lines.add(line)
        clean_text = '\n'.join(unique_lines)
        
        # Limit length
        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars] + "\n\n... (content truncated)"
        
        return {
            'success': True,
            'content': clean_text,
            'title': title,
            'url': url,
            'source': domain
        }
    except requests.exceptions.Timeout:
        return {
            'success': False,
            'content': '',
            'title': '',
            'url': url,
            'source': '',
            'error': 'Request timeout - website took too long to respond'
        }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'content': '',
            'title': '',
            'url': url,
            'source': '',
            'error': f'Request error: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'content': '',
            'title': '',
            'url': url,
            'source': '',
            'error': f'Unexpected error: {str(e)}'
        }
