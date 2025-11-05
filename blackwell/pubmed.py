"""
PubMed Research Module
A streamlined library for researching treatment options from PubMed given diagnosis hypotheses.
"""

import requests
import time
from typing import List, Dict, Optional, Any
from xml.etree import ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class PubMedArticle:
    """Represents a PubMed article with relevant information."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: str
    doi: Optional[str] = None
    url: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert article to dictionary format."""
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "journal": self.journal,
            "publication_date": self.publication_date,
            "doi": self.doi,
            "url": self.url
        }
    
    def get_summary(self) -> str:
        """Get a formatted summary of the article."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        
        return f"""
**{self.title}**
Authors: {authors_str}
Journal: {self.journal} ({self.publication_date})
PMID: {self.pmid}
{f'DOI: {self.doi}' if self.doi else ''}
URL: {self.url}

Abstract: {self.abstract[:500]}{'...' if len(self.abstract) > 500 else ''}
"""


class PubMedClient:
    """Client for interacting with PubMed API."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None, tool: str = "blackwell"):
        """
        Initialize PubMed client.
        
        Args:
            email: Your email (recommended by NCBI)
            api_key: NCBI API key for higher rate limits (optional)
            tool: Tool name for API identification
        """
        self.email = email
        self.api_key = api_key
        self.tool = tool
        self.rate_limit = 0.34 if not api_key else 0.1  # seconds between requests
        self.last_request_time = 0
    
    def _wait_for_rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _build_params(self, **kwargs) -> Dict[str, str]:
        """Build common parameters for API requests."""
        params = {}
        if self.email:
            params['email'] = self.email
        if self.api_key:
            params['api_key'] = self.api_key
        if self.tool:
            params['tool'] = self.tool
        params.update(kwargs)
        return params
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        years_back: Optional[int] = None,
        sort: str = "relevance"
    ) -> List[str]:
        """
        Search PubMed and return list of PMIDs.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            years_back: Limit to articles from last N years (None for all time)
            sort: Sort order ('relevance' or 'date')
            
        Returns:
            List of PMIDs
        """
        self._wait_for_rate_limit()
        
        # Add date filter if specified
        if years_back:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365)
            date_filter = f" AND {start_date.year}/{start_date.month}/{start_date.day}:{end_date.year}/{end_date.month}/{end_date.day}[dp]"
            query = query + date_filter
        
        params = self._build_params(
            db="pubmed",
            term=query,
            retmax=max_results,
            retmode="json",
            sort=sort
        )
        
        try:
            response = requests.get(f"{self.BASE_URL}esearch.fcgi", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []
    
    def fetch_details(self, pmids: List[str]) -> List[PubMedArticle]:
        """
        Fetch detailed information for given PMIDs.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of PubMedArticle objects
        """
        if not pmids:
            return []
        
        self._wait_for_rate_limit()
        
        params = self._build_params(
            db="pubmed",
            id=",".join(pmids),
            retmode="xml"
        )
        
        try:
            response = requests.get(f"{self.BASE_URL}efetch.fcgi", params=params, timeout=30)
            response.raise_for_status()
            
            return self._parse_articles(response.text)
        except Exception as e:
            print(f"Error fetching article details: {e}")
            return []
    
    def _parse_articles(self, xml_text: str) -> List[PubMedArticle]:
        """Parse XML response into PubMedArticle objects."""
        articles = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article_elem in root.findall(".//PubmedArticle"):
                try:
                    article = self._parse_single_article(article_elem)
                    if article:
                        articles.append(article)
                except Exception as e:
                    print(f"Error parsing individual article: {e}")
                    continue
            
            return articles
        except Exception as e:
            print(f"Error parsing XML: {e}")
            return []
    
    def _parse_single_article(self, article_elem) -> Optional[PubMedArticle]:
        """Parse a single article element."""
        try:
            # PMID
            pmid_elem = article_elem.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            # Title
            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title available"
            
            # Abstract
            abstract_parts = article_elem.findall(".//AbstractText")
            abstract = " ".join([
                (elem.text or "") for elem in abstract_parts
            ]) if abstract_parts else "No abstract available"
            
            # Authors
            author_elems = article_elem.findall(".//Author")
            authors = []
            for author in author_elems:
                lastname = author.find("LastName")
                forename = author.find("ForeName")
                if lastname is not None:
                    name = lastname.text or ""
                    if forename is not None and forename.text:
                        name = f"{forename.text} {name}"
                    authors.append(name)
            
            # Journal
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "Unknown journal"
            
            # Publication date
            pub_date = article_elem.find(".//PubDate")
            pub_date_str = "Unknown date"
            if pub_date is not None:
                year = pub_date.find("Year")
                month = pub_date.find("Month")
                day = pub_date.find("Day")
                
                parts = []
                if year is not None and year.text:
                    parts.append(year.text)
                if month is not None and month.text:
                    parts.append(month.text)
                if day is not None and day.text:
                    parts.append(day.text)
                
                if parts:
                    pub_date_str = " ".join(parts)
            
            # DOI
            doi_elem = article_elem.find(".//ArticleId[@IdType='doi']")
            doi = doi_elem.text if doi_elem is not None else None
            
            # URL
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            
            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=pub_date_str,
                doi=doi,
                url=url
            )
        except Exception as e:
            print(f"Error parsing article: {e}")
            return None


class TreatmentResearcher:
    """High-level interface for researching treatments for diagnoses."""
    
    def __init__(self, email: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize treatment researcher.
        
        Args:
            email: Your email (recommended by NCBI)
            api_key: NCBI API key for higher rate limits (optional)
        """
        self.client = PubMedClient(email=email, api_key=api_key)
    
    def research_treatment(
        self,
        diagnosis: str,
        max_results: int = 10,
        years_back: int = 5,
        include_reviews: bool = True,
        include_clinical_trials: bool = True
    ) -> Dict[str, Any]:
        """
        Research treatment options for a given diagnosis.
        
        Args:
            diagnosis: The diagnosis hypothesis (e.g., "psoriasis", "inverse psoriasis")
            max_results: Maximum number of articles to retrieve
            years_back: Limit to articles from last N years
            include_reviews: Include systematic reviews and meta-analyses
            include_clinical_trials: Include clinical trials
            
        Returns:
            Dictionary with treatment information and articles
        """
        # Build search query
        query_parts = [f"{diagnosis}[Title/Abstract]", "treatment[Title/Abstract]"]
        
        filters = []
        if include_reviews:
            filters.append("systematic review[Publication Type]")
            filters.append("meta-analysis[Publication Type]")
        if include_clinical_trials:
            filters.append("clinical trial[Publication Type]")
            filters.append("randomized controlled trial[Publication Type]")
        
        if filters:
            filter_query = " OR ".join([f"({f})" for f in filters])
            query_parts.append(f"({filter_query})")
        
        query = " AND ".join(query_parts)
        
        # Search PubMed
        pmids = self.client.search(
            query=query,
            max_results=max_results,
            years_back=years_back,
            sort="relevance"
        )
        
        # Fetch article details
        articles = self.client.fetch_details(pmids)
        
        return {
            "diagnosis": diagnosis,
            "query": query,
            "total_results": len(articles),
            "articles": [article.to_dict() for article in articles],
            "article_objects": articles  # For programmatic access
        }
    
    def research_specific_treatment(
        self,
        diagnosis: str,
        treatment: str,
        max_results: int = 10,
        years_back: int = 5
    ) -> Dict[str, Any]:
        """
        Research effectiveness of a specific treatment for a diagnosis.
        
        Args:
            diagnosis: The diagnosis hypothesis
            treatment: Specific treatment to research (e.g., "methotrexate", "phototherapy")
            max_results: Maximum number of articles to retrieve
            years_back: Limit to articles from last N years
            
        Returns:
            Dictionary with treatment information and articles
        """
        query = f"{diagnosis}[Title/Abstract] AND {treatment}[Title/Abstract] AND (treatment[Title/Abstract] OR therapy[Title/Abstract])"
        
        pmids = self.client.search(
            query=query,
            max_results=max_results,
            years_back=years_back,
            sort="relevance"
        )
        
        articles = self.client.fetch_details(pmids)
        
        return {
            "diagnosis": diagnosis,
            "treatment": treatment,
            "query": query,
            "total_results": len(articles),
            "articles": [article.to_dict() for article in articles],
            "article_objects": articles
        }
    
    def compare_treatments(
        self,
        diagnosis: str,
        treatments: List[str],
        max_results_per_treatment: int = 5,
        years_back: int = 5
    ) -> Dict[str, Any]:
        """
        Compare multiple treatments for a diagnosis.
        
        Args:
            diagnosis: The diagnosis hypothesis
            treatments: List of treatments to compare
            max_results_per_treatment: Maximum articles per treatment
            years_back: Limit to articles from last N years
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        for treatment in treatments:
            results[treatment] = self.research_specific_treatment(
                diagnosis=diagnosis,
                treatment=treatment,
                max_results=max_results_per_treatment,
                years_back=years_back
            )
        
        return {
            "diagnosis": diagnosis,
            "treatments_compared": treatments,
            "results": results
        }
    
    def get_treatment_guidelines(
        self,
        diagnosis: str,
        max_results: int = 5,
        years_back: int = 3
    ) -> Dict[str, Any]:
        """
        Search for treatment guidelines and recommendations.
        
        Args:
            diagnosis: The diagnosis hypothesis
            max_results: Maximum number of guidelines to retrieve
            years_back: Limit to guidelines from last N years
            
        Returns:
            Dictionary with guidelines and articles
        """
        query = f"{diagnosis}[Title/Abstract] AND (guideline[Publication Type] OR practice guideline[Publication Type] OR consensus[Title/Abstract] OR recommendation[Title/Abstract])"
        
        pmids = self.client.search(
            query=query,
            max_results=max_results,
            years_back=years_back,
            sort="date"
        )
        
        articles = self.client.fetch_details(pmids)
        
        return {
            "diagnosis": diagnosis,
            "query": query,
            "total_results": len(articles),
            "articles": [article.to_dict() for article in articles],
            "article_objects": articles
        }
    
    def format_results_for_llm(self, research_results: Dict[str, Any]) -> str:
        """
        Format research results into a string suitable for LLM consumption.
        
        Args:
            research_results: Results from any research method
            
        Returns:
            Formatted string with article information
        """
        output = []
        output.append(f"# Treatment Research: {research_results.get('diagnosis', 'Unknown')}\n")
        
        if 'treatment' in research_results:
            output.append(f"Specific Treatment: {research_results['treatment']}\n")
        
        output.append(f"Total Articles Found: {research_results.get('total_results', 0)}\n")
        output.append("=" * 80)
        output.append("")
        
        articles = research_results.get('article_objects', [])
        
        for i, article in enumerate(articles, 1):
            output.append(f"\n## Article {i}\n")
            output.append(article.get_summary())
            output.append("-" * 80)
        
        return "\n".join(output)


# Convenience functions for quick access
def quick_treatment_search(
    diagnosis: str,
    max_results: int = 10,
    years_back: int = 5,
    email: Optional[str] = None
) -> str:
    """
    Quick function to search for treatment options and return formatted results.
    
    Args:
        diagnosis: The diagnosis hypothesis
        max_results: Maximum number of articles
        years_back: Limit to recent years
        email: Your email (optional but recommended)
        
    Returns:
        Formatted string with research results
    """
    researcher = TreatmentResearcher(email=email)
    results = researcher.research_treatment(
        diagnosis=diagnosis,
        max_results=max_results,
        years_back=years_back
    )
    return researcher.format_results_for_llm(results)


def search_specific_treatment(
    diagnosis: str,
    treatment: str,
    max_results: int = 10,
    email: Optional[str] = None
) -> str:
    """
    Quick function to search for a specific treatment and return formatted results.
    
    Args:
        diagnosis: The diagnosis hypothesis
        treatment: Specific treatment to research
        max_results: Maximum number of articles
        email: Your email (optional but recommended)
        
    Returns:
        Formatted string with research results
    """
    researcher = TreatmentResearcher(email=email)
    results = researcher.research_specific_treatment(
        diagnosis=diagnosis,
        treatment=treatment,
        max_results=max_results
    )
    return researcher.format_results_for_llm(results)
