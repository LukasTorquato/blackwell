from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.runnables.config import RunnableConfig
from dotenv import load_dotenv
import logging

############### CONFIG FLAGS ############
LOCAL_LLMS = False  # Set to True to use local LLMs (Ollama) instead of Gemini
DB_PATH = "database/blackwell"  # Path to the database
DB_COLLECTION = "medline_vector_store"  # Collection name in the database
DATA_FOLDER = "data/"  # Folder containing data files
DOCS_RETRIEVED = 20  # Number of documents to retrieve for each query
RUNNABLE_CONFIG = RunnableConfig(recursion_limit=100) # Increase default recursion limit for agentic LLM tool recursion
QUOTA_AGENT_LIMIT = "3-20"
QUOTA_RATE = 0.1  # RPM rate limit for Gemini API calls
#########################################
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='evaluation/blackwell.log',
                    filemode='a')
logger = logging.getLogger("blackwell")
ACCEPTED_EXTENSIONS = [
    "pdf",
    "txt",  # Uncommented to accept txt files
    # "docx",  # Added to accept docx files
    # "pptx",  # Added to accept pptx files
    "csv",  # Added to accept csv files
    "xlsx",  # Added to accept xlsx files
]

load_dotenv()

# Gemini LLM
if LOCAL_LLMS:
    # Ollama LLM
    llm = ChatOllama(
        model="qwen3:4b",
        temperature=0,
        max_tokens=4096,
        streaming=True,
        callbacks=[],
    )

else:
    fast_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        max_tokens=900000,
        timeout=None,
        max_retries=1,
    )

    pro_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        max_tokens=900000,
        timeout=None,
        max_retries=1,
    )

    agent_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_tokens=900000,
        timeout=None,
        max_retries=1,
    )

# Gemini Embeddings
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    temperature=0,
    max_tokens=2048,
    max_retries=2,
    timeout=None,
)
