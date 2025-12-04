from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import logging

############### CONFIG FLAGS ############
LOCAL_LLMS = False  # Set to True to use local LLMs (Ollama) instead of Gemini
DB_PATH = "database/blackwell"  # Path to the database
DB_COLLECTION = "medline_vector_store"  # Collection name in the database
DATA_FOLDER = "data/"  # Folder containing data files
QUOTA_AGENT_LIMIT = "2-15"
QUOTA_RATE = 10  # RPM rate limit for Gemini API calls
#########################################
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='evaluation/blackwell.log',
                    filemode='a')
logger = logging.getLogger("blackwell")
ACCEPTED_EXTENSIONS = [
    "pdf",
    "txt",  # Uncommented to accept txt files
    "csv",  # Added to accept csv files
    "xlsx",  # Added to accept xlsx files
]

load_dotenv()

# Gemini LLM
if LOCAL_LLMS:
    # Ollama LLM
    fast_model = ChatOllama(
        model="qwen3:4b",
        temperature=0,
        max_tokens=128000,
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