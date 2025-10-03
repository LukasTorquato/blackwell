# Tales - AI Document Assistant

<img width="2490" height="1016" alt="image" src="https://github.com/user-attachments/assets/94887f74-7933-44cf-8a8a-511b37adb1c7" />

Tales is an AI-powered document assistant that utilizes a Retrieval-Augmented Generation (RAG) agent to provide insightful answers from your documents and generate presentations. This agentic system is built with LangChain and LangGraph, and it can leverage both local LLMs via Ollama and powerful models like Google's Gemini.

At its core, Tales indexes your documents (supporting PDF, TXT, CSV, and XLSX) into a ChromaDB vector store. When you ask a question, the RAG agent retrieves the most relevant information from your documents to formulate a comprehensive answer. The agent is designed to be conversational, allowing you to ask follow-up questions and delve deeper into the subject matter.

One of the standout features of Tales is its ability to generate PowerPoint presentations directly from your conversation. Using the Model Context Protocol (MCP), the agent can take the context of your discussion and automatically create a presentation, saving you time and effort.

The project also includes a user-friendly web interface for interacting with the agent. You can manage your documents, switch between conversations, and initiate the PowerPoint generation process, all from your browser. For developers, Tales provides a robust evaluation system to measure the performance of the RAG agent and the quality of the generated presentations.

#### Key technologies used in this project include:

- LangChain & LangGraph: For building the core agentic RAG pipeline.
- Google Gemini & Ollama: Providing the language models that power the agent.
- ChromaDB: For creating and managing the vector store.
- MCP (Model Context Protocol): To enable the generation of PowerPoint presentations.
- Flask: For the web application that serves the user interface.

### Installing Ollama and Pulling Models

- Go to https://ollama.com/download
- Download Ollama and install
- Run Ollama, open CMD and run:
  - ollama pull llama3.2:3b (Regular model)
  - ollama pull deepseek-r1 (Reasoning model)
  - ollama pull nomic-embed-text (Embbeding model)
- Get your Google Gemini API Key at https://aistudio.google.com/

### Installing dependencies

- Run pip install .

### Organizing Data

- Create a folder in the base directory named "data"
- Put in all wanted pdf, csv files in that folder

### Running

- CLI: Run app.py
- Web UI: Start the web server and open the browser
  - python web_app.py
  - Visit http://localhost:8000
  - The web interface provides:
    - A conversation selector on the left
    - A chat interface in the center
    - A file viewer on the right showing documents in your vector database
  - You can create new conversations, continue existing ones, and generate PowerPoint presentations.

### Evaluation System

The project includes a comprehensive evaluation system for both the RAG agent and PowerPoint generation capabilities.

#### Command-line Evaluation

```bash
# Evaluate a single query
python evaluate.py --query "What are the key concepts in information theory?"

# Generate and evaluate a PowerPoint presentation for a query
python evaluate.py --query "Explain the main principles of information theory" --ppt

# Run batch evaluation on predefined queries
python evaluate.py --batch

# Run batch evaluation on custom queries from a file
python evaluate.py --batch --input my_queries.txt --output my_results.json
```

#### Evaluation Metrics

- **RAG Agent Metrics**:

  - Response Time: Time taken to generate the response
  - Context Relevance: How relevant the retrieved documents are to the query (0-10)
  - Answer Correctness: How factually correct the answer is based on the context (0-10)
  - Answer Completeness: How completely the answer addresses the query (0-10)
  - Hallucination Score: Degree of hallucination in the response (0-10, lower is better)
  - Number of Documents Retrieved: Count of documents used as context
  - Research Iterations: Number of research iterations performed
