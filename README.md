# Blackwell - AI Clinical Assistant
<img width="2642" height="1155" alt="image" src="https://github.com/user-attachments/assets/b9fd78f4-682d-4851-9dcc-f8f2d02f7927" />

Blackwell is a Multi Agent Clinical Decision Support System that pragmatically investigate the Patient's (user) complaints via an Anamnesis procedure that rapidly ask all the necessary questions for proper screening and reporting. 
After Anamnesis is complete, the multi-agent evaluator utilizes the capabilities of Retrieval-Augmented Generation (RAG) and Web Crawling Agents, to provide insightful hypothesis over the patient's complaints and history, which then is passed to a treatment-focused agent that investigates the state-of-the-art treatment's procedure, efficacy available at PubMed's large web database. This agentic system is built with LangChain and LangGraph, and it can be powered by local LLMs via Ollama or powerful models like Google's Gemini.

At its core, Blackwell has Medline complete database indexed into a ChromaDB vector store. The PubMed's agent goes deep through PubMed's vast library in search for treatment options, efficacy and guidelines.

The project also includes a user-friendly web interface for interacting with the agent where the patient can interact with the agent and receive a detailed report of the agent's finding and hypothesis, which then can be presented to Clinical Practitioner. The Application does not hold patient information and conversation to protect their personal data.

#### Key technologies used in this project include:

- LangChain & LangGraph: For building the core multi-agent pipeline.
- Google Gemini & Ollama: Providing the language models that power the agent.
- ChromaDB: For creating and managing the vector store.
- MCP (Model Context Protocol): To enable the generation of Word Documents.
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

### Running

- CLI: Run app.py
- Web UI: Start the web server and open the browser
  - uvicorn web_app:app --reload
  - Visit http://localhost:8000
  - The web interface provides:
    - A chat box to interact with the agent
    - A charming UI for presenting the reports

### Evaluation System

The project includes a comprehensive evaluation system with a special AI Patient to simulate patient's history based on the Prognosis Disease Symptoms Dataset available at (https://www.kaggle.com/datasets/noeyislearning/disease-prediction-based-on-symptoms)

#### Evaluation Metrics

