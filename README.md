# Multilingual Culture & Language Tutor (RAG System)

A Retrieval-Augmented Generation (RAG) conversational pipeline designed to explain complex idioms, grammar rules, and cultural nuances across multiple global languages (e.g., Bengali, Spanish, Japanese, French, Arabic, German, Mandarin).

This tool acts as an interactive linguistic and cultural professor. It grounds its answers in specific "textbook" knowledge rather than relying purely on a Large Language Model's general (and sometimes hallucinated or overly broad) knowledge base. 

## Features
- **Retrieval-Augmented Generation (RAG)**: Retrieves specific cultural context from a curated dataset before generating an answer.
- **Offline Vector Store**: Uses a completely local ChromaDB database and a lightweight Hugging Face embedding model (`all-MiniLM-L6-v2`) to keep semantic search fast and private.
- **Advanced LLM Generation**: Integrates Google's Gemini 2.5 Flash model (via LangChain) to synthesize retrieved data into empathetic, beautifully formatted, and culturally accurate explanations.
- **Dynamic CLI**: An interactive command-line interface to easily query phrases and see the generated explanation.

## Project Structure
- `data.json`: The "textbook" knowledge base containing structural examples of idioms, customs, and grammar rules.
- `rag.py`: Contains the `MultilingualRAG` class handling embedding, ChromaDB vector indexing, and the generation chain.
- `main.py`: The CLI application entry point.
- `test_rag.py`: An offline verification script to test vector DB ingestion and semantic retrieval without needing an LLM key.

## Requirements
To run this application, you need Python 3 installed and the following dependencies (listed in `requirements.txt`):
- `langchain`
- `langchain-huggingface`
- `langchain-google-genai`
- `chromadb`
- `langchain-chroma`
- `sentence-transformers`
- `python-dotenv`

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/Himanish-18/multilingual-tutor.git
   cd multilingual-tutor
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # On Windows
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   
   # On macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your API Key**:
   Create a file named `.env` in the root directory and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_actual_key_here
   ```

## Usage

Start the interactive CLI by running:
```bash
python main.py
```

### Example Queries
*   *"What does 'eating head' mean in Bengali?"*
*   *"Why do Spanish speakers sometimes say 'usted' and sometimes 'tú'?"*
*   *"Please explain the Japanese concept of wearing a cat."*

## Testing Offline Components

If you want to verify that the local embedding model and Chroma database are working correctly without using your Gemini API key, run:
```bash
python test_rag.py
```
