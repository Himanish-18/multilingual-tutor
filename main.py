import logging
import os
import sys
from dotenv import load_dotenv
from rag import MultilingualRAG

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(message)s",
)


def _configure_utf8_console():
    """Ensure multilingual output works on Windows terminals."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def main():
    _configure_utf8_console()
    print("==================================================")
    print("   Multilingual Culture & Language RAG System   ")
    print("==================================================")
    
    # 1. Load environment variables
    load_dotenv()
    os.environ["ANONYMIZED_TELEMETRY"] = "False"  # Disable ChromaDB telemetry robustly
    
    # Check for Gemini API Key (or Google API Key)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n[ERROR] No Google Gemini API key found.")
        print("Please create a .env file and set GEMINI_API_KEY=your_key")
        print("or set the environment variable directly. Exiting.")
        sys.exit(1)
        
    os.environ["GOOGLE_API_KEY"] = api_key # LangChain specifically looks for GOOGLE_API_KEY
        
    # 2. Initialize the RAG system
    print("\n[System] Initializing RAG pipeline...")
    try:
        rag_system = MultilingualRAG()
        rag_system.ingest_data(force=False) # Will ingest only if DB is empty
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize RAG system: {e}")
        sys.exit(1)
        
    print("\n[System] Initialization complete! Ready for queries.\n")
    print("Type 'exit' or 'quit' to stop.")
    print("Tip: prefix your query with a language name for better results,")
    print("     e.g. 'Japanese: explain neko wo kaburu'\n")
    
    while True:
        try:
            query = input("\n> Ask about a cultural phrase or rule: ").strip()
            if query.lower() in ['exit', 'quit', 'q']:
                break
            if not query:
                continue

            # Try to detect a language hint from "Language: query" format
            language = None
            if ":" in query:
                prefix = query.split(":", 1)[0].strip()
                # Check if the prefix matches a known language name
                from scraper import LANGUAGE_CONFIG
                if prefix in LANGUAGE_CONFIG:
                    language = prefix
                    query = query.split(":", 1)[1].strip()

            explanation = rag_system.generate_explanation(query, language=language)
            
            print("\n" + "="*50)
            print("  EXPLANATION")
            print("="*50)
            print(explanation)
            print("==================================================")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred during generation: {e}")

if __name__ == "__main__":
    main()

