import os
import sys
from dotenv import load_dotenv
from rag import MultilingualRAG

def main():
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
    
    while True:
        try:
            query = input("\n> Ask about a cultural phrase or rule: ").strip()
            if query.lower() in ['exit', 'quit', 'q']:
                break
            if not query:
                continue
                
            explanation = rag_system.generate_explanation(query)
            
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
