from rag import MultilingualRAG

def test_retrieval():
    print("Initializing RAG for testing...")
    rag = MultilingualRAG()
    
    # 1. Test Ingestion
    print("\n[Test 1] Forcing data ingestion...")
    rag.ingest_data(force=True)
    
    # 2. Test Retrieval
    print("\n[Test 2] Testing Semantic Retrieval...")
    test_queries = [
        "Why do Spanish people say tomar el pelo?",
        "What does 'eating head' mean in Bengali?",
        "Explain the Japanese phrase for wearing a cat",
        "Why do Germans like seeing others fail?"
    ]
    
    success = True
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            context = rag.retrieve(query, top_k=1)
            if context.strip():
                print("Result Context Sample:")
                # Print just the first few lines of context to verify
                print("\n".join(context.split("\n")[:3]) + "...")
            else:
                print("Result: FAILED (No context retrieved)")
                success = False
        except Exception as e:
            print(f"Exception during retrieval: {e}")
            success = False
            
    if success:
        print("\n✅ Local Vector DB and Embeddings are working perfectly!")
    else:
        print("\n❌ Retrieval test failed.")

if __name__ == "__main__":
    test_retrieval()
