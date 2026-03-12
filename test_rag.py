import sys
from rag import MultilingualRAG


def _configure_utf8_console():
    """Ensure multilingual output works on Windows terminals."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def test_retrieval():
    _configure_utf8_console()

    print("Initializing RAG for testing...")
    rag = MultilingualRAG()

    # 1. Test ingestion
    print("\n[Test 1] Forcing data ingestion...")
    rag.ingest_data(force=True)

    # 2. Test retrieval
    print("\n[Test 2] Testing semantic retrieval...")
    test_queries = [
        "Why do Spanish people say tomar el pelo?",
        "What does 'eating head' mean in Bengali?",
        "Explain the Japanese phrase for wearing a cat",
        "Why do Germans like seeing others fail?",
    ]

    success = True
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            context = rag.retrieve(query, top_k=1)
            if context.strip():
                print("Result Context Sample:")
                print("\n".join(context.split("\n")[:3]) + "...")
            else:
                print("Result: FAILED (No context retrieved)")
                success = False
        except Exception as e:
            print(f"Exception during retrieval: {e}")
            success = False

    if success:
        print("\n[PASS] Local vector DB and embeddings are working.")
    else:
        print("\n[FAIL] Retrieval test failed.")


if __name__ == "__main__":
    test_retrieval()
