import json
import shutil
import sys
from pathlib import Path
from rag import MultilingualRAG


def _configure_utf8_console():
    """Ensure multilingual output works on Windows terminals."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _seed_cache_from_data_json(cache_path: str = "./data_cache.json"):
    """
    Copy data.json → data_cache.json so that ingest_data() loads from the
    cache instead of hitting the network.  This keeps tests fast and offline.
    """
    src = Path("data.json")
    if src.exists():
        shutil.copy(src, cache_path)
        print(f"[Setup] Seeded cache from data.json → {cache_path}")
    else:
        print("[Setup] WARNING: data.json not found; tests may hit the network.")


def test_retrieval():
    _configure_utf8_console()

    # Seed cache to avoid network calls during tests
    _seed_cache_from_data_json()

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

    # 3. Test language-filtered retrieval
    print("\n[Test 3] Testing language-filtered retrieval...")
    try:
        context = rag.retrieve_with_language("greeting customs", "Japanese", top_k=1)
        if context.strip():
            print("Language-filtered result:")
            print("\n".join(context.split("\n")[:3]) + "...")
            print("[PASS] Language-filtered retrieval working.")
        else:
            print("[FAIL] No results from language-filtered retrieval.")
    except Exception as e:
        print(f"[FAIL] Language-filtered retrieval error: {e}")


if __name__ == "__main__":
    test_retrieval()

