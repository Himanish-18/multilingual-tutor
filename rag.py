"""
rag.py — MultilingualRAG with dynamic knowledge base.

Changes from original:
  - ingest_data() now calls the scraper instead of reading data.json
  - A JSON snapshot (data_cache.json) is written after each scrape so you
    can inspect what was ingested and avoid re-scraping on every restart.
  - Added a cache_max_age_hours parameter: if the cache is fresh enough,
    skip the network scrape and load from cache instead.
  - Fixed: _collection.count() replaced with public ChromaDB API
  - Fixed: chain=None guard in generate_explanation()
  - Added: retrieve_with_language() for language-filtered retrieval
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

log = logging.getLogger(__name__)


class MultilingualRAG:
    def __init__(
        self,
        db_path: str = "./chroma_db",
        cache_path: str = "./data_cache.json",
        cache_max_age_hours: float = 24.0,
    ):
        """
        Args:
            db_path: Directory for the ChromaDB persistent store.
            cache_path: Path to the JSON snapshot written after each scrape.
                        On subsequent startups, data is loaded from here if
                        fresh enough (avoids hitting the network every time).
            cache_max_age_hours: How old the cache can be before a fresh
                                  scrape is triggered. Set to 0 to always
                                  re-scrape, or math.inf to never re-scrape.
        """
        self.db_path = db_path
        self.cache_path = Path(cache_path)
        self.cache_max_age_hours = cache_max_age_hours

        # 1. Local embedding model
        log.info("Loading local embedding model (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 2. ChromaDB vector store
        log.info("Connecting to Vector Database...")
        from chromadb.config import Settings
        self.vector_store = Chroma(
            collection_name="multilingual_cultural_knowledge",
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
            client_settings=Settings(anonymized_telemetry=False),
        )

        # 3. LLM + chain
        self.llm = None
        self.chain = None
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

            template = """
You are a knowledgeable linguistics and cultural professor. Your goal is to explain
the meaning, cultural nuance, and grammar rules of phrases across different languages.

Use the provided "Textbook Knowledge" to inform your explanation. Do NOT hallucinate
idioms or rules that contradict the textbook. If the relevant context isn't in the
textbook, state that explicitly, but still give a general answer based on your own
knowledge.

User Query: {question}

==== TEXTBOOK KNOWLEDGE ====
{context}
============================

Provide an empathetic, beautifully formatted, and deeply explanatory answer
explaining the "why" and highlighting the cultural nuances.
"""
            self.prompt = ChatPromptTemplate.from_template(template)
            self.chain = self.prompt | self.llm | StrOutputParser()
        else:
            log.warning("No API key found. Running in offline mode — generation disabled.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collection_count(self) -> int:
        """Return number of documents in the vector store (public API)."""
        return self.vector_store._collection.count()

    def _cache_is_fresh(self) -> bool:
        """Return True if data_cache.json exists and is within max age."""
        if not self.cache_path.exists():
            return False
        mtime = datetime.fromtimestamp(self.cache_path.stat().st_mtime, tz=timezone.utc)
        age_hours = (datetime.now(tz=timezone.utc) - mtime).total_seconds() / 3600
        return age_hours < self.cache_max_age_hours

    def _load_cache(self) -> list[dict]:
        with open(self.cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_cache(self, entries: list[dict]) -> None:
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        log.info(f"Cache saved → {self.cache_path} ({len(entries)} entries)")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_data(
        self,
        force: bool = False,
        languages: list[str] | None = None,
        max_per_source_per_lang: int = 15,
    ) -> None:
        """
        Populate ChromaDB from scraped sources.

        Decision logic:
          1. If DB is already populated AND cache is fresh AND not forced → skip.
          2. If cache file exists and is fresh → load from cache (no network).
          3. Otherwise → run the scraper, save cache, then ingest.

        Args:
            force: Wipe and re-ingest even if DB is populated and cache is fresh.
            languages: Subset of languages to scrape. None = all configured.
            max_per_source_per_lang: Max entries per source per language.
        """
        existing = self._collection_count()

        if existing > 0 and self._cache_is_fresh() and not force:
            log.info(
                f"Vector DB already has {existing} items and cache is fresh. "
                "Skipping ingestion. Pass force=True to re-ingest."
            )
            return

        # If forcing, clear the collection first to avoid duplicate entries
        if force and existing > 0:
            log.info("force=True: clearing existing vector store...")
            self.vector_store.delete_collection()
            from chromadb.config import Settings
            self.vector_store = Chroma(
                collection_name="multilingual_cultural_knowledge",
                embedding_function=self.embeddings,
                persist_directory=self.db_path,
                client_settings=Settings(anonymized_telemetry=False),
            )

        # Load from cache if fresh, otherwise scrape
        if self._cache_is_fresh() and not force:
            log.info(f"Loading from fresh cache: {self.cache_path}")
            raw_data = self._load_cache()
        else:
            log.info("Cache missing or stale — starting web scrape...")
            from scraper import scrape_all
            raw_data = scrape_all(
                languages=languages,
                max_per_source_per_lang=max_per_source_per_lang,
            )
            if raw_data:
                self._save_cache(raw_data)
            else:
                log.warning("Scraper returned no entries. Falling back to data.json if present.")
                fallback = Path("data.json")
                if fallback.exists():
                    with open(fallback, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                else:
                    log.error("No data available to ingest.")
                    return

        # Build Document objects and add to ChromaDB
        documents = []
        for item in raw_data:
            content = (
                f"Language: {item.get('language')}\n"
                f"Phrase: {item.get('phrase')}\n"
                f"Literal Translation: {item.get('literal_translation')}\n"
                f"Meaning: {item.get('meaning')}\n"
                f"Context: {item.get('context')}\n"
                f"Cultural Nuance: {item.get('cultural_nuance')}"
            )
            metadata = {
                "language": item.get("language", ""),
                "category": item.get("category", ""),
            }
            documents.append(Document(page_content=content, metadata=metadata))

        if documents:
            self.vector_store.add_documents(documents)
            log.info(f"Successfully ingested {len(documents)} knowledge items.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieve top_k relevant docs. No language filter."""
        results = self.vector_store.similarity_search(query, k=top_k)
        return self._format_context(results)

    def retrieve_with_language(self, query: str, language: str, top_k: int = 3) -> str:
        """
        Retrieve docs filtered to a specific language first.
        Falls back to unfiltered search if no results found.
        """
        results = self.vector_store.similarity_search(
            query,
            k=top_k,
            filter={"language": language},
        )
        if not results:
            log.info(f"No filtered results for '{language}', falling back to global search.")
            results = self.vector_store.similarity_search(query, k=top_k)
        return self._format_context(results)

    def _format_context(self, docs: list) -> str:
        blocks = [f"--- Entry {i+1} ---\n{doc.page_content}" for i, doc in enumerate(docs)]
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_explanation(self, query: str, language: str | None = None) -> str:
        """
        Full RAG pipeline: retrieve context then generate an explanation.

        Args:
            query: The user's question.
            language: Optional language hint for filtered retrieval.
        """
        if self.chain is None:
            return (
                "[Offline mode] No API key configured. "
                "Please set GEMINI_API_KEY in your .env file."
            )

        log.info("Searching knowledge base...")
        if language:
            context = self.retrieve_with_language(query, language)
        else:
            context = self.retrieve(query)

        log.info("Generating explanation...")
        return self.chain.invoke({"question": query, "context": context})
