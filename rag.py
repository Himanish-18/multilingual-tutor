import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class MultilingualRAG:
    def __init__(self, data_path="data.json", db_path="./chroma_db"):
        self.data_path = data_path
        self.db_path = db_path
        
        # 1. Initialize the embedding model (runs locally)
        print("[System] Loading local embedding model (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. Initialize Vector Store
        print("[System] Connecting to Vector Database...")
        from chromadb.config import Settings
        self.vector_store = Chroma(
            collection_name="multilingual_cultural_knowledge",
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
            client_settings=Settings(anonymized_telemetry=False)
        )
        
        # 3. Setup the Generative AI (Requires valid GEMINI_API_KEY)
        self.llm = None
        self.chain = None
        
        # We will attempt to initialize it if an API key is available
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
            
            # 4. RAG Prompt Template
            template = """
            You are a knowledgeable linguistics and cultural professor. Your goal is to explain the meaning, cultural nuance, 
            and grammar rules of phrases across different languages.
            
            Use the provided "Textbook Knowledge" to inform your explanation. Do NOT hallucinate idioms or rules that contradict the textbook.
            If the relevant context isn't in the textbook, state that explicitly, but you can still try to give a general answer based on your own knowledge.
            
            User Query: {question}
            
            ==== TEXTBOOK KNOWLEDGE ====
            {context}
            ============================
            
            Provide an empathetic, beautifully formatted, and deeply explanatory answer explaining the "why" and highlighting the cultural nuances.
            """
            self.prompt = ChatPromptTemplate.from_template(template)
            self.chain = self.prompt | self.llm | StrOutputParser()
        else:
            print("[System] Running in offline mode (No API Key). Generation is disabled.")
        
    def ingest_data(self, force=False):
        """Loads data.json and populates the vector store if empty or forced."""
        # Simple check: if we already have documents and aren't forcing, skip
        existing_docs = self.vector_store._collection.count()
        if existing_docs > 0 and not force:
            print(f"[System] Vector DB already populated with {existing_docs} items. Skipping ingestion.")
            return

        print(f"[System] Ingesting knowledge from {self.data_path}...")
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as e:
            print(f"Error reading {self.data_path}: {e}")
            return
            
        documents = []
        for item in raw_data:
            # We construct a rich text representation to be embedded
            content = (
                f"Language: {item.get('language')}\n"
                f"Phrase: {item.get('phrase')}\n"
                f"Literal Translation: {item.get('literal_translation')}\n"
                f"Meaning: {item.get('meaning')}\n"
                f"Context: {item.get('context')}\n"
                f"Cultural Nuance: {item.get('cultural_nuance')}"
            )
            metadata = {"language": item.get("language"), "category": item.get("category")}
            documents.append(Document(page_content=content, metadata=metadata))
            
        # Add to vector store
        if documents:
            self.vector_store.add_documents(documents)
            print(f"[System] Successfully ingested {len(documents)} knowledge items.")

    def retrieve(self, query: str, top_k: int = 3) -> str:
        """Retrieves relevant textbook context for a given query."""
        results = self.vector_store.similarity_search(query, k=top_k)
        
        context_blocks = []
        for i, doc in enumerate(results):
            context_blocks.append(f"--- Rule/Idiom {i+1} ---\n{doc.page_content}")
            
        return "\n\n".join(context_blocks)

    def generate_explanation(self, query: str) -> str:
        """Runs the full RAG pipeline to answer the user's query."""
        # Step 1: Retrieve context
        print("[System] Searching textbook knowledge...")
        context = self.retrieve(query)
        
        # Step 2: Generate answer using the LLM chain
        print("[System] Generating explanation...\n")
        response = self.chain.invoke({"question": query, "context": context})
        
        return response
