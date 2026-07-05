from typing import List, Dict, Any
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_PERSIST_DIR, SESSION_MEMORY_LIMIT


# ─── SESSION MEMORY (in-memory, resets each run) ──────────────────────────────

class SessionMemory:
    """
    Lightweight in-memory store for current conversation turn history.
    Keeps the last N messages to avoid context overflow.
    """

    def __init__(self, limit: int = SESSION_MEMORY_LIMIT):
        self.messages: List[Dict[str, str]] = []
        self.limit = limit

    def add(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Trim to limit, always keep system message if present
        if len(self.messages) > self.limit:
            self.messages = self.messages[-self.limit:]

    def get_history(self) -> List[Dict[str, str]]:
        # Return only role + content for LLM context
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def clear(self):
        self.messages = []

    def summary(self) -> str:
        return f"{len(self.messages)} messages in session memory"


# ─── PERSISTENT MEMORY (ChromaDB, survives restarts) ──────────────────────────

class PersistentMemory:
    """
    ChromaDB-backed memory for storing and retrieving past conversations
    across sessions using semantic similarity search.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="assistant_memory",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"}
        )

    def store(self, user_input: str, assistant_response: str):
        """Store a Q&A pair with timestamp metadata."""
        doc_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        combined = f"User: {user_input}\nAssistant: {assistant_response}"
        self.collection.add(
            documents=[combined],
            ids=[doc_id],
            metadatas=[{"timestamp": datetime.now().isoformat(), "type": "conversation"}]
        )

    def retrieve(self, query: str, n_results: int = 3) -> str:
        """Retrieve semantically similar past conversations."""
        try:
            count = self.collection.count()
            if count == 0:
                return "No past memories found."

            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, count)
            )

            if not results["documents"] or not results["documents"][0]:
                return "No relevant memories found."

            output = ["RELEVANT PAST CONTEXT:"]
            for i, doc in enumerate(results["documents"][0], 1):
                meta = results["metadatas"][0][i - 1]
                output.append(f"\n[Memory {i} — {meta.get('timestamp', 'unknown')}]\n{doc}")

            return "\n".join(output)

        except Exception as e:
            return f"Memory retrieval error: {str(e)}"

    def clear(self):
        """Wipe all stored memories."""
        self.client.delete_collection("assistant_memory")
        self.collection = self.client.get_or_create_collection(
            name="assistant_memory",
            embedding_function=self.ef,
        )

    def count(self) -> int:
        return self.collection.count()


# ─── UNIFIED MEMORY MANAGER ───────────────────────────────────────────────────

class MemoryManager:
    """
    Single interface for both memory modes.
    The UI toggle controls which modes are active.
    """

    def __init__(self):
        self.session = SessionMemory()
        self.persistent = PersistentMemory()
        self.use_persistent = False  # toggled by UI

    def add_user(self, content: str):
        self.session.add("user", content)

    def add_assistant(self, content: str, user_input: str = ""):
        self.session.add("assistant", content)
        if self.use_persistent and user_input:
            self.persistent.store(user_input, content)

    def get_context(self, query: str) -> Dict[str, Any]:
        """Return all relevant context for the agent."""
        context = {
            "session_history": self.session.get_history(),
            "persistent_memories": ""
        }
        if self.use_persistent:
            context["persistent_memories"] = self.persistent.retrieve(query)
        return context

    def clear_session(self):
        self.session.clear()

    def clear_all(self):
        self.session.clear()
        self.persistent.clear()

    def status(self) -> str:
        mode = "Session + Persistent" if self.use_persistent else "Session only"
        return (
            f"Memory mode: {mode} | "
            f"Session: {self.session.summary()} | "
            f"Persistent: {self.persistent.count()} memories stored"
        )
