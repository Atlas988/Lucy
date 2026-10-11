import hashlib

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:  # pragma: no cover - optional dependency
    chromadb = None
    embedding_functions = None


class SemanticMemory:
    def __init__(self, persist_directory="memory/chroma_store"):
        self._fallback_store = {}
        if chromadb is None:
            self.client = None
            self.collection = None
            return

        self.client = chromadb.Client(
            chromadb.config.Settings(persist_directory=persist_directory)
        )
        self.collection = self.client.get_or_create_collection(
            name="lucy_memory",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )

    def _id_for(self, text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def add(self, text, metadata=None):
        doc_id = self._id_for(text)
        if self.collection is None:
            self._fallback_store[doc_id] = {"text": text, "metadata": metadata or {}}
            return
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )

    def save(self, text, metadata=None):
        # Alias for add() so old code stays compatible
        self.add(text, metadata)

    def query(self, text, n_results=3):
        if self.collection is None:
            normalized = text.lower()
            matches = []
            for item in self._fallback_store.values():
                item_text = item["text"].lower()
                if normalized in item_text or item_text in normalized:
                    matches.append(item["text"])
            return matches[:n_results]

        results = self.collection.query(
            query_texts=[text],
            n_results=n_results
        )
        return results["documents"][0] if results["documents"] else []

    def list_all(self):
        if self.collection is None:
            return [item["text"] for item in self._fallback_store.values()]
        return self.collection.get()["documents"]
