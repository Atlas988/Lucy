import chromadb
from chromadb.utils import embedding_functions
import hashlib

class SemanticMemory:
    def __init__(self, persist_directory="memory/chroma_store"):
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
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )

    def save(self, text, metadata=None):
        # Alias for add() so old code stays compatible
        self.add(text, metadata)

    def query(self, text, n_results=3):
        results = self.collection.query(
            query_texts=[text],
            n_results=n_results
        )
        return results["documents"][0] if results["documents"] else []

    def list_all(self):
        return self.collection.get()["documents"]
