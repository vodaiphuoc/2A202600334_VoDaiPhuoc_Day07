from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        r"""
        config ChormaDB to use cosine similarity for HNSW index
        can look at [config](https://docs.trychroma.com/docs/collections/configure)
        """
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            client = chromadb.Client()
            self._collection = client.create_collection(
                name=collection_name, 
                configuration={
                "hnsw": { "space": "cosine" }
            })
            self._use_chroma = True

        except Exception:
            print('fall back')
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        embedding = self._embedding_fn(doc.content)
        metadata = dict(doc.metadata or {})
        if "doc_id" not in metadata:
            metadata["doc_id"] = doc.id

        record = {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": metadata,
        }
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        if not query or not records or top_k <= 0:
            return []

        query_embedding = self._embedding_fn(query)
        scored: list[dict[str, Any]] = []

        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append(
                {
                    "id": record["id"],
                    "content": record["content"],
                    "metadata": record["metadata"],
                    "score": score,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        if not docs:
            return

        records = [self._make_record(doc) for doc in docs]

        if self._use_chroma and self._collection is not None:
            ids = []
            documents = []
            embeddings = []
            metadatas = []

            curr_count = self._collection.count()

            for i, record in enumerate(records):
                record_id = f"{self._collection_name}-{self._next_index + curr_count+ i}"
                ids.append(str(record_id))
                documents.append(record["content"])
                embeddings.append(record["embedding"])
                metadatas.append(record["metadata"])

            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            self._next_index += len(records)
            return

        self._store.extend(records)
        self._next_index += len(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.

        and retrun output `top_k` document in descending score
        """
        # TODO: embed query, compute similarities, return top_k
        if not query or top_k <= 0:
            return []

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )

            ids = result.get("ids", [[]])[0]
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0] if "distances" in result else [None] * len(ids)

            output: list[dict[str, Any]] = []
            for record_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
                output.append(
                    {
                        "id": record_id,
                        "content": content,
                        "metadata": metadata or {},
                        "score": distance,
                    }
                )
            
            output.sort(key=lambda x: x["score"], reverse=True)
            return output[:top_k]

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        if self._use_chroma and self._collection is not None:
            return int(self._collection.count())
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        if not query or top_k <= 0:
            return []

        metadata_filter = metadata_filter or {}

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": top_k,
            }
            if metadata_filter:
                kwargs["where"] = metadata_filter

            result = self._collection.query(**kwargs)

            ids = result.get("ids", [[]])[0]
            documents = result.get("documents", [[]])[0]
            metadatas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0] if "distances" in result else [None] * len(ids)

            output: list[dict[str, Any]] = []
            for record_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
                output.append(
                    {
                        "id": record_id,
                        "content": content,
                        "metadata": metadata or {},
                        "score": distance,
                    }
                )
            return output

        filtered_records = self._store
        if metadata_filter:
            filtered_records = [
                record
                for record in self._store
                if all(record["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]

        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        if not doc_id:
            return False

        if self._use_chroma and self._collection is not None:
            before = self._collection.count()
            self._collection.delete(where={"doc_id": doc_id})
            after = self._collection.count()
            return after < before

        before = len(self._store)
        self._store = [
            record
            for record in self._store
            if record["metadata"].get("doc_id") != doc_id
        ]
        return len(self._store) < before
