from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        if not question or not question.strip():
            return "Question must not be empty."

        results = self.store.search(question, top_k=top_k)

        if not results:
            prompt = (
                "You are a helpful assistant.\n"
                "No relevant context was found in the knowledge base.\n\n"
                f"Question: {question.strip()}\n\n"
                "Answer as helpfully as possible, and clearly say when the knowledge base does not contain enough information."
            )
            return self.llm_fn(prompt)

        context_blocks = []
        for i, result in enumerate(results, start=1):
            content = result.get("content", "").strip()
            metadata = result.get("metadata", {})
            source = metadata.get("source") or metadata.get("doc_id") or result.get("id", f"chunk-{i}")
            context_blocks.append(f"[{i}] Source: {source}\n{content[:30]}")

        context = "\n\n".join(context_blocks)

        prompt = (
            "You are a helpful assistant answering questions using the provided knowledge base context.\n"
            "Use the context below to answer the question.\n"
            "If the answer is not fully supported by the context, say so clearly.\n"
            "Prefer concise, accurate answers grounded in the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question.strip()}\n\n"
            "Answer:"
        )

        return self.llm_fn(prompt)
