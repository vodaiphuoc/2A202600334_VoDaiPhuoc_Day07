from __future__ import annotations

import os
import sys
from pathlib import Path
import uuid
from dotenv import load_dotenv

load_dotenv()

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore
from src.chunking import SentenceChunker

MAX_SENTENCES_PER_CHUNK = 16


BENCHMARK_QUERIES = [
    {
        "id": "1",
        "query": "Mã số của Quy chế đào tạo VinUni là gì?",
        "gold_answer": """VU_HT03.VN"""
    },{
        "id": "2",
        "query": "Tổng số tín chỉ tối thiểu cần đăng ký một kỳ?",
        "gold_answer": """12 tín chỉ đối với sinh viên hệ chính quy"""
    },{
        "id": "3",
        "query": "GPA bao nhiêu thì được xếp loại học lực Giỏi?",
        "gold_answer": """Từ 3.20 đến 3.59 """
    },{
        "id": "4",
        "query": "Một tín chỉ tương đương bao nhiêu giờ học?",
        "gold_answer": """50 giờ học định mức"""
    },{
        "id": "5",
        "query": "Thời gian bảo lưu kết quả học tập tối đa?",
        "gold_answer": """ Từ một đến hai học kỳ"""
    }

]




SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
    "data/customer_support_playbook.txt",
    "data/chunking_experiment_report.md",
    "data/vi_retrieval_notes.md",
    "data/team_selection_custom_data.txt"
]

def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""

    # chunk the content right after loading it
    chunker = SentenceChunker(max_sentences_per_chunk=MAX_SENTENCES_PER_CHUNK)

    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        for ith, chunk_str in enumerate(chunker.chunk(content)):
            # use uuid for global indexing for all chunks for all files
            # anh use chunk order in metadata
            documents.append(
                Document(
                    id=str(uuid.uuid4()),
                    content=chunk_str,
                    metadata={
                        "chunk_id": ith,
                        "source": str(path), 
                        "extension": path.suffix.lower()
                    },
                )
            )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:900].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def main():
    files = SAMPLE_FILES
    
    print("=== SETTING UP BENCHAMRK ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Benchmark ===")
    
    inputs = """| # | Query | Gold Answer |
|---|-------|-------------|
"""
    outputs = """| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
"""
    for case in BENCHMARK_QUERIES:
        _id = case['id']
        query = case['query']
        gold_answer = case['gold_answer'].replace('\n','<br>')
        search_results = store.search(query, top_k=1)
        
        result = search_results[0]
        inputs += f"|{_id}|{query}|{gold_answer}|\n"
        summary_content = result['content'][:150].replace('\n','<br>')
        outputs += f"|{_id}|{query}|{summary_content}|{result['score']:.3f}|||\n"
            
    with open("report/Benchmark.md", "w") as fp:
        fp.write(inputs)
    with open("report/Benchmark_results.md", "w") as fp:
        fp.write(outputs)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
