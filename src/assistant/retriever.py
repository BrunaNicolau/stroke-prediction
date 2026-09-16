"""
src/assistant/retriever.py

Builds a FAISS vector store over the medical corpus
(data/medical_corpus/train.jsonl + eval.jsonl, treated as a collection of
protocol/FAQ/laudo documents) using local sentence-transformers embeddings
— no external API needed for retrieval, so RAG stays reproducible/offline
and every answer can cite which document it came from (explainability,
challenge item 3).
"""

from __future__ import annotations

import os

from data.medical_corpus.preprocessing import read_jsonl

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CORPUS_DIR = os.path.join(_PROJECT_ROOT, "data", "medical_corpus")
DEFAULT_INDEX_DIR = os.path.join(_PROJECT_ROOT, "results", "faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus_documents(corpus_dir: str = CORPUS_DIR) -> list[dict]:
    """
    Load every {"instruction", "input", "output", "source"} example from
    train.jsonl and eval.jsonl in corpus_dir as retrievable documents.
    """
    documents = []
    for filename in ("train.jsonl", "eval.jsonl"):
        path = os.path.join(corpus_dir, filename)
        if os.path.exists(path):
            documents.extend(read_jsonl(path))
    return documents


def build_vectorstore(
    documents: list[dict] | None = None, embedding_model_name: str = EMBEDDING_MODEL_NAME
):
    """
    Build an in-memory FAISS vector store from corpus documents. Each
    document's "output" text (the protocol/FAQ answer) is embedded; its
    "instruction" and "source" are kept as metadata for citation.

    Returns a langchain_community.vectorstores.FAISS instance.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    documents = documents if documents is not None else load_corpus_documents()
    lc_documents = [
        Document(
            page_content=doc.get("output", ""),
            metadata={
                "instruction": doc.get("instruction", ""),
                "source": doc.get("source", "desconhecida"),
            },
        )
        for doc in documents
        if doc.get("output")
    ]
    if not lc_documents:
        raise ValueError(
            "No documents to index — run "
            "'python -m data.medical_corpus.build_corpus' first."
        )
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return FAISS.from_documents(lc_documents, embeddings)


def save_vectorstore(vectorstore, index_dir: str = DEFAULT_INDEX_DIR) -> str:
    """Persist a FAISS vector store to disk (index_dir). Returns index_dir."""
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    return index_dir


def load_vectorstore(
    index_dir: str = DEFAULT_INDEX_DIR, embedding_model_name: str = EMBEDDING_MODEL_NAME
):
    """Load a previously-saved FAISS vector store from index_dir."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def retrieve(vectorstore, query: str, k: int = 3) -> list[dict]:
    """
    Retrieve the top-k most relevant documents for `query`.

    Returns a list of {"text": str, "instruction": str, "source": str}.
    """
    results = vectorstore.similarity_search(query, k=k)
    return [
        {
            "text": doc.page_content,
            "instruction": doc.metadata.get("instruction", ""),
            "source": doc.metadata.get("source", "desconhecida"),
        }
        for doc in results
    ]
