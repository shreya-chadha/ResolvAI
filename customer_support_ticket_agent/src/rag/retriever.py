from __future__ import annotations

import os
from pathlib import Path

from langchain_chroma import Chroma

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.embeddings import build_embeddings
from src.utils.errors import ComponentNotReadyError


class KnowledgeRetriever:
    """Persistent Chroma retrieval scaffold with stable output contracts."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.documents_dir = documents_dir
        self._store: Chroma | None = None

    async def initialize(self) -> None:
        try:
            documents = split_support_documents(load_support_documents(self.documents_dir))
            embeddings = build_embeddings(self.settings)

            vector_db_path = Path(self.settings.vector_db_path)
            vector_db_path.mkdir(parents=True, exist_ok=True)

            store = Chroma(
                collection_name=self.settings.rag_collection,
                embedding_function=embeddings,
                persist_directory=str(vector_db_path),
            )

            if documents:
                # Assign deterministic IDs based on content hash/source to avoid duplicate insertions
                doc_ids = [
                    f"{Path(doc.metadata.get('source', 'doc')).name}_{i}"
                    for i, doc in enumerate(documents)
                ]
                store.add_documents(documents=documents, ids=doc_ids)

            self._store = store
        except Exception as e:
            raise RuntimeError(f"Failed to initialize KnowledgeRetriever: {e}") from e

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, str]]:
        if not query or not query.strip():
            return []

        if self._store is None:
            raise ComponentNotReadyError("KnowledgeRetriever is not initialized.")

        top_k = limit if limit is not None else self.settings.rag_top_k

        # Perform similarity search with score
        results_with_scores = self._store.similarity_search_with_relevance_scores(
            query, k=top_k
        )

        formatted_results = []
        for doc, score in results_with_scores:
            # Clean source metadata to prevent leaking absolute local paths
            raw_source = doc.metadata.get("source", "knowledge_base")
            clean_source = Path(str(raw_source)).name

            formatted_results.append({
                "content": doc.page_content,
                "source": clean_source,
                "score": str(score),
            })

        return formatted_results