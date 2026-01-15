"""
Knowledge Brain Package - Local RAG System for SOLAT

Provides 100% local, offline-capable RAG (Retrieval Augmented Generation)
for searching and answering questions about PDF documents.
"""

from src.knowledge.brain import (
    PDFDocumentLoader,
    LocalEmbeddingModel,
    FAISSVectorStore,
    LocalKnowledgeBrain,
)

__all__ = [
    "PDFDocumentLoader",
    "LocalEmbeddingModel",
    "FAISSVectorStore",
    "LocalKnowledgeBrain",
]
