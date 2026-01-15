"""
Local Knowledge Brain - Retrieval Augmented Generation (RAG) System

This module provides a 100% local RAG system for SOLAT that:
1. Reads and indexes PDF documents locally
2. Uses sentence-transformers for embeddings (no API calls)
3. Uses FAISS for vector similarity search (CPU-based)
4. Retrieves relevant document chunks
5. Answers strategy and trading questions using local documents

No external APIs, no cloud storage, completely offline capable.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)


class PDFDocumentLoader:
    """Load and parse PDF documents into searchable chunks."""

    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        Initialize PDF loader.

        Args:
            chunk_size (int): Characters per chunk
            overlap (int): Character overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.documents: List[Dict[str, str]] = []

    def load_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Load and chunk a PDF file.

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            List[Dict]: Chunks with metadata
        """
        try:
            pdf_reader = PdfReader(pdf_path)
            text = ""

            # Extract text from all pages
            for page_num, page in enumerate(pdf_reader.pages):
                text += f"\n[Page {page_num + 1}]\n"
                text += page.extract_text()

            # Split into overlapping chunks
            chunks = self._create_chunks(text)

            # Add metadata
            filename = Path(pdf_path).name
            documents = [
                {
                    "text": chunk,
                    "source": filename,
                    "chunk_id": idx,
                    "char_count": len(chunk)
                }
                for idx, chunk in enumerate(chunks)
            ]

            logger.info(f"Loaded {filename}: {len(documents)} chunks, {len(text)} chars")
            return documents

        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            return []

    def _create_chunks(self, text: str) -> List[str]:
        """Create overlapping chunks from text."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If we're not at the end, find a sentence boundary
            if end < len(text):
                # Look for last period/newline within chunk
                last_period = text.rfind(".", start, end)
                if last_period > start:
                    end = last_period + 1

            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.overlap

        return chunks

    def load_directory(self, directory: str) -> List[Dict[str, str]]:
        """
        Load all PDFs from a directory.

        Args:
            directory (str): Directory containing PDFs

        Returns:
            List[Dict]: All chunks from all PDFs
        """
        all_documents = []
        pdf_dir = Path(directory)

        if not pdf_dir.exists():
            logger.warning(f"Directory not found: {directory}")
            return []

        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDFs in {directory}")

        for pdf_file in pdf_files:
            docs = self.load_pdf(str(pdf_file))
            all_documents.extend(docs)

        return all_documents


class LocalEmbeddingModel:
    """Generate embeddings using local sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.

        Args:
            model_name (str): HuggingFace model name (small, fast, local)
        """
        self.model_name = model_name
        try:
            # Small, fast model for local use (41MB)
            self.model = SentenceTransformer(model_name)
            logger.info(f"✓ Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None

    def embed_documents(self, documents: List[Dict[str, str]]) -> np.ndarray:
        """
        Embed a list of document chunks.

        Args:
            documents (List[Dict]): List of document chunks with 'text' key

        Returns:
            np.ndarray: Embeddings matrix (n_docs, 384)
        """
        if not self.model:
            return None

        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        logger.info(f"Generated {len(embeddings)} embeddings (shape: {embeddings.shape})")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Args:
            query (str): Query text

        Returns:
            np.ndarray: Query embedding (1, 384)
        """
        if not self.model:
            return None

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        return embedding[0]  # Return 1D array


class FAISSVectorStore:
    """Local vector similarity search using FAISS."""

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize FAISS vector store.

        Args:
            embedding_dim (int): Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        self.documents: List[Dict[str, str]] = []

    def add_documents(self, documents: List[Dict[str, str]], embeddings: np.ndarray) -> None:
        """
        Add documents and their embeddings to the index.

        Args:
            documents (List[Dict]): Document chunks
            embeddings (np.ndarray): Embedding vectors
        """
        if embeddings is None or len(embeddings) == 0:
            logger.warning("No embeddings to add")
            return

        # Convert to float32 for FAISS
        embeddings_f32 = embeddings.astype(np.float32)

        # Add to index
        self.index.add(embeddings_f32)
        self.documents.extend(documents)

        logger.info(f"✓ Added {len(documents)} documents to FAISS index")
        logger.info(f"  Total documents: {len(self.documents)}")
        logger.info(f"  Index size: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Search for similar documents.

        Args:
            query_embedding (np.ndarray): Query embedding (1, 384)
            k (int): Number of results to return

        Returns:
            List[Dict]: Top-k similar documents with scores
        """
        if self.index.ntotal == 0:
            logger.warning("Vector index is empty")
            return []

        # Convert query to float32
        query_f32 = query_embedding.astype(np.float32).reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_f32, min(k, self.index.ntotal))

        # Build results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            doc = self.documents[idx].copy()
            doc["score"] = float(distance)  # L2 distance (lower is better)
            doc["relevance"] = 1.0 / (1.0 + doc["score"])  # Convert to relevance
            results.append(doc)

        return results

    def save(self, path: str) -> None:
        """
        Save index and documents to disk.

        Args:
            path (str): Directory to save to
        """
        try:
            os.makedirs(path, exist_ok=True)

            # Save FAISS index
            index_path = os.path.join(path, "faiss.index")
            faiss.write_index(self.index, index_path)

            # Save documents
            docs_path = os.path.join(path, "documents.pkl")
            with open(docs_path, "wb") as f:
                pickle.dump(self.documents, f)

            logger.info(f"✓ Saved vector store to {path}")

        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

    def load(self, path: str) -> bool:
        """
        Load index and documents from disk.

        Args:
            path (str): Directory to load from

        Returns:
            bool: Success status
        """
        try:
            # Load FAISS index
            index_path = os.path.join(path, "faiss.index")
            self.index = faiss.read_index(index_path)

            # Load documents
            docs_path = os.path.join(path, "documents.pkl")
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)

            logger.info(f"✓ Loaded vector store from {path}")
            logger.info(f"  Documents: {len(self.documents)}")
            logger.info(f"  Index size: {self.index.ntotal}")

            return True

        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False


class LocalKnowledgeBrain:
    """Complete RAG system combining PDF loading, embeddings, and retrieval."""

    def __init__(
        self,
        pdf_directory: str = "data/knowledge",
        cache_dir: str = "data/cache/brain",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the Local Knowledge Brain.

        Args:
            pdf_directory (str): Directory containing PDF documents
            cache_dir (str): Directory to cache embeddings and index
            embedding_model (str): Sentence-transformer model name
        """
        self.pdf_directory = pdf_directory
        self.cache_dir = cache_dir

        # Initialize components
        self.loader = PDFDocumentLoader(chunk_size=500, overlap=100)
        self.embedder = LocalEmbeddingModel(model_name=embedding_model)
        self.vector_store = FAISSVectorStore(embedding_dim=384)

        # Load or create index
        self._initialize_index()

    def _initialize_index(self) -> None:
        """Load cached index or create new one from PDFs."""
        # Try to load from cache
        if os.path.exists(self.cache_dir):
            if self.vector_store.load(self.cache_dir):
                logger.info("✓ Loaded cached knowledge index")
                return

        # Create new index from PDFs
        logger.info("Creating new knowledge index from PDFs...")
        documents = self.loader.load_directory(self.pdf_directory)

        if not documents:
            logger.warning(f"No PDFs found in {self.pdf_directory}")
            return

        # Generate embeddings
        embeddings = self.embedder.embed_documents(documents)

        if embeddings is not None:
            self.vector_store.add_documents(documents, embeddings)
            self.vector_store.save(self.cache_dir)
            logger.info("✓ Created and cached knowledge index")

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents for a query.

        Args:
            query (str): User query
            k (int): Number of results

        Returns:
            List[Dict]: Top-k relevant documents
        """
        if self.vector_store.index.ntotal == 0:
            logger.warning("Knowledge base is empty")
            return []

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        if query_embedding is None:
            return []

        # Search
        results = self.vector_store.search(query_embedding, k=k)

        return results

    def answer_question(self, query: str, k: int = 3) -> Tuple[List[Dict], str]:
        """
        Answer a question using retrieved documents.

        Args:
            query (str): Question to answer
            k (int): Number of documents to retrieve

        Returns:
            Tuple[List, str]: Retrieved docs and formatted context
        """
        docs = self.retrieve(query, k=k)

        if not docs:
            return [], "No relevant documents found in knowledge base."

        # Format context
        context = "# Relevant Knowledge\n\n"
        for i, doc in enumerate(docs, 1):
            context += f"**[{doc['source']} - Chunk {doc['chunk_id']}]**\n"
            context += f"(Relevance: {doc['relevance']:.1%})\n\n"
            context += f"{doc['text'][:300]}...\n\n"
            context += "---\n\n"

        return docs, context

    def get_stats(self) -> Dict:
        """Get knowledge base statistics."""
        return {
            "total_documents": len(self.vector_store.documents),
            "index_size": self.vector_store.index.ntotal,
            "embedding_model": self.embedder.model_name,
            "pdf_directory": self.pdf_directory,
            "cache_directory": self.cache_dir
        }
