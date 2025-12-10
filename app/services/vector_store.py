"""
- Store only article_id in metadata
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path
from app.core.models import NewsArticle
from app.core.config_loader import get_config


class VectorStore:
    """All metadata and document text now stored in MongoDB.Pure vector index for semantic search. """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: Any = None
    ):
        config = get_config()
    
        # Fallback to config if arguments are not provided
        self.collection_name = collection_name or config.vector_store.collection_name
        persist_dir = persist_directory or config.vector_store.persist_directory
        embedding_model = embedding_model or config.vector_store.embedding_model
        
        self.persist_directory = Path(persist_dir)
        self.persist_directory.mkdir(exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        if isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = embedding_model
        
        print(f"✓ VectorStore initialized (embeddings-only mode)")
        print(f"  - Collection: {self.collection_name}")
        print(f"  - Metadata: Minimal (article_id only)")

    def create_embedding(self, text: str) -> List[float]:
        """Generate embedding using the configured model."""
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def index_article(
        self,
        article: NewsArticle,
        embedding: Optional[List[float]] = None
    ) -> None:
        """
        Index article with MINIMAL metadata (article_id only).
        Store empty string as document (ChromaDB requirement).
        """
        if embedding is None:
            text = f"{article.title}. {article.content}"
            embedding = self.create_embedding(text)
        
        minimal_metadata = {
            "article_id": article.id
        }
        
        # Store empty string as document (ChromaDB requires it)
        self.collection.add(
            ids=[article.id],
            embeddings=[embedding],
            documents=[""],  
            metadatas=[minimal_metadata]
        )
    
    def search_by_ids(
        self,
        query_embedding: List[float],
        article_ids: List[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search within a filtered set of article IDs.
        Used for MongoDB filter → Vector search strategy.
        
        Args:
            query_embedding: Query vector
            article_ids: List of article IDs to search within
            top_k: Number of results to return
            
        Returns:
            List of dicts with article_id, similarity, distance
        """
        if not article_ids:
            return []
        
        # ChromaDB where clause to filter by article_ids
        where_filter = {
            "article_id": {"$in": article_ids}
        }
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(article_ids)),
            where=where_filter,
            include=["metadatas", "distances"]
        )
        
        # Format results with minimal data
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "article_id": results["ids"][0][i],
                "similarity": 1 - results["distances"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return formatted_results
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        query_embedding: Optional[List[float]] = None,
        article_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search with optional article_id filtering.
        Returns:
            List of dicts with article_id, similarity, distance
        """
        if query_embedding is None and query:
            query_embedding = self.create_embedding(query)
        
        if query_embedding is None:
            raise ValueError("Either query text or query_embedding must be provided")
        
        # Use search_by_ids if article_ids filter provided
        if article_ids:
            return self.search_by_ids(query_embedding, article_ids, top_k)
        
        # Unrestricted vector search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        # Format results with minimal data
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "article_id": results["ids"][0][i],
                "similarity": 1 - results["distances"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return formatted_results
    
    def get_all_ids(self) -> List[str]:
        """
        Get all article IDs in the vector store.
        Used for inverted search strategy (Vector search → MongoDB validation).
        """
        results = self.collection.get(
            include=[],  # Don't include any data, just IDs
            limit=None
        )
        
        return results["ids"]
    
    def get_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve article metadata by ID.
        Note: Only returns article_id and embedding (no document text).
        """
        results = self.collection.get(
            ids=[article_id],
            include=["metadatas", "embeddings"]
        )
        
        if not results["ids"]:
            return None
        
        return {
            "article_id": results["ids"][0],
            "metadata": results["metadatas"][0],
            "embedding": results["embeddings"][0]
        }
    
    def delete_article(self, article_id: str) -> None:
        """Delete article from vector store."""
        self.collection.delete(ids=[article_id])
    
    def count(self) -> int:
        """Get total number of articles in vector store."""
        return self.collection.count()
    
    def reset(self) -> None:
        """Wipe the current collection and recreate it."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✓ VectorStore collection '{self.collection_name}' reset")