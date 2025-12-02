import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from news_storage import NewsArticle
import numpy as np

class VectorStore:
    def __init__(
        self,
        collection_name: str = "financial_news",
        persist_directory: str = "./chroma_db",
        embedding_model: Any = "all-mpnet-base-v2"
    ):
        
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        if isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = embedding_model

    def create_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def index_article(
        self,
        article: NewsArticle,
        embedding: Optional[List[float]] = None
    ) -> None:
        # Combine title and content for embedding
        text = f"{article.title}. {article.content}"
        
        # Generate embedding if not provided
        if embedding is None:
            embedding = self.create_embedding(text)
        
        # Prepare base metadata
        metadata = {
            "article_id": article.id,
            "title": article.title,
            "source": article.source,
            "timestamp": article.timestamp.isoformat(),
            "entities": json.dumps(getattr(article, "entities", {})),
            "impacted_stocks": json.dumps([
                {
                    "symbol": stock["symbol"],
                    "confidence": stock["confidence"],
                    "impact_type": stock["impact_type"]
                }
                for stock in article.impacted_stocks
            ])
        }
        
        # Add sentiment data if available
        if article.has_sentiment():
            sentiment_data = article.sentiment
            
            # Store full sentiment object as JSON
            metadata["sentiment"] = json.dumps(sentiment_data)
            
            # Add individual sentiment fields for filtering/querying
            metadata["sentiment_classification"] = sentiment_data.get("classification", "Neutral")
            metadata["sentiment_confidence"] = sentiment_data.get("confidence_score", 0.0)
            metadata["sentiment_signal_strength"] = sentiment_data.get("signal_strength", 0.0)
            metadata["sentiment_method"] = sentiment_data.get("analysis_method", "unknown")
        else:
            # Default values when sentiment is not available
            metadata["sentiment"] = json.dumps(None)
            metadata["sentiment_classification"] = "Unknown"
            metadata["sentiment_confidence"] = 0.0
            metadata["sentiment_signal_strength"] = 0.0
            metadata["sentiment_method"] = "not_analyzed"
        
        # Add to collection
        self.collection.add(
            ids=[article.id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        query_embedding: Optional[List[float]] = None,
        where: Optional[Dict[str, Any]] = None,
        sentiment_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        # Generate query embedding if not provided
        if query_embedding is None:
            query_embedding = self.create_embedding(query)
        
        # Build where clause with sentiment filter
        if sentiment_filter and where is None:
            where = {"sentiment_classification": sentiment_filter}
        elif sentiment_filter and where:
            where["sentiment_classification"] = sentiment_filter
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            result = {
                "article_id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
            }
            
            # Parse JSON fields in metadata
            if "entities" in result["metadata"]:
                result["metadata"]["entities"] = json.loads(result["metadata"]["entities"])
            if "impacted_stocks" in result["metadata"]:
                result["metadata"]["impacted_stocks"] = json.loads(result["metadata"]["impacted_stocks"])
            if "sentiment" in result["metadata"]:
                sentiment_json = result["metadata"]["sentiment"]
                # Parse sentiment JSON (handle None case)
                result["metadata"]["sentiment"] = json.loads(sentiment_json) if sentiment_json != "null" else None
            
            formatted_results.append(result)
        
        return formatted_results
    
    def get_by_id(self, article_id: str) -> Optional[Dict[str, Any]]:
        results = self.collection.get(
            ids=[article_id],
            include=["documents", "metadatas", "embeddings"]
        )
        
        if not results["ids"]:
            return None
        
        result = {
            "article_id": results["ids"][0],
            "document": results["documents"][0],
            "metadata": results["metadatas"][0],
            "embedding": results["embeddings"][0]
        }
        
        # Parse JSON fields
        if "entities" in result["metadata"]:
            result["metadata"]["entities"] = json.loads(result["metadata"]["entities"])
        if "impacted_stocks" in result["metadata"]:
            result["metadata"]["impacted_stocks"] = json.loads(result["metadata"]["impacted_stocks"])
        if "sentiment" in result["metadata"]:
            sentiment_json = result["metadata"]["sentiment"]
            result["metadata"]["sentiment"] = json.loads(sentiment_json) if sentiment_json != "null" else None
        
        return result
    
    def delete_article(self, article_id: str) -> None:
        self.collection.delete(ids=[article_id])
    
    def count(self) -> int:
        """Get total number of indexed articles"""
        return self.collection.count()
    
    def get_sentiment_statistics(self) -> Dict[str, Any]:
        # Get all documents
        all_results = self.collection.get(include=["metadatas"])
        
        if not all_results["ids"]:
            return {
                "total_articles": 0,
                "analyzed_count": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "unknown_count": 0
            }
        
        # Count sentiment classifications
        sentiment_counts = {
            "Bullish": 0,
            "Bearish": 0,
            "Neutral": 0,
            "Unknown": 0
        }
        
        for metadata in all_results["metadatas"]:
            classification = metadata.get("sentiment_classification", "Unknown")
            sentiment_counts[classification] = sentiment_counts.get(classification, 0) + 1
        
        total = len(all_results["ids"])
        analyzed = total - sentiment_counts["Unknown"]
        
        return {
            "total_articles": total,
            "analyzed_count": analyzed,
            "bullish_count": sentiment_counts["Bullish"],
            "bearish_count": sentiment_counts["Bearish"],
            "neutral_count": sentiment_counts["Neutral"],
            "unknown_count": sentiment_counts["Unknown"],
            "bullish_percentage": round(sentiment_counts["Bullish"] / total * 100, 2) if total > 0 else 0,
            "bearish_percentage": round(sentiment_counts["Bearish"] / total * 100, 2) if total > 0 else 0,
            "neutral_percentage": round(sentiment_counts["Neutral"] / total * 100, 2) if total > 0 else 0
        }
    
    def reset(self) -> None:
        """Reset vector store by deleting and recreating collection"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )