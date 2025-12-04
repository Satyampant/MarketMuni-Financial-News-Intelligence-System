import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from app.core.models import NewsArticle
import numpy as np
from app.core.config_loader import get_config

class VectorStore:
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
        
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
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
        """Index article with enriched metadata (sentiment, cross-impacts)."""
        text = f"{article.title}. {article.content}"
        
        if embedding is None:
            embedding = self.create_embedding(text)
        
        # Prepare base metadata (complex objects must be JSON stringified for ChromaDB)
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
        
        # Handle Sentiment Data
        if article.has_sentiment():
            sentiment_data = article.sentiment
            metadata["sentiment"] = json.dumps(sentiment_data)
            
            # Flatten critical fields for filtering
            metadata["sentiment_classification"] = sentiment_data.get("classification", "Neutral")
            metadata["sentiment_confidence"] = sentiment_data.get("confidence_score", 0.0)
            metadata["sentiment_signal_strength"] = sentiment_data.get("signal_strength", 0.0)
            metadata["sentiment_method"] = sentiment_data.get("analysis_method", "unknown")
        else:
            metadata["sentiment"] = json.dumps(None)
            metadata["sentiment_classification"] = "Unknown"
            metadata["sentiment_confidence"] = 0.0
            metadata["sentiment_signal_strength"] = 0.0
            metadata["sentiment_method"] = "not_analyzed"
        
        # Handle Cross-Impact Data
        if hasattr(article, 'cross_impacts') and article.cross_impacts:
            metadata["cross_impacts"] = json.dumps(article.cross_impacts)
            metadata["has_cross_impacts"] = True
            metadata["cross_impact_count"] = len(article.cross_impacts)
            
            target_sectors = list(set(
                impact.get("target_sector", "") 
                for impact in article.cross_impacts
            ))
            metadata["cross_impact_sectors"] = json.dumps(target_sectors)
            
            metadata["max_cross_impact_score"] = max(
                (impact.get("impact_score", 0.0) for impact in article.cross_impacts),
                default=0.0
            )
        else:
            metadata["cross_impacts"] = json.dumps([])
            metadata["has_cross_impacts"] = False
            metadata["cross_impact_count"] = 0
            metadata["cross_impact_sectors"] = json.dumps([])
            metadata["max_cross_impact_score"] = 0.0
        
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
        sentiment_filter: Optional[str] = None,
        include_cross_impacts: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for articles with optional metadata filtering."""
        if query_embedding is None:
            query_embedding = self.create_embedding(query)
        
        # Apply sentiment filtering if requested
        if sentiment_filter and where is None:
            where = {"sentiment_classification": sentiment_filter}
        elif sentiment_filter and where:
            where["sentiment_classification"] = sentiment_filter
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        for i in range(len(results["ids"][0])):
            result = {
                "article_id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "similarity": 1 - results["distances"][0][i]
            }
            
            # Rehydrate JSON strings back to Python objects
            if "entities" in result["metadata"]:
                result["metadata"]["entities"] = json.loads(result["metadata"]["entities"])
            if "impacted_stocks" in result["metadata"]:
                result["metadata"]["impacted_stocks"] = json.loads(result["metadata"]["impacted_stocks"])
            if "sentiment" in result["metadata"]:
                sentiment_json = result["metadata"]["sentiment"]
                result["metadata"]["sentiment"] = json.loads(sentiment_json) if sentiment_json != "null" else None
            
            if include_cross_impacts and "cross_impacts" in result["metadata"]:
                cross_impacts_json = result["metadata"]["cross_impacts"]
                result["metadata"]["cross_impacts"] = json.loads(cross_impacts_json) if cross_impacts_json else []
                
                if "cross_impact_sectors" in result["metadata"]:
                    result["metadata"]["cross_impact_sectors"] = json.loads(result["metadata"]["cross_impact_sectors"])
            
            formatted_results.append(result)
        
        return formatted_results
    
    def get_by_id(
        self, 
        article_id: str,
        include_cross_impacts: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Retrieve article by ID and rehydrate JSON metadata."""
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
        
        if include_cross_impacts and "cross_impacts" in result["metadata"]:
            cross_impacts_json = result["metadata"]["cross_impacts"]
            result["metadata"]["cross_impacts"] = json.loads(cross_impacts_json) if cross_impacts_json else []
            
            if "cross_impact_sectors" in result["metadata"]:
                result["metadata"]["cross_impact_sectors"] = json.loads(result["metadata"]["cross_impact_sectors"])
        
        return result
    
    def delete_article(self, article_id: str) -> None:
        self.collection.delete(ids=[article_id])
    
    def count(self) -> int:
        return self.collection.count()
    
    def get_sentiment_statistics(self) -> Dict[str, Any]:
        """Aggregate sentiment distribution statistics from all documents."""
        all_results = self.collection.get(include=["metadatas"])
        
        if not all_results["ids"]:
            return {
                "total_articles": 0, "analyzed_count": 0,
                "bullish_count": 0, "bearish_count": 0,
                "neutral_count": 0, "unknown_count": 0
            }
        
        sentiment_counts = {
            "Bullish": 0, "Bearish": 0, "Neutral": 0, "Unknown": 0
        }
        
        for metadata in all_results["metadatas"]:
            classification = metadata.get("sentiment_classification", "Unknown")
            sentiment_counts[classification] = sentiment_counts.get(classification, 0) + 1
        
        total = len(all_results["ids"])
        
        return {
            "total_articles": total,
            "analyzed_count": total - sentiment_counts["Unknown"],
            "bullish_count": sentiment_counts["Bullish"],
            "bearish_count": sentiment_counts["Bearish"],
            "neutral_count": sentiment_counts["Neutral"],
            "unknown_count": sentiment_counts["Unknown"],
            "bullish_percentage": round(sentiment_counts["Bullish"] / total * 100, 2) if total > 0 else 0,
            "bearish_percentage": round(sentiment_counts["Bearish"] / total * 100, 2) if total > 0 else 0,
            "neutral_percentage": round(sentiment_counts["Neutral"] / total * 100, 2) if total > 0 else 0
        }
    
    def get_cross_impact_statistics(self) -> Dict[str, Any]:
        """Calculate statistics regarding cross-impact coverage and distribution."""
        all_results = self.collection.get(include=["metadatas"])
        
        if not all_results["ids"]:
            return {
                "total_articles": 0, "articles_with_cross_impacts": 0,
                "total_cross_impacts": 0, "avg_impacts_per_article": 0.0,
                "max_impact_score_overall": 0.0
            }
        
        total_articles = len(all_results["ids"])
        articles_with_impacts = 0
        total_impacts = 0
        max_score_overall = 0.0
        
        for metadata in all_results["metadatas"]:
            if metadata.get("has_cross_impacts", False):
                articles_with_impacts += 1
                total_impacts += metadata.get("cross_impact_count", 0)
                
                max_score = metadata.get("max_cross_impact_score", 0.0)
                if max_score > max_score_overall:
                    max_score_overall = max_score
        
        avg_impacts = total_impacts / articles_with_impacts if articles_with_impacts > 0 else 0.0
        
        return {
            "total_articles": total_articles,
            "articles_with_cross_impacts": articles_with_impacts,
            "coverage_percentage": round(articles_with_impacts / total_articles * 100, 2) if total_articles > 0 else 0,
            "total_cross_impacts": total_impacts,
            "avg_impacts_per_article": round(avg_impacts, 2),
            "max_impact_score_overall": round(max_score_overall, 2)
        }
    
    def reset(self) -> None:
        """Wipe the current collection and recreate it."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )