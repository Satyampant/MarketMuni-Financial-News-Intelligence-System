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
        self.collection = self.client.get_or_create_collection(name=collection_name,metadata={"hnsw:space": "cosine"})

        if isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformer(embedding_model)
        else:
            self.embedding_model = embedding_model

    def create_embedding(self, text: str) -> List[float]:
        
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def index_article(self,article: NewsArticle,embedding: Optional[List[float]] = None) -> None:
        
        # Combine title and content for embedding
        text = f"{article.title}. {article.content}"
        
        # Generate embedding if not provided
        if embedding is None:
            embedding = self.create_embedding(text)
        
        # Prepare metadata
        metadata = {
            "article_id": article.id,
            "title": article.title,
            "source": article.source,
            "timestamp": article.timestamp.isoformat(),
            "entities": json.dumps(getattr(article, "entities", {})),
            "impacted_stocks": json.dumps([
                {
                    "symbol": stock.symbol,
                    "confidence": stock.confidence,
                    "impact_type": stock.impact_type
                }
                for stock in article.impacted_stocks
            ])
        }
        
        # Add to collection
        self.collection.add(
            ids=[article.id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )
    
    def search(self,query: str,top_k: int = 10,
        query_embedding: Optional[List[float]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        
        # Generate query embedding if not provided
        if query_embedding is None:
            query_embedding = self.create_embedding(query)
        
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
        
        return result
    
    def delete_article(self, article_id: str) -> None:
        self.collection.delete(ids=[article_id])
    
    def count(self) -> int:
        return self.collection.count()
    
    def reset(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )