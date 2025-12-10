"""
Deduplication Agent with MongoDB Hydration 
"""

from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import CrossEncoder

from app.core.models import NewsArticle
from app.core.config_loader import get_config

if TYPE_CHECKING:
    from app.services.vector_store import VectorStore
    from app.services.mongodb_store import MongoDBStore

class DeduplicationAgent:
    """
    Two-stage semantic deduplication pipeline:
    1. Vector Search (ChromaDB): Fast candidate retrieval (IDs only).
    2. MongoDB Hydration: Fetch full article text.
    3. Cross-Encoder: High-precision verification on hydrated candidates.
    """
    
    def __init__(self, cross_encoder_threshold: float = None, candidate_pool_size: int = 10):
        config = get_config()
        
        self.cross_threshold = cross_encoder_threshold or config.deduplication.cross_encoder_threshold
        self.min_similarity = config.deduplication.bi_encoder_threshold
        self.candidate_pool_size = candidate_pool_size
        
        self.cross_encoder = CrossEncoder(config.deduplication.cross_encoder_model)
        
        print(f"✓ DeduplicationAgent initialized (MongoDB Hydration)")
    
    def _prepare_text(self, article: NewsArticle) -> str:
        """Prepare article text for comparison."""
        return f"{article.title}. {article.content}"
    
    def find_duplicates_with_vector_search(
        self,
        article: NewsArticle,
        article_embedding: List[float],
        vector_store: 'VectorStore',
        mongodb_store: 'MongoDBStore'
    ) -> List[str]:
        """
        Retrieves candidates via vector search, hydrates from MongoDB, 
        then verifies duplicates using cross-encoder.
        """
        # Step 1: Retrieve candidate IDs from ChromaDB (vector search only)
        candidates = vector_store.search(
            query="",
            top_k=self.candidate_pool_size,
            query_embedding=article_embedding
        )
        
        if not candidates:
            return []
        
        # Step 2: Extract candidate IDs and filter by similarity threshold
        candidate_ids = [
            c["article_id"] for c in candidates 
            if c["article_id"] != article.id and c.get("similarity", 0.0) >= self.min_similarity
        ]
        
        if not candidate_ids:
            return []
        
        # Step 3: Hydrate candidates from MongoDB (fetch full text)
        candidate_articles = mongodb_store.get_articles_by_ids(candidate_ids)
        
        if not candidate_articles:
            return []
        
        # Step 4: Run Cross-Encoder verification
        article_text = self._prepare_text(article)
        pairs = []
        candidate_id_map = {}
        
        for idx, candidate_article in enumerate(candidate_articles):
            candidate_text = self._prepare_text(candidate_article)
            pairs.append([article_text, candidate_text])
            candidate_id_map[idx] = candidate_article.id
        
        if not pairs:
            return []
        
        # Run cross-encoder on hydrated candidate pairs
        cross_scores = self.cross_encoder.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Filter by cross-encoder threshold
        duplicate_ids = [
            candidate_id_map[idx]
            for idx, score in enumerate(cross_scores)
            if score >= self.cross_threshold
        ]
        
        return duplicate_ids
    
    def consolidate_duplicates(self, articles: List[NewsArticle]) -> NewsArticle:
        """
        Merges duplicates, keeping the earliest timestamp and aggregating unique sources.
        """
        if not articles:
            raise ValueError("Cannot consolidate empty article list")
        
        if len(articles) == 1:
            return articles[0]
        
        # Sort by timestamp (earliest first)
        sorted_articles = sorted(articles, key=lambda x: x.timestamp)
        primary = sorted_articles[0]
        
        # Aggregate unique sources
        seen_sources = set()
        unique_sources = []
        
        for article in sorted_articles:
            sources = [s.strip() for s in article.source.split(',')]
            for source in sources:
                if source not in seen_sources:
                    seen_sources.add(source)
                    unique_sources.append(source)
        
        primary.source = ", ".join(unique_sources)
        
        return primary
    
    def verify_similarity(self, article1_text: str, article2_text: str) -> float:
        """Calculates cross-encoder similarity score (0-1) between two texts."""
        score = self.cross_encoder.predict(
            [[article1_text, article2_text]],
            show_progress_bar=False,
            convert_to_numpy=True
        )[0]
        
        return float(score)


def find_duplicates_batch(
    articles_with_embeddings: List[Tuple[NewsArticle, List[float]]],
    vector_store: 'VectorStore',
    mongodb_store: 'MongoDBStore',
    dedup_agent: DeduplicationAgent
) -> dict:
    """
    Batch processes multiple articles for duplicate detection.
    """
    results = {}
    for article, embedding in articles_with_embeddings:
        duplicates = dedup_agent.find_duplicates_with_vector_search(
            article, 
            embedding,
            vector_store,
            mongodb_store
        )
        results[article.id] = duplicates
    return results