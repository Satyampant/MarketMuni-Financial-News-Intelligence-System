"""
File: app/agents/deduplication.py
Fixed Deduplication Agent - VectorStore as single source of truth for embeddings.
"""

from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import CrossEncoder

from app.core.models import NewsArticle
from app.core.config_loader import get_config


class DeduplicationAgent:
    """
    Two-stage semantic deduplication pipeline:
    1. Vector Search (ChromaDB): Fast candidate retrieval.
    2. Cross-Encoder: High-precision verification on top-K candidates.
    """
    
    def __init__(self, cross_encoder_threshold: float = None,candidate_pool_size: int = 10):
        config = get_config()
        
        self.cross_threshold = cross_encoder_threshold or config.deduplication.cross_encoder_threshold
        self.candidate_pool_size = candidate_pool_size
        
        # Only initialize cross-encoder; embeddings are delegated to VectorStore
        self.cross_encoder = CrossEncoder(config.deduplication.cross_encoder_model)
        
        print(f"✓ DeduplicationAgent initialized (Vector Search Optimized)")
    
    def _prepare_text(self, article: NewsArticle) -> str:
        return f"{article.title}. {article.content}"
    
    def find_duplicates_with_vector_search(self,
        article: NewsArticle,
        article_embedding: List[float],
        vector_store: 'VectorStore'
    ) -> List[str]:
        """
        Retrieves candidates via vector search, then verifies duplicates using a cross-encoder.
        """
        # Stage 1: Retrieve top-K candidates using Vector Search (ANN)
        candidates = vector_store.search(
            query="",
            top_k=self.candidate_pool_size,
            query_embedding=article_embedding
        )
        
        if not candidates:
            return []
        
        # Stage 2: Cross-Encoder Verification (Precision Check)
        article_text = self._prepare_text(article)
        
        pairs = []
        candidate_ids = []
        
        config = get_config()
        min_similarity = config.deduplication.bi_encoder_threshold

        for candidate in candidates:
            candidate_id = candidate["metadata"]["article_id"]
            
            # Skip self-match
            if candidate_id == article.id:
                continue
            
            # Filter candidates below vector search similarity threshold
            similarity = candidate.get("similarity", 0.0)
            if similarity < min_similarity:
                continue
                
            candidate_text = candidate["document"]
            pairs.append([article_text, candidate_text])
            candidate_ids.append(candidate_id)
        
        if not pairs:
            return []
        
        # Run cross-encoder on selected candidate pairs
        cross_scores = self.cross_encoder.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Filter by cross-encoder threshold
        duplicate_ids = [
            candidate_ids[idx] 
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
    dedup_agent: DeduplicationAgent
) -> dict:
    """Batch processes multiple articles for duplicate detection."""
    results = {}
    for article, embedding in articles_with_embeddings:
        duplicates = dedup_agent.find_duplicates_with_vector_search(
            article, 
            embedding,
            vector_store
        )
        results[article.id] = duplicates
    return results