"""
Fixed Deduplication Agent - No Dual Models
VectorStore is the single source of truth for embeddings.
File: app/agents/deduplication.py
"""

from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import CrossEncoder

from app.core.models import NewsArticle
from app.core.config_loader import get_config


class DeduplicationAgent:
    """
    Two-stage semantic deduplication pipeline optimized for large-scale datasets.
    
    ARCHITECTURE PRINCIPLES:
    1. VectorStore owns all embedding generation (single source of truth)
    2. DeduplicationAgent focuses only on similarity logic and cross-encoder verification
    3. Embeddings are computed once and reused across ingestion pipeline
    
    Pipeline:
    1. Vector Search (ChromaDB): Fast O(log N) candidate retrieval using ANN
    2. Cross-Encoder: High-precision verification on top-K candidates only
    """
    
    def __init__(
        self, 
        cross_encoder_threshold: float = None,
        candidate_pool_size: int = 10
    ):
        """
        Initialize deduplication agent.
        
        IMPORTANT: No embedding model here - VectorStore handles that.
        
        Args:
            cross_encoder_threshold: Minimum probability for duplicate verification
            candidate_pool_size: Number of candidates to retrieve from vector DB
        """
        config = get_config()
        
        self.cross_threshold = cross_encoder_threshold or config.deduplication.cross_encoder_threshold
        self.candidate_pool_size = candidate_pool_size
        
        # Only initialize cross-encoder (not bi-encoder)
        self.cross_encoder = CrossEncoder(config.deduplication.cross_encoder_model)
        
        print(f"✓ DeduplicationAgent initialized (Vector Search Optimized)")
        print(f"  - Candidate pool size: {self.candidate_pool_size}")
        print(f"  - Cross-encoder threshold: {self.cross_threshold}")
        print(f"  - Embedding model: Delegated to VectorStore")
    
    def _prepare_text(self, article: NewsArticle) -> str:
        """Prepare article text for comparison."""
        return f"{article.title}. {article.content}"
    
    def find_duplicates_with_vector_search(
        self,
        article: NewsArticle,
        article_embedding: List[float],
        vector_store: 'VectorStore'
    ) -> List[str]:
        """
        Find duplicate articles using vector search + cross-encoder verification.
        
        OPTIMIZED APPROACH:
        - Stage 1: Use ChromaDB vector search to retrieve top-K similar candidates (fast)
        - Stage 2: Apply cross-encoder only on these candidates (precise)
        
        Args:
            article: New article to check for duplicates
            article_embedding: Pre-computed embedding from VectorStore (REUSED)
            vector_store: ChromaDB vector store instance
            
        Returns:
            List of duplicate article IDs
        """
        # STAGE 1: Vector Search (ANN - Approximate Nearest Neighbors)
        # Query ChromaDB for top-K most similar articles
        # This is O(log N) with HNSW indexing
        candidates = vector_store.search(
            query="",  # Empty query since we provide embedding directly
            top_k=self.candidate_pool_size,
            query_embedding=article_embedding
        )
        
        # Note: bi_encoder_threshold filtering is handled by ChromaDB's similarity scoring
        # We only get back candidates that are reasonably similar
        
        if not candidates:
            return []
        
        # STAGE 2: Cross-Encoder Verification (Precision Check)
        # Only run expensive model on the small pool of candidates
        article_text = self._prepare_text(article)
        
        pairs = []
        candidate_ids = []
        
        for candidate in candidates:
            # Skip self-match
            candidate_id = candidate["metadata"]["article_id"]
            if candidate_id == article.id:
                continue
                
            # Skip candidates below vector search threshold
            # ChromaDB returns similarity (1 - cosine_distance)
            similarity = candidate.get("similarity", 0.0)
            
            # Use config threshold or reasonable default
            config = get_config()
            min_similarity = config.deduplication.bi_encoder_threshold
            
            if similarity < min_similarity:
                continue
                
            candidate_text = candidate["document"]
            pairs.append([article_text, candidate_text])
            candidate_ids.append(candidate_id)
        
        if not pairs:
            return []
        
        # Run cross-encoder on candidate pairs
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
        Merge duplicate articles, keeping earliest as primary and aggregating sources.
        
        Args:
            articles: List of duplicate articles to consolidate
            
        Returns:
            Consolidated primary article
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
            # Handle comma-separated sources
            sources = [s.strip() for s in article.source.split(',')]
            for source in sources:
                if source not in seen_sources:
                    seen_sources.add(source)
                    unique_sources.append(source)
        
        primary.source = ", ".join(unique_sources)
        
        return primary
    
    def verify_similarity(
        self, 
        article1_text: str, 
        article2_text: str
    ) -> float:
        """
        Calculate cross-encoder similarity score between two article texts.
        
        Args:
            article1_text: First article text
            article2_text: Second article text
            
        Returns:
            Cross-encoder similarity score (0-1)
        """
        score = self.cross_encoder.predict(
            [[article1_text, article2_text]],
            show_progress_bar=False,
            convert_to_numpy=True
        )[0]
        
        return float(score)


# Utility function for batch deduplication
def find_duplicates_batch(
    articles_with_embeddings: List[Tuple[NewsArticle, List[float]]],
    vector_store: 'VectorStore',
    dedup_agent: DeduplicationAgent
) -> dict:
    """
    Batch process multiple articles for duplicate detection.
    
    Args:
        articles_with_embeddings: List of (article, embedding) tuples
        vector_store: Vector store instance
        dedup_agent: Deduplication agent instance
        
    Returns:
        Dict mapping article IDs to their duplicate IDs
    """
    results = {}
    for article, embedding in articles_with_embeddings:
        duplicates = dedup_agent.find_duplicates_with_vector_search(
            article, 
            embedding,
            vector_store
        )
        results[article.id] = duplicates
    return results