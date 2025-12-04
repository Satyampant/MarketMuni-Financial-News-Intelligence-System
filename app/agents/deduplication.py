from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from app.core.models import NewsArticle
from app.core.config_loader import get_config

class DeduplicationAgent:
    """
    Implements a two-stage semantic deduplication pipeline:
    1. Bi-Encoder: Fast candidate retrieval via cosine similarity.
    2. Cross-Encoder: High-precision duplicate verification.
    """
    
    def __init__(
        self, 
        bi_encoder_threshold: float = None,
        cross_encoder_threshold: float = None
    ):
        config = get_config()
    
        # Prioritize explicit args over config defaults
        self.bi_threshold = bi_encoder_threshold or config.deduplication.bi_encoder_threshold
        self.cross_threshold = cross_encoder_threshold or config.deduplication.cross_encoder_threshold
        
        self.embedding_model = SentenceTransformer(config.deduplication.bi_encoder_model)
        self.cross_encoder = CrossEncoder(config.deduplication.cross_encoder_model)
        
        self.embeddings_cache = {}
    
    def _prepare_text(self, article: NewsArticle) -> str:
        return f"{article.title}. {article.content}"
    
    def _get_embedding(self, article: NewsArticle) -> np.ndarray:
        if article.id not in self.embeddings_cache:
            text = self._prepare_text(article)
            self.embeddings_cache[article.id] = self.embedding_model.encode(
                text, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
        return self.embeddings_cache[article.id]
    
    def get_cached_embedding(self, article_id: str) -> Optional[np.ndarray]:
        return self.embeddings_cache.get(article_id)
    
    def clear_cache(self) -> None:
        self.embeddings_cache.clear()
    
    def find_duplicates(self, article: NewsArticle, existing_articles: List[NewsArticle]) -> List[str]:
        if not existing_articles:
            return []
        
        # Stage 1: Bi-Encoder (Fast Filtering)
        article_emb = self._get_embedding(article).reshape(1, -1)
        existing_embs = np.array([
            self._get_embedding(a) for a in existing_articles
        ])
        
        bi_scores = cosine_similarity(article_emb, existing_embs)[0]
        candidate_indices = np.where(bi_scores >= self.bi_threshold)[0]
        
        if len(candidate_indices) == 0:
            return []
        
        # Stage 2: Cross-Encoder (Precision Verification)
        candidates = [existing_articles[i] for i in candidate_indices]
        
        article_text = self._prepare_text(article)
        pairs = [[article_text, self._prepare_text(cand)] for cand in candidates]
        
        cross_scores = self.cross_encoder.predict(
            pairs,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Filter based on final probability threshold
        duplicate_ids = [
            candidates[idx].id 
            for idx, score in enumerate(cross_scores) 
            if score >= self.cross_threshold
        ]
        
        return duplicate_ids
    
    def find_duplicates_batch(self, articles: List[NewsArticle], existing_articles: List[NewsArticle]) -> dict[str, List[str]]:
        results = {}
        for article in articles:
            results[article.id] = self.find_duplicates(article, existing_articles)
        return results
    
    def consolidate_duplicates(self, articles: List[NewsArticle]) -> NewsArticle:
        """
        Merges duplicates, keeping the earliest article (by timestamp) as primary
        and aggregating unique sources.
        """
        if not articles:
            raise ValueError("Cannot consolidate empty article list")
        
        if len(articles) == 1:
            return articles[0]
        
        sorted_articles = sorted(articles, key=lambda x: x.timestamp)
        primary = sorted_articles[0]
        
        seen_sources = set()
        unique_sources = []
        
        for article in sorted_articles:
            # Normalize and split potentially comma-separated sources
            sources = [s.strip() for s in article.source.split(',')]
            for source in sources:
                if source not in seen_sources:
                    seen_sources.add(source)
                    unique_sources.append(source)
        
        primary.source = ", ".join(unique_sources)
        
        return primary
    
    def get_similarity_scores(self, article1: NewsArticle, article2: NewsArticle) -> Tuple[float, float]:
        """Returns (bi_encoder_score, cross_encoder_score) for debugging/analysis."""
        emb1 = self._get_embedding(article1).reshape(1, -1)
        emb2 = self._get_embedding(article2).reshape(1, -1)
        bi_score = cosine_similarity(emb1, emb2)[0][0]
        
        text1 = self._prepare_text(article1)
        text2 = self._prepare_text(article2)
        cross_score = self.cross_encoder.predict(
            [[text1, text2]],
            show_progress_bar=False,
            convert_to_numpy=True
        )[0]
        
        return float(bi_score), float(cross_score)