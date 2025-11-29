from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from news_storage import NewsArticle
from typing import List
import numpy as np

class DeduplicationAgent:
    def __init__(self, similarity_threshold: float = 0.80): # High threshold for final precision
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        # CrossEncoder is much more accurate for checking "are these two distinct texts actually the same?"
        self.reranker = CrossEncoder('cross-encoder/stsb-distilroberta-base')
        self.threshold = similarity_threshold
        self.embeddings_cache = {}
    
    def _get_embedding(self, article: NewsArticle) -> np.ndarray:
        if article.id not in self.embeddings_cache:
            text = f"{article.title}. {article.content}"
            self.embeddings_cache[article.id] = self.embedding_model.encode(text)
        return self.embeddings_cache[article.id]
    
    def find_duplicates(self, article: NewsArticle, existing_articles: List[NewsArticle]) -> List[str]:
        if not existing_articles: return []
        
        # Step 1: Fast filtering (Bi-Encoder) - Find candidates > 0.50 similarity
        article_emb = self._get_embedding(article).reshape(1, -1)
        existing_embs = np.array([self._get_embedding(a) for a in existing_articles])
        
        # Calculate cosine sim for all at once (vectorized)
        bi_scores = cosine_similarity(article_emb, existing_embs)[0]
        candidate_indices = [i for i, score in enumerate(bi_scores) if score > 0.50]
        
        duplicate_ids = []
        if not candidate_indices:
            return duplicate_ids

        # Step 2: High Precision verification (Cross-Encoder)
        # Prepare pairs: [(new_article_text, candidate_1_text), ...]
        candidates = [existing_articles[i] for i in candidate_indices]
        pairs = [[f"{article.title} {article.content}", f"{cand.title} {cand.content}"] for cand in candidates]
        
        cross_scores = self.reranker.predict(pairs)
        
        # Normalize scores to 0-1 range (stsb model outputs 0-1 usually, but safer to check)
        for idx, score in enumerate(cross_scores):
            # Cross-encoder scores are very accurate. >0.7 usually implies same meaning.
            # Using self.threshold (0.90) ensures extremely high confidence.
            if score > 0.5: # Adjusted because CrossEncoder range varies, but 0.5+ on STSB is usually semantically equal
                 duplicate_ids.append(candidates[idx].id)
                 
        return duplicate_ids
    
    def consolidate_duplicates(self, articles: List[NewsArticle]) -> NewsArticle:
        sorted_articles = sorted(articles, key=lambda x: x.timestamp)
        primary = sorted_articles[0]
        # Deduplicate sources while preserving order
        unique_sources = sorted(list(set(a.source for a in articles)))
        primary.source = ", ".join(unique_sources)
        return primary