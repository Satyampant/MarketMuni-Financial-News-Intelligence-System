from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from news_storage import NewsArticle
from typing import List
import numpy as np

class DeduplicationAgent:
    def __init__(self, similarity_threshold: float = 0.85):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.threshold = similarity_threshold
        self.embeddings_cache = {}
    
    def _get_embedding(self, article: NewsArticle) -> np.ndarray:
        if article.id not in self.embeddings_cache:
            text = f"{article.title} {article.content}"
            self.embeddings_cache[article.id] = self.model.encode(text, normalize_embeddings=True)
        return self.embeddings_cache[article.id]
    
    def find_duplicates(self, article: NewsArticle, existing_articles: List[NewsArticle]) -> List[str]:
        if not existing_articles:
            return []
        article_embedding = self._get_embedding(article).reshape(1, -1)
        duplicate_ids = []
        for existing in existing_articles:
            existing_embedding = self._get_embedding(existing).reshape(1, -1)
            similarity = cosine_similarity(article_embedding, existing_embedding)[0][0]
            if similarity > self.threshold:
                duplicate_ids.append(existing.id)
        return duplicate_ids
    
    def consolidate_duplicates(self, articles: List[NewsArticle]) -> NewsArticle:
        sorted_articles = sorted(articles, key=lambda x: x.timestamp)
        primary = sorted_articles[0]
        sources = ", ".join(set(a.source for a in articles))
        primary.source = sources
        return primary