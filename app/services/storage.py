from typing import List, Optional
from app.core.models import NewsArticle

class NewsStorage:
    def __init__(self):
        self.articles = {}
    
    def add_article(self, article: NewsArticle) -> None:
        self.articles[article.id] = article
    
    def get_all_articles(self) -> List[NewsArticle]:
        return list(self.articles.values())
    
    def get_by_id(self, article_id: str) -> Optional[NewsArticle]:
        return self.articles.get(article_id)
    
    def article_count(self) -> int:
        return len(self.articles)
    
    def get_articles_with_sentiment(self) -> List[NewsArticle]:
        return [a for a in self.articles.values() if a.has_sentiment()]