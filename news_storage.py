from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json

@dataclass
class NewsArticle:
    id: str
    title: str
    content: str
    source: str
    timestamp: datetime
    raw_text: str = field(default="")
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

class NewsStorage:
    def __init__(self):
        self.articles = {}
    
    def add_article(self, article: NewsArticle) -> None:
        self.articles[article.id] = article
    
    def get_all_articles(self) -> list[NewsArticle]:
        return list(self.articles.values())
    
    def get_by_id(self, article_id: str) -> Optional[NewsArticle]:
        return self.articles.get(article_id)

def load_mock_dataset(filepath: str) -> list[NewsArticle]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [NewsArticle(**article) for article in data]