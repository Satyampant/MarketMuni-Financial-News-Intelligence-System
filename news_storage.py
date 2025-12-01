from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
import json

@dataclass
class StockImpact:
    symbol: str
    confidence: float
    impact_type: str  # "direct" | "sector" | "regulatory"
    source_entity: Optional[str] = None
    rule: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class NewsArticle:
    id: str
    title: str
    content: str
    source: str
    timestamp: datetime
    raw_text: str = field(default="")
    impacted_stocks: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)



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
        """Return total number of stored articles."""
        return len(self.articles)


def load_mock_dataset(filepath: str) -> List[NewsArticle]:
    """Load mock news dataset from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [NewsArticle(**article) for article in data]