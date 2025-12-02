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
class SentimentData:
    classification: str  # "Bullish" | "Bearish" | "Neutral"
    confidence_score: float  # 0-100 scale
    signal_strength: float  # 0-100 scale
    sentiment_breakdown: Dict[str, Any]  #  type hint Any to allow metadata strings
    analysis_method: str  # "rule_based" | "ml_classifier" | "hybrid"
    timestamp: Optional[str] = None  # ISO format timestamp
    
    def __post_init__(self):
        valid_classifications = ["Bullish", "Bearish", "Neutral"]
        if self.classification not in valid_classifications:
            raise ValueError(
                f"Invalid classification: {self.classification}. "
                f"Must be one of {valid_classifications}"
            )
        
        if not 0 <= self.confidence_score <= 100:
            raise ValueError(f"confidence_score must be 0-100, got {self.confidence_score}")
        
        if not 0 <= self.signal_strength <= 100:
            raise ValueError(f"signal_strength must be 0-100, got {self.signal_strength}")
        
        # Validate sentiment_breakdown scores
        for key, value in self.sentiment_breakdown.items():
            if isinstance(value, (int, float)):
                if not 0 <= value <= 100:
                    raise ValueError(
                        f"sentiment_breakdown[{key}] must be 0-100, got {value}"
                    )
        
        # Set timestamp if not provided
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentimentData':
        """Create SentimentData from dictionary"""
        return cls(**data)


@dataclass
class NewsArticle:
    id: str
    title: str
    content: str
    source: str
    timestamp: datetime
    raw_text: str = field(default="")
    impacted_stocks: List[Dict] = field(default_factory=list)
    sentiment: Optional[Dict[str, Any]] = None  # Sentiment analysis data
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
    
    def set_sentiment(self, sentiment_data: SentimentData) -> None:
        self.sentiment = sentiment_data.to_dict()
    
    def get_sentiment(self) -> Optional[SentimentData]:
        if self.sentiment is None:
            return None
        return SentimentData.from_dict(self.sentiment)
    
    def has_sentiment(self) -> bool:
        """Check if article has sentiment analysis data"""
        return self.sentiment is not None


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
        """Get all articles that have sentiment analysis data"""
        return [
            article for article in self.articles.values()
            if article.has_sentiment()
        ]
    
    def get_articles_by_sentiment(self, classification: str) -> List[NewsArticle]:
        return [
            article for article in self.articles.values()
            if article.sentiment and article.sentiment.get("classification") == classification
        ]


def load_mock_dataset(filepath: str) -> List[NewsArticle]:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [NewsArticle(**article) for article in data]