from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any

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
    sentiment_breakdown: Dict[str, Any]
    analysis_method: str  # "rule_based" | "ml_classifier" | "hybrid"
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentimentData':
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
    sentiment: Optional[Dict[str, Any]] = None  
    cross_impacts: List[Dict] = field(default_factory=list)
    
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
        return self.sentiment is not None

    def set_cross_impacts(self, impacts: List[Dict]) -> None:
        self.cross_impacts = impacts
    
    def has_cross_impacts(self) -> bool:
        return len(self.cross_impacts) > 0