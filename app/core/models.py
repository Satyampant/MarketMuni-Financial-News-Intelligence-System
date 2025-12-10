"""
Updated app/core/models.py
Adds MongoDB document conversion methods (Task 3)
"""

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
    analysis_method: str = "llm"
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
class QueryRouting:
    """Query routing result with MongoDB filter support."""
    entities: List[str]
    sectors: List[str]
    stock_symbols: List[str]
    sentiment_filter: Optional[str]
    refined_query: str
    strategy: QueryIntent
    confidence: float
    reasoning: str
    regulators: List[str] = None
    temporal_scope: Optional[str] = None
    
    def __post_init__(self):
        if self.regulators is None:
            self.regulators = []

@dataclass
class NewsArticle:
    id: str
    title: str
    content: str
    source: str
    timestamp: datetime
    raw_text: str = field(default="")
    
    # Rich entity data (EntityExtractionSchema as dict)
    entities_rich: Optional[Dict[str, Any]] = None
    
    # Legacy entity storage (for backward compatibility)
    entities: Optional[Dict[str, List[str]]] = None
    
    impacted_stocks: List[Dict] = field(default_factory=list)
    sentiment: Optional[Dict[str, Any]] = None
    cross_impacts: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
    
    # ========================================================================
    # TASK 3: MONGODB DOCUMENT CONVERSION METHODS
    # ========================================================================
    
    def to_mongo_document(self) -> dict:
        """
        Convert NewsArticle to MongoDB-compatible document format.
        Flattens nested structures for efficient indexing and querying.
        """
        # Base document structure
        mongo_doc = {
            "id": self.id,
            "title": self.title,
            "content": self.content,  # KEPT for deduplication hydration
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "raw_text": self.raw_text
        }
        
        # Flatten entities (legacy format for backward compatibility)
        if self.entities:
            mongo_doc["entities"] = {
                "Companies": self.entities.get("Companies", []),
                "Sectors": self.entities.get("Sectors", []),
                "Regulators": self.entities.get("Regulators", []),
                "People": self.entities.get("People", []),
                "Events": self.entities.get("Events", [])
            }
        else:
            mongo_doc["entities"] = {}
        
        # Store rich entity data if available
        if self.entities_rich:
            mongo_doc["entities_rich"] = self.entities_rich
        
        # Flatten impacted stocks
        if self.impacted_stocks:
            mongo_doc["impacted_stocks"] = [
                {
                    "symbol": stock.get("symbol", ""),
                    "company_name": stock.get("company_name", ""),
                    "confidence": stock.get("confidence", 0.0),
                    "impact_type": stock.get("impact_type", ""),
                    "reasoning": stock.get("reasoning", "")
                }
                for stock in self.impacted_stocks
            ]
        else:
            mongo_doc["impacted_stocks"] = []
        
        # Flatten sentiment
        if self.sentiment:
            mongo_doc["sentiment"] = {
                "classification": self.sentiment.get("classification", "Neutral"),
                "confidence_score": self.sentiment.get("confidence_score", 0.0),
                "signal_strength": self.sentiment.get("signal_strength", 0.0),
                "sentiment_breakdown": self.sentiment.get("sentiment_breakdown", {}),
                "analysis_method": self.sentiment.get("analysis_method", "llm"),
                "timestamp": self.sentiment.get("timestamp", "")
            }
        else:
            mongo_doc["sentiment"] = None
        
        # Flatten cross impacts (supply chain)
        if self.cross_impacts:
            mongo_doc["cross_impacts"] = [
                {
                    "source_sector": impact.get("source_sector", ""),
                    "target_sector": impact.get("target_sector", ""),
                    "relationship_type": impact.get("relationship_type", ""),
                    "impact_score": impact.get("impact_score", 0.0),
                    "dependency_weight": impact.get("dependency_weight", 0.0),
                    "reasoning": impact.get("reasoning", ""),
                    "impacted_stocks": impact.get("impacted_stocks", []),
                    "time_horizon": impact.get("time_horizon", "")
                }
                for impact in self.cross_impacts
            ]
        else:
            mongo_doc["cross_impacts"] = []
        
        return mongo_doc
    
    @classmethod
    def from_mongo_document(cls, doc: dict) -> 'NewsArticle':
        """
        Reconstruct NewsArticle from MongoDB document.
        Parses ISO timestamps and validates required fields.
        """
        # Validate required fields
        required_fields = ["id", "title", "content", "source", "timestamp"]
        missing_fields = [field for field in required_fields if field not in doc]
        if missing_fields:
            raise ValueError(f"Missing required fields in MongoDB document: {missing_fields}")
        
        # Parse timestamp
        timestamp = doc["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif not isinstance(timestamp, datetime):
            raise ValueError(f"Invalid timestamp format: {type(timestamp)}")
        
        # Create article instance
        article = cls(
            id=doc["id"],
            title=doc["title"],
            content=doc["content"],
            source=doc["source"],
            timestamp=timestamp,
            raw_text=doc.get("raw_text", doc.get("content", ""))  # Use content as raw_text if not separately stored
        )
        
        # Restore entities (legacy format)
        if "entities" in doc and doc["entities"]:
            article.entities = doc["entities"]
        
        # Restore rich entity data
        if "entities_rich" in doc and doc["entities_rich"]:
            article.entities_rich = doc["entities_rich"]
        
        # Restore impacted stocks
        if "impacted_stocks" in doc and doc["impacted_stocks"]:
            article.impacted_stocks = doc["impacted_stocks"]
        
        # Restore sentiment
        if "sentiment" in doc and doc["sentiment"]:
            article.sentiment = doc["sentiment"]
        
        # Restore cross impacts
        if "cross_impacts" in doc and doc["cross_impacts"]:
            article.cross_impacts = doc["cross_impacts"]
        
        return article
    

    
    def set_entities_rich(self, entities_schema: 'EntityExtractionSchema') -> None:
        """
        Store rich entity data from LLM extraction.
        Also populates legacy 'entities' dict for backward compatibility.
        
        Args:
            entities_schema: EntityExtractionSchema from LLM extraction
        """
        from app.core.llm_schemas import EntityExtractionSchema
        
        # Store rich data
        if hasattr(entities_schema, 'model_dump'):
            self.entities_rich = entities_schema.model_dump()
        else:
            self.entities_rich = asdict(entities_schema)
        
        # Populate legacy format for backward compatibility
        self.entities = {
            "Companies": [c.get("name") if isinstance(c, dict) else c.name 
                         for c in self.entities_rich.get("companies", [])],
            "Sectors": self.entities_rich.get("sectors", []),
            "Regulators": [r.get("name") if isinstance(r, dict) else r.name 
                          for r in self.entities_rich.get("regulators", [])],
            "People": self.entities_rich.get("people", []),
            "Events": [e.get("event_type") if isinstance(e, dict) else e.event_type 
                      for e in self.entities_rich.get("events", [])]
        }
    
    def get_entities_rich(self) -> Optional[Dict[str, Any]]:
        """Get rich entity data with tickers and confidence scores."""
        return self.entities_rich
    
    def get_company_tickers(self) -> List[Dict[str, str]]:
        """
        Extract company ticker symbols from rich entity data.
        
        Returns:
            List of dicts: [{"name": "HDFC Bank", "ticker": "HDFCBANK", "sector": "Banking"}, ...]
        """
        if not self.entities_rich:
            return []
        
        companies = self.entities_rich.get("companies", [])
        return [
            {
                "name": c.get("name"),
                "ticker": c.get("ticker_symbol"),
                "sector": c.get("sector"),
                "confidence": c.get("confidence")
            }
            for c in companies
            if c.get("ticker_symbol")
        ]
    
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