"""
Pydantic Schemas for LLM Structured Outputs
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional
from enum import Enum


# ENUMS FOR CONSTRAINED VALUES
class ImpactType(str, Enum):
    """Stock impact classification types."""
    DIRECT = "direct"
    SECTOR = "sector"
    REGULATORY = "regulatory"


class SentimentClassification(str, Enum):
    """Sentiment polarity categories."""
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"


class RelationshipType(str, Enum):
    """Supply chain relationship categories."""
    UPSTREAM_DEMAND_SHOCK = "upstream_demand_shock"
    DOWNSTREAM_SUPPLY_IMPACT = "downstream_supply_impact"


class QueryIntent(str, Enum):
    """Query routing strategy types."""
    DIRECT_ENTITY = "direct_entity"
    SECTOR_WIDE = "sector_wide"
    REGULATORY = "regulatory"
    SENTIMENT_DRIVEN = "sentiment_driven"
    CROSS_IMPACT = "cross_impact"
    SEMANTIC_SEARCH = "semantic_search"
    TEMPORAL = "temporal"


# ENTITY EXTRACTION SCHEMAS

class CompanyEntity(BaseModel):
    """Represents a company mentioned in financial news."""
    name: str = Field(..., description="Full company name as mentioned in text")
    ticker_symbol: Optional[str] = Field(None, description="Stock ticker symbol (e.g., HDFCBANK, TCS)")
    sector: Optional[str] = Field(None, description="Industry sector (e.g., Banking, IT, Pharma)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this extraction")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("Company name must be at least 2 characters")
        return v.strip()


class RegulatorEntity(BaseModel):
    """Represents a regulatory body mentioned in the news."""
    name: str = Field(..., description="Regulator name (e.g., RBI, SEBI, US FDA)")
    jurisdiction: Optional[str] = Field(None, description="Geographic/domain jurisdiction (e.g., India, US, Banking)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this extraction")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("Regulator name must be at least 2 characters")
        return v.strip()


class EventEntity(BaseModel):
    """Represents a market event mentioned in the news."""
    event_type: str = Field(..., description="Event category (e.g., dividend, merger, policy_change)")
    description: str = Field(..., description="Brief description of the event")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this extraction")
    
    @field_validator('event_type')
    @classmethod
    def validate_event_type(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("Event type must be specified")
        return v.strip().lower().replace(" ", "_")


class EntityExtractionSchema(BaseModel):
    """
    Complete entity extraction output schema.
    LLM extracts all relevant financial entities from news text.
    """
    companies: List[CompanyEntity] = Field(default_factory=list,description="List of companies mentioned in the article")
    sectors: List[str] = Field(default_factory=list,description="Industry sectors mentioned or inferred (e.g., Banking, IT, Auto)")
    regulators: List[RegulatorEntity] = Field(default_factory=list,description="Regulatory bodies mentioned in the article")
    people: List[str] = Field(default_factory=list,description="Names of key individuals mentioned (CEOs, policymakers, etc.)")
    events: List[EventEntity] = Field(default_factory=list,description="Market events identified in the article")
    confidence_score: float = Field(...,ge=0.0,le=1.0,description="Overall confidence in entity extraction quality")
    extraction_reasoning: Optional[str] = Field(None,description="Brief explanation of extraction logic")
    
    @field_validator('sectors')
    @classmethod
    def validate_sectors(cls, v: List[str]) -> List[str]:
        return [s.strip() for s in v if s and len(s.strip()) > 1]


# STOCK IMPACT MAPPING SCHEMAS

class StockImpact(BaseModel):
    """Represents the impact of news on a specific stock."""
    symbol: str = Field(..., description="Stock ticker symbol (e.g., HDFCBANK, RELIANCE)")
    company_name: str = Field(..., description="Full company name for clarity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in impact assessment")
    impact_type: ImpactType = Field(..., description="Type of impact: direct, sector-wide, or regulatory")
    reasoning: str = Field(..., description="Explanation for why this stock is impacted")
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        if not v or len(v.strip()) < 1:
            raise ValueError("Stock symbol cannot be empty")
        return v.strip().upper()


class StockImpactSchema(BaseModel):
    """
    Complete stock impact mapping output schema.
    LLM maps news to affected stocks with confidence and reasoning.
    """
    impacted_stocks: List[StockImpact] = Field(default_factory=list,description="List of stocks impacted by the news")
    overall_market_impact: Optional[str] = Field(None,description="Broader market implications (if any)")
    confidence_score: float = Field(...,ge=0.0,le=1.0,description="Overall confidence in impact analysis")


# SENTIMENT ANALYSIS SCHEMAS

class SentimentAnalysisSchema(BaseModel):
    """
    Sentiment analysis output schema.
    LLM determines market sentiment with detailed reasoning.
    """
    classification: SentimentClassification = Field(...,description="Overall sentiment: Bullish, Bearish, or Neutral")
    confidence_score: float = Field(...,ge=0.0,le=100.0,description="Confidence in sentiment classification (0-100 scale)")
    key_factors: List[str] = Field(...,min_length=1,description="Bullet points explaining the sentiment decision")
    signal_strength: float = Field(...,ge=0.0,le=100.0,description="Trading signal strength based on sentiment intensity (0-100)")
    sentiment_breakdown: Optional[dict] = Field(None,description="Detailed percentage breakdown: {bullish: %, bearish: %, neutral: %}")
    entity_influence: Optional[dict] = Field(None,description="How specific entities influenced the sentiment")
    
    @field_validator('key_factors')
    @classmethod
    def validate_key_factors(cls, v: List[str]) -> List[str]:
        if not v or len(v) < 1:
            raise ValueError("At least one key factor must be provided")
        return [f.strip() for f in v if f and len(f.strip()) > 5]


# SUPPLY CHAIN IMPACT SCHEMAS

class CrossImpact(BaseModel):
    """Represents cross-sectoral impact via supply chain relationships."""
    source_sector: str = Field(..., description="Sector where the news originated")
    target_sector: str = Field(..., description="Sector that will be impacted")
    relationship_type: RelationshipType = Field(...,description="Type of relationship: upstream demand shock or downstream supply impact")
    impact_score: float = Field(...,ge=0.0,le=100.0,description="Impact magnitude score (0-100)")
    dependency_weight: float = Field(...,ge=0.0,le=1.0,description="Strength of dependency between sectors (0-1)")
    reasoning: str = Field(...,description="Natural language explanation of the cross-sectoral impact")
    impacted_stocks: List[str] = Field(default_factory=list,description="Stock symbols in the target sector that will be affected")
    time_horizon: Optional[str] = Field(None,description="Expected timeframe for impact (e.g., immediate, short-term, long-term)")
    
    @field_validator('source_sector', 'target_sector')
    @classmethod
    def validate_sector(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError("Sector name must be at least 2 characters")
        return v.strip()


class SupplyChainImpactSchema(BaseModel):
    """
    Complete supply chain impact analysis schema.
    LLM identifies upstream and downstream effects of news.
    """
    upstream_impacts: List[CrossImpact] = Field(default_factory=list,description="Impacts on upstream suppliers/dependencies")
    downstream_impacts: List[CrossImpact] = Field(default_factory=list,description="Impacts on downstream customers/consumers")
    reasoning: str = Field(...,description="Overall reasoning for supply chain impact assessment")
    confidence_score: float = Field(...,ge=0.0,le=1.0,description="Overall confidence in supply chain analysis")
    total_sectors_impacted: int = Field(...,ge=0,description="Total number of sectors identified as impacted")


# QUERY ROUTER SCHEMA

class QueryRouterSchema(BaseModel):
    """
    Query routing schema for LLM-based query understanding.
    Determines optimal search strategy and extracts relevant entities.
    """
    strategy: QueryIntent = Field(..., description="Primary search strategy to execute")
    entities: List[str] = Field(
        default_factory=list,
        description="Company names identified in query"
    )
    stock_symbols: List[str] = Field(
        default_factory=list,
        description="Stock ticker symbols extracted or inferred"
    )
    sectors: List[str] = Field(
        default_factory=list,
        description="Industry sectors mentioned or implied"
    )
    regulators: List[str] = Field(
        default_factory=list,
        description="Regulatory bodies mentioned in query"
    )
    sentiment_filter: Optional[str] = Field(
        None,
        description="Sentiment filter: Bullish, Bearish, or Neutral"
    )
    temporal_scope: Optional[str] = Field(
        None,
        description="Time scope: recent, last_week, last_month, etc."
    )
    refined_query: str = Field(
        ...,
        description="Optimized semantic search query for vector retrieval"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in routing decision (0.0-1.0)"
    )
    reasoning: str = Field(
        ...,
        description="Explanation for why this strategy was chosen"
    )
    
    @field_validator('sentiment_filter')
    @classmethod
    def validate_sentiment_filter(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            valid_sentiments = ["Bullish", "Bearish", "Neutral"]
            if v not in valid_sentiments:
                raise ValueError(f"Sentiment filter must be one of {valid_sentiments}")
        return v
    
    @field_validator('entities', 'stock_symbols', 'sectors', 'regulators')
    @classmethod
    def validate_list_fields(cls, v: List[str]) -> List[str]:
        return [item.strip() for item in v if item and len(item.strip()) > 0]
    
    @field_validator('refined_query')
    @classmethod
    def validate_refined_query(cls, v: str) -> str:
        if not v or len(v.strip()) < 3:
            raise ValueError("Refined query must be at least 3 characters")
        return v.strip()


# DEDUPLICATION SCHEMA

class DeduplicationSchema(BaseModel):
    """
    Schema for LLM-based semantic deduplication.
    LLM determines if two articles cover the same story.
    """
    is_duplicate: bool = Field(...,description="Whether the articles cover the same underlying story")
    confidence_score: float = Field(...,ge=0.0,le=1.0,description="Confidence in duplication assessment")
    reasoning: str = Field(...,description="Explanation for duplicate/unique determination")
    key_overlaps: List[str] = Field(default_factory=list,description="Key facts/entities that overlap between articles (if duplicate)")
    unique_aspects: List[str] = Field(default_factory=list,description="Unique information present in each article (if not duplicate)")
    similarity_score: float = Field(...,ge=0.0,le=1.0,description="Semantic similarity score between articles"
    )


# QUERY EXPANSION SCHEMA

class QueryExpansionSchema(BaseModel):
    """
    Schema for LLM-based query understanding and expansion.
    LLM interprets user intent and generates context-aware queries.
    """
    original_query: str = Field(..., description="Original user query")
    interpreted_intent: str = Field(...,description="LLM's interpretation of what the user wants to find")
    primary_query: str = Field(...,description="Refined primary search query")
    context_queries: List[str] = Field(default_factory=list,max_length=5,description="Additional context-aware queries for comprehensive retrieval")
    identified_entities: dict = Field(default_factory=dict,description="Entities identified in the query: {companies: [], sectors: [], regulators: []}")
    query_type: str = Field(...,description="Query classification: direct_mention, sector_wide, regulator_filter, thematic")
    suggested_filters: Optional[dict] = Field(None,description="Suggested filters for search: {sentiment: str, date_range: str, etc.}")
    confidence_score: float = Field(...,ge=0.0,le=1.0,description="Confidence in query interpretation")


# VALIDATION HELPERS

def validate_schema_output(schema_class: type[BaseModel], llm_output: dict) -> BaseModel:
    """
    Utility function to validate and parse LLM output against a Pydantic schema.
    
    Args:
        schema_class: Pydantic model class to validate against
        llm_output: Raw dictionary output from LLM
        
    Returns:
        Validated Pydantic model instance
        
    Raises:
        ValidationError: If LLM output doesn't match schema
    """
    return schema_class.model_validate(llm_output)