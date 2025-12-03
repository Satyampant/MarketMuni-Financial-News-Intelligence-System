from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime

from app.core.models import NewsArticle
from app.api.schemas import (
    ArticleInput, ArticleOutput, IngestResponse, 
    QueryResponse, StatsResponse, SentimentDetailResponse
)

# Import Services & Workflows
from app.services.storage import NewsStorage
from app.services.vector_store import VectorStore
from app.agents.deduplication import DeduplicationAgent
from app.agents.entity_extraction import EntityExtractor
from app.agents.stock_impact import StockImpactMapper
from app.agents.supply_chain import SupplyChainImpactMapper
from app.workflows.graph import NewsIntelligenceGraph
from app.core.config import Paths

router = APIRouter()

# Initialize Components (Singleton)
# Using Paths.CHROMA_DB ensures persistence in the data folder
storage = NewsStorage()
vector_store = VectorStore(
    collection_name="financial_news",
    persist_directory=str(Paths.CHROMA_DB)
)

dedup_agent = DeduplicationAgent()
entity_extractor = EntityExtractor()
stock_mapper = StockImpactMapper()
supply_chain_mapper = SupplyChainImpactMapper()

news_graph = NewsIntelligenceGraph(
    storage=storage,
    vector_store=vector_store,
    dedup_agent=dedup_agent,
    entity_extractor=entity_extractor,
    stock_mapper=stock_mapper,
    supply_chain_mapper=supply_chain_mapper,
    sentiment_method="hybrid"
)

@router.get("/", tags=["General"])
async def root():
    return {
        "message": "Financial News Intelligence API (Modular)",
        "version": "1.0.0",
        "docs": "/docs"
    }

@router.get("/health", tags=["General"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "storage": storage.article_count(),
            "vector_store": vector_store.count()
        }
    }

@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_article(article_input: ArticleInput):
    try:
        article = NewsArticle(
            id=article_input.id,
            title=article_input.title,
            content=article_input.content,
            source=article_input.source,
            timestamp=article_input.timestamp,
            raw_text=article_input.raw_text or article_input.content
        )
        
        result = news_graph.run_pipeline(article)
        
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        
        stats = result.get("stats", {})
        entities_stats = stats.get("entities_extracted", {})
        impact_breakdown = stats.get("impact_breakdown", {})
        
        return IngestResponse(
            success=True,
            article_id=article.id,
            message="Article processed successfully",
            is_duplicate=stats.get("is_duplicate", False),
            duplicates_found=stats.get("duplicates_found", 0),
            entities_extracted=entities_stats,
            stocks_impacted=stats.get("stocks_impacted", 0),
            sentiment_classification=stats.get("sentiment_classification"),
            sentiment_confidence=stats.get("sentiment_confidence"),
            signal_strength=stats.get("sentiment_signal_strength"),
            stats={
                "ingestion_time": stats.get("ingestion_time"),
                "impact_breakdown": impact_breakdown,
                "total_articles": stats.get("total_articles"),
                "indexed": stats.get("indexed", False),
                "sentiment_analyzed": stats.get("sentiment_analyzed", False)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@router.get("/query", response_model=QueryResponse, tags=["Query"])
async def query_articles(
    q: str = Query(..., description="Natural language query"),
    top_k: int = Query(10, ge=1, le=50),
    filter_by_sentiment: Optional[str] = Query(None, description="Bullish, Bearish, or Neutral")
):
    try:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        result = news_graph.run_query(q, sentiment_filter=filter_by_sentiment)
        
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
            
        articles = result.get("query_results", [])[:top_k]
        
        article_outputs = []
        for article in articles:
            sentiment_dict = article.sentiment if article.has_sentiment() else None
            article_outputs.append(ArticleOutput(
                id=article.id,
                title=article.title,
                content=article.content,
                source=article.source,
                timestamp=article.timestamp.isoformat(),
                entities=getattr(article, "entities", None),
                impacted_stocks=getattr(article, "impacted_stocks", None),
                relevance_score=getattr(article, "relevance_score", None),
                sentiment=sentiment_dict
            ))
            
        return QueryResponse(
            query=q,
            results_count=len(article_outputs),
            articles=article_outputs,
            stats=result.get("stats", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@router.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats():
    try:
        stats = news_graph.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

@router.get("/article/{article_id}", tags=["Retrieval"])
async def get_article(article_id: str):
    article = storage.get_by_id(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
        
    sentiment_dict = article.sentiment if article.has_sentiment() else None
    
    return ArticleOutput(
        id=article.id,
        title=article.title,
        content=article.content,
        source=article.source,
        timestamp=article.timestamp.isoformat(),
        entities=getattr(article, "entities", None),
        impacted_stocks=getattr(article, "impacted_stocks", None),
        sentiment=sentiment_dict
    )

@router.get("/article/{article_id}/sentiment", response_model=SentimentDetailResponse, tags=["Sentiment"])
async def get_article_sentiment(article_id: str):
    article = storage.get_by_id(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    if not article.has_sentiment():
        raise HTTPException(status_code=404, detail="No sentiment data for article")
        
    data = article.get_sentiment()
    return SentimentDetailResponse(
        article_id=article.id,
        title=article.title,
        classification=data.classification,
        confidence_score=data.confidence_score,
        signal_strength=data.signal_strength,
        sentiment_breakdown=data.sentiment_breakdown,
        analysis_method=data.analysis_method,
        timestamp=data.timestamp
    )