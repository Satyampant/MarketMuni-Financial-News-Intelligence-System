"""
Updated app/api/routes.py - Full LLM Pipeline with LLM Supply Chain Analysis
All agents now use LLM-based approach.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime

from app.core.models import NewsArticle
from app.api.schemas import (
    ArticleInput, ArticleOutput, IngestResponse, 
    QueryResponse, StatsResponse, SentimentDetailResponse
)
from app.services.storage import NewsStorage
from app.services.vector_store import VectorStore
from app.agents.deduplication import DeduplicationAgent
from app.agents.llm_entity_extractor import LLMEntityExtractor
from app.agents.llm_stock_mapper import LLMStockImpactMapper
from app.agents.llm_sentiment import LLMSentimentAnalyzer
from app.agents.llm_supply_chain import LLMSupplyChainAnalyzer  # NEW: LLM supply chain
from app.workflows.graph import NewsIntelligenceGraph
from app.core.config import Paths
from app.core.config_loader import get_config

config = get_config()
router = APIRouter()

# Service initialization
storage = NewsStorage()
vector_store = VectorStore(
    collection_name=config.vector_store.collection_name,
    persist_directory=str(Paths.CHROMA_DB)
)

dedup_agent = DeduplicationAgent()

# LLM entity extraction
print("✓ Initializing LLM-based entity extraction")
entity_extractor = LLMEntityExtractor(
    enable_caching=config.performance.cache_embeddings
)

# LLM stock impact mapper
print("✓ Initializing LLM-based stock impact mapper")
stock_mapper = LLMStockImpactMapper()

# LLM sentiment analyzer
print("✓ Initializing LLM-based sentiment analyzer")
sentiment_analyzer = LLMSentimentAnalyzer(
    use_entity_context=True
)

# NEW: LLM supply chain analyzer
print("✓ Initializing LLM-based supply chain analyzer")
supply_chain_analyzer = LLMSupplyChainAnalyzer()

# Orchestrator setup with full LLM pipeline
news_graph = NewsIntelligenceGraph(
    storage=storage,
    vector_store=vector_store,
    dedup_agent=dedup_agent,
    entity_extractor=entity_extractor,
    stock_mapper=stock_mapper,
    sentiment_analyzer=sentiment_analyzer,
    supply_chain_analyzer=supply_chain_analyzer  # NEW: Pass LLM supply chain analyzer
)

@router.get("/", tags=["General"])
async def root():
    return {
        "message": "MarketMuni - Financial News Intelligence API",
        "version": "2.0.0",
        "features": {
            "entity_extraction": "llm",
            "stock_impact_mapping": "llm",
            "sentiment_analysis": "llm",
            "supply_chain_analysis": "llm",  # UPDATED
            "deduplication": "semantic"
        },
        "docs": "/docs"
    }

@router.get("/health", tags=["General"])
async def health_check():
    """
    Comprehensive health check including Redis cache and LLM services.
    """
    cache_stats = entity_extractor.get_cache_stats()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "storage": {
                "status": "healthy",
                "articles_count": storage.article_count()
            },
            "vector_store": {
                "status": "healthy",
                "count": vector_store.count()
            },
            "entity_extraction": {
                "method": "llm",
                "cache_type": cache_stats.get("cache_type", "none"),
                "cache_enabled": cache_stats.get("cache_enabled", False)
            },
            "stock_impact_mapping": {
                "method": "llm",
                "status": "operational"
            },
            "sentiment_analysis": {
                "method": "llm",
                "status": "operational"
            },
            "supply_chain_analysis": {
                "method": "llm",
                "status": "operational"
            }
        }
    }
    
    # Add Redis-specific stats if connected
    if cache_stats.get("connected"):
        health_status["components"]["redis_cache"] = {
            "status": "healthy",
            "host": cache_stats.get("host"),
            "port": cache_stats.get("port"),
            "db": cache_stats.get("db"),
            "cached_keys": cache_stats.get("cached_keys"),
            "memory_used": cache_stats.get("memory_used"),
            "ttl_seconds": cache_stats.get("ttl_seconds")
        }
    else:
        health_status["components"]["redis_cache"] = {
            "status": "unavailable",
            "message": "Redis not connected - using in-memory fallback"
        }
        if config.redis.enabled:
            health_status["status"] = "degraded"
    
    return health_status


@router.get("/cache/stats", tags=["Cache"])
async def get_cache_stats():
    """Get detailed cache statistics."""
    return entity_extractor.get_cache_stats()


@router.post("/cache/clear", tags=["Cache"])
async def clear_cache(article_id: Optional[str] = None):
    """
    Clear cache entries.
    
    Args:
        article_id: Specific article to clear (None = clear all)
    """
    cleared = entity_extractor.clear_cache(article_id)
    
    return {
        "success": True,
        "cleared_count": cleared,
        "article_id": article_id,
        "message": f"Cleared {cleared} cache entries"
    }

@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_article(article_input: ArticleInput):
    """
    Ingest a financial news article with full LLM pipeline.
    - LLM entity extraction (companies with tickers, sectors, regulators, events)
    - LLM stock impact mapping with reasoning
    - LLM sentiment analysis with key factors
    - LLM supply chain impact analysis with cross-sectoral reasoning
    """
    try:
        article = NewsArticle(
            id=article_input.id,
            title=article_input.title,
            content=article_input.content,
            source=article_input.source,
            timestamp=article_input.timestamp,
            raw_text=article_input.raw_text or article_input.content
        )
        
        # Run graph pipeline
        result = news_graph.run_pipeline(article)
        
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Extract pipeline stats for response
        stats = result.get("stats", {})
        entities_stats = stats.get("entities_extracted", {})
        impact_breakdown = stats.get("impact_breakdown", {})
        
        return IngestResponse(
            success=True,
            article_id=article.id,
            message="Article processed successfully with full LLM pipeline",
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
                "sentiment_analyzed": stats.get("sentiment_analyzed", False),
                "tickers_extracted": stats.get("tickers_extracted", 0),
                "avg_company_confidence": stats.get("avg_company_confidence"),
                "extraction_reasoning": stats.get("extraction_reasoning"),
                "stock_impact_method": stats.get("stock_impact_method", "llm"),
                "sentiment_method": stats.get("sentiment_method", "llm"),
                "sentiment_key_factors_count": stats.get("sentiment_key_factors_count", 0),
                "overall_market_impact": stats.get("overall_market_impact"),
                "supply_chain_method": stats.get("supply_chain_method", "llm"),
                "cross_impacts_found": stats.get("cross_impacts_found", 0),
                "upstream_dependencies": stats.get("upstream_dependencies", 0),
                "downstream_impacts": stats.get("downstream_impacts", 0),
                "total_sectors_impacted": stats.get("total_sectors_impacted", 0)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@router.get("/query", response_model=QueryResponse, tags=["Query"])
async def query_articles(
    q: str = Query(..., description="Natural language query"),
    top_k: int = Query(None, ge=1, le=50),
    filter_by_sentiment: Optional[str] = Query(None, description="Bullish, Bearish, or Neutral")
):
    config = get_config()
    if top_k is None:
        top_k = config.query_processing.default_top_k

    try:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        result = news_graph.run_query(q, sentiment_filter=filter_by_sentiment)
        
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
            
        # Format results
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
    """
    Retrieve article by ID with rich entity data, stock impact reasoning, and supply chain impacts.
    """
    article = storage.get_by_id(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
        
    sentiment_dict = article.sentiment if article.has_sentiment() else None
    
    # Include rich entity data if available
    response_data = {
        "id": article.id,
        "title": article.title,
        "content": article.content,
        "source": article.source,
        "timestamp": article.timestamp.isoformat(),
        "entities": article.entities,
        "impacted_stocks": article.impacted_stocks,
        "sentiment": sentiment_dict,
        "cross_impacts": article.cross_impacts if hasattr(article, 'cross_impacts') else []
    }
    
    # Add rich entity data with tickers
    if article.entities_rich:
        response_data["entities_rich"] = article.entities_rich
        response_data["company_tickers"] = article.get_company_tickers()
    
    # Add stock impact reasoning (from LLM)
    if article.impacted_stocks:
        response_data["stock_impact_summary"] = {
            "total_stocks": len(article.impacted_stocks),
            "direct_impacts": sum(1 for s in article.impacted_stocks if s["impact_type"] == "direct"),
            "sector_impacts": sum(1 for s in article.impacted_stocks if s["impact_type"] == "sector"),
            "regulatory_impacts": sum(1 for s in article.impacted_stocks if s["impact_type"] == "regulatory")
        }
    
    # Add supply chain impact summary (from LLM)
    if article.has_cross_impacts():
        upstream_count = sum(1 for i in article.cross_impacts if i["relationship_type"] == "upstream_demand_shock")
        downstream_count = sum(1 for i in article.cross_impacts if i["relationship_type"] == "downstream_supply_impact")
        
        response_data["supply_chain_summary"] = {
            "total_impacts": len(article.cross_impacts),
            "upstream_impacts": upstream_count,
            "downstream_impacts": downstream_count,
            "max_impact_score": max((i["impact_score"] for i in article.cross_impacts), default=0.0),
            "affected_sectors": list(set(i["target_sector"] for i in article.cross_impacts)),
            "analysis_method": "llm"
        }
    
    # Add sentiment key factors if available
    if sentiment_dict:
        sentiment_breakdown = sentiment_dict.get("sentiment_breakdown", {})
        key_factors = sentiment_breakdown.get("key_factors", [])
        if key_factors:
            response_data["sentiment_key_factors"] = key_factors
    
    return response_data

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

@router.get("/article/{article_id}/entities", tags=["Entities"])
async def get_article_entities(article_id: str):
    """
    Get detailed entity extraction data for an article.
    Returns companies with tickers, confidence scores, and extraction reasoning.
    """
    article = storage.get_by_id(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    if not article.entities_rich:
        raise HTTPException(status_code=404, detail="No entity data available")
    
    return {
        "article_id": article.id,
        "title": article.title,
        "entities_rich": article.entities_rich,
        "company_tickers": article.get_company_tickers(),
        "extraction_method": "llm"
    }

@router.get("/article/{article_id}/stock_impacts", tags=["Stock Impact"])
async def get_article_stock_impacts(article_id: str):
    """
    Get detailed stock impact analysis for an article.
    Returns LLM reasoning for each impacted stock.
    """
    article = storage.get_by_id(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    if not article.impacted_stocks:
        raise HTTPException(status_code=404, detail="No stock impact data available")
    
    return {
        "article_id": article.id,
        "title": article.title,
        "impacted_stocks": article.impacted_stocks,
        "impact_summary": {
            "total_stocks": len(article.impacted_stocks),
            "impact_breakdown": {
                "direct": sum(1 for s in article.impacted_stocks if s["impact_type"] == "direct"),
                "sector": sum(1 for s in article.impacted_stocks if s["impact_type"] == "sector"),
                "regulatory": sum(1 for s in article.impacted_stocks if s["impact_type"] == "regulatory")
            },
            "avg_confidence": round(
                sum(s["confidence"] for s in article.impacted_stocks) / len(article.impacted_stocks), 2
            ) if article.impacted_stocks else 0.0
        },
        "analysis_method": "llm"
    }

@router.get("/article/{article_id}/supply_chain", tags=["Supply Chain"])
async def get_article_supply_chain_impacts(article_id: str):
    """
    Get detailed supply chain impact analysis for an article.
    Returns LLM reasoning for upstream and downstream effects.
    """
    article = storage.get_by_id(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    
    if not article.has_cross_impacts():
        raise HTTPException(status_code=404, detail="No supply chain impact data available")
    
    upstream_impacts = [i for i in article.cross_impacts if i["relationship_type"] == "upstream_demand_shock"]
    downstream_impacts = [i for i in article.cross_impacts if i["relationship_type"] == "downstream_supply_impact"]
    
    return {
        "article_id": article.id,
        "title": article.title,
        "upstream_impacts": upstream_impacts,
        "downstream_impacts": downstream_impacts,
        "impact_summary": {
            "total_impacts": len(article.cross_impacts),
            "upstream_count": len(upstream_impacts),
            "downstream_count": len(downstream_impacts),
            "max_impact_score": max((i["impact_score"] for i in article.cross_impacts), default=0.0),
            "affected_sectors": list(set(i["target_sector"] for i in article.cross_impacts)),
            "avg_dependency_weight": round(
                sum(i["dependency_weight"] for i in article.cross_impacts) / len(article.cross_impacts), 2
            ) if article.cross_impacts else 0.0
        },
        "analysis_method": "llm"
    }