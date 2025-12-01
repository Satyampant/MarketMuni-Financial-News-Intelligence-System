from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from langgraph_orchestration import NewsIntelligenceGraph
from news_storage import NewsArticle, NewsStorage
from vector_store import VectorStore
from deduplication import DeduplicationAgent
from entity_extraction import EntityExtractor
from stock_impact import StockImpactMapper


# Pydantic models for API
class ArticleInput(BaseModel):
    """Input model for article ingestion"""
    id: str = Field(..., description="Unique article ID")
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content")
    source: str = Field(..., description="Source name")
    timestamp: str = Field(..., description="ISO format timestamp")
    raw_text: Optional[str] = Field(None, description="Raw article text")


class ArticleOutput(BaseModel):
    """Output model for article with enriched data"""
    id: str
    title: str
    content: str
    source: str
    timestamp: str
    entities: Optional[Dict[str, List[str]]] = None
    impacted_stocks: Optional[List[Dict[str, Any]]] = None
    relevance_score: Optional[float] = None


class IngestResponse(BaseModel):
    """Response model for article ingestion"""
    success: bool
    article_id: str
    message: str
    is_duplicate: bool
    duplicates_found: int
    entities_extracted: Dict[str, int]
    stocks_impacted: int
    stats: Dict[str, Any]


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    query: str
    results_count: int
    articles: List[ArticleOutput]
    stats: Dict[str, Any]


class StatsResponse(BaseModel):
    """Response model for system statistics"""
    total_articles_stored: int
    vector_store_count: int
    dedup_threshold: Dict[str, float]
    status: str


# Initialize FastAPI app
app = FastAPI(
    title="Financial News Intelligence API",
    description="Multi-agent AI system for processing and querying financial news",
    version="1.0.0"
)

# Initialize components (singleton pattern)
storage = NewsStorage()
vector_store = VectorStore(
    collection_name="financial_news",
    persist_directory="./chroma_db"
)
dedup_agent = DeduplicationAgent()
entity_extractor = EntityExtractor()
stock_mapper = StockImpactMapper()

# Initialize LangGraph orchestration
news_graph = NewsIntelligenceGraph(
    storage=storage,
    vector_store=vector_store,
    dedup_agent=dedup_agent,
    entity_extractor=entity_extractor,
    stock_mapper=stock_mapper
)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Financial News Intelligence API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ingest": "Ingest a new article",
            "GET /query": "Query articles by text",
            "GET /stats": "Get system statistics",
            "GET /health": "Health check"
        }
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "storage": storage.article_count(),
            "vector_store": vector_store.count()
        }
    }


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_article(article_input: ArticleInput):
    """
    Ingest a new financial news article through the multi-agent pipeline.
    
    Pipeline Flow:
    1. Ingestion Agent: Validates article
    2. Deduplication Agent: Checks for duplicates
    3. Entity Extraction Agent: Extracts companies, sectors, regulators
    4. Impact Mapper Agent: Maps to affected stocks
    5. Indexing Agent: Stores in database and vector store
    
    Args:
        article_input: Article data to ingest
        
    Returns:
        IngestResponse with processing results and statistics
    """
    try:
        # Convert input to NewsArticle
        article = NewsArticle(
            id=article_input.id,
            title=article_input.title,
            content=article_input.content,
            source=article_input.source,
            timestamp=article_input.timestamp,
            raw_text=article_input.raw_text or article_input.content
        )
        
        # Run through LangGraph pipeline
        result = news_graph.run_pipeline(article)
        
        # Check for errors
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Extract stats
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
            stats={
                "ingestion_time": stats.get("ingestion_time"),
                "impact_breakdown": impact_breakdown,
                "total_articles": stats.get("total_articles"),
                "indexed": stats.get("indexed", False)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/query", response_model=QueryResponse, tags=["Query"])
async def query_articles(
    q: str = Query(..., description="Natural language query text"),
    top_k: int = Query(10, ge=1, le=50, description="Maximum number of results")
):
    """
    Query articles using natural language with context-aware retrieval.
    
    Query Behavior:
    - Company queries: Returns direct mentions + sector-wide news
    - Sector queries: Returns all related news across companies
    - Regulator queries: Returns regulator-specific articles
    - Thematic queries: Uses semantic matching
    
    Examples:
    - "HDFC Bank news" → Direct mentions + Banking sector news
    - "Banking sector update" → All banking-related articles
    - "RBI policy changes" → RBI-specific regulatory news
    - "Interest rate impact" → Semantic theme matching
    
    Args:
        q: Natural language query
        top_k: Maximum number of results to return
        
    Returns:
        QueryResponse with matching articles and query statistics
    """
    try:
        if not q or len(q.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Run query through LangGraph
        result = news_graph.run_query(q)
        
        # Check for errors
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Extract results
        articles = result.get("query_results", [])[:top_k]
        stats = result.get("stats", {})
        
        # Convert to output format
        article_outputs = []
        for article in articles:
            article_outputs.append(ArticleOutput(
                id=article.id,
                title=article.title,
                content=article.content,
                source=article.source,
                timestamp=article.timestamp.isoformat(),
                entities=getattr(article, "entities", None),
                impacted_stocks=getattr(article, "impacted_stocks", None),
                relevance_score=getattr(article, "relevance_score", None)
            ))
        
        return QueryResponse(
            query=q,
            results_count=len(article_outputs),
            articles=article_outputs,
            stats=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_stats():
    """
    Get system statistics including article counts and deduplication metrics.
    
    Returns:
        StatsResponse with system statistics
    """
    try:
        stats = news_graph.get_stats()
        
        return StatsResponse(
            total_articles_stored=stats["total_articles_stored"],
            vector_store_count=stats["vector_store_count"],
            dedup_threshold=stats["dedup_threshold"],
            status=stats["status"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@app.get("/article/{article_id}", tags=["Retrieval"])
async def get_article(article_id: str):
    """
    Retrieve a specific article by ID.
    
    Args:
        article_id: Unique article identifier
        
    Returns:
        Article data with entities and stock impacts
    """
    try:
        article = storage.get_by_id(article_id)
        
        if not article:
            raise HTTPException(status_code=404, detail=f"Article {article_id} not found")
        
        return ArticleOutput(
            id=article.id,
            title=article.title,
            content=article.content,
            source=article.source,
            timestamp=article.timestamp.isoformat(),
            entities=getattr(article, "entities", None),
            impacted_stocks=getattr(article, "impacted_stocks", None)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")


@app.delete("/reset", tags=["Admin"])
async def reset_system():
    """
    Reset the entire system (for testing purposes).
    
    Warning: This deletes all articles and vector store data.
    """
    try:
        # Reset storage
        global storage, vector_store
        storage = NewsStorage()
        
        # Reset vector store
        vector_store.reset()
        
        return {
            "success": True,
            "message": "System reset successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)