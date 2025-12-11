"""
Two-Step Query Processor with Broad Filter Optimization
Implements MongoDB-first filtering with vector search fallback strategy.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.core.models import NewsArticle, QueryRouting
from app.core.llm_schemas import QueryIntent
from app.services.mongodb_store import MongoDBStore
from app.services.vector_store import VectorStore
from app.agents.query_router import QueryRouter
from app.core.config_loader import get_config


class TwoStepQueryProcessor:
    """
    Two-step query processing with adaptive strategy selection:
    
    STRATEGY A (Small result set <= max_filter_ids):
    MongoDB Filter → Vector Search
    - Filter articles in MongoDB first
    - Then perform vector search on filtered IDs
    
    STRATEGY B (Large result set > max_filter_ids):
    Vector Search → MongoDB Validation (Inverted)
    - Perform unrestricted vector search first
    - Then validate results against MongoDB filters
    """
    
    def __init__(
        self,
        mongodb_store: MongoDBStore,
        vector_store: VectorStore,
        query_router: QueryRouter,
        config: Optional[Any] = None
    ):
        self.mongodb_store = mongodb_store
        self.vector_store = vector_store
        self.query_router = query_router
        self.config = config or get_config()
        
        # Threshold for strategy selection
        self.max_filter_ids = self.config.mongodb.max_filter_ids
        
        print(f"✓ TwoStepQueryProcessor initialized")
        print(f"  - Strategy threshold: {self.max_filter_ids} IDs")
        print(f"  - MongoDB filter optimization enabled")
    
    def process_query(
        self,
        query: str,
        top_k: int = 10,
        sentiment_filter: Optional[str] = None
    ) -> Tuple[List[NewsArticle], QueryRouting]:
        """Implements adaptive strategy selection based on filter result size."""
        
        # Step 1: Route query using LLM
        routing = self.query_router.route_query(query)
        
        # Step 2: Override sentiment filter if provided
        if sentiment_filter:
            routing.sentiment_filter = sentiment_filter
        
        # Step 3: Generate MongoDB filter
        mongodb_filter = self.query_router.generate_mongodb_filter(routing)
        
        # This ensures the filter is applied regardless of the strategy chosen by the LLM
        if sentiment_filter:
            mongodb_filter["sentiment.classification"] = sentiment_filter

        # Step 4: Count potential matches
        filtered_count = self.mongodb_store.count_articles(mongodb_filter)
        
        if filtered_count == 0 and sentiment_filter:
                    mongodb_filter = {"sentiment.classification": sentiment_filter}
                    filtered_count = self.mongodb_store.count_articles(mongodb_filter)

        # Broad Filter Optimization
        if filtered_count == 0:
            routing.strategy_metadata = {
                "strategy_used": "no_results",
                "filtered_count": 0,
                "vector_candidates": 0,
                "mongodb_filter_applied": bool(mongodb_filter),
                "threshold": self.max_filter_ids
            }
            return [], routing
        
        elif filtered_count <= self.max_filter_ids:
            # STRATEGY A: MongoDB Filter → Vector Search
            strategy_used = "mongo_filter_first"
            
            # Get filtered article IDs from MongoDB
            filtered_ids = self.mongodb_store.filter_by_metadata(
                mongodb_filter,
                limit=None  # Get all matching IDs
            )
            
            # Generate query embedding
            query_embedding = self.vector_store.create_embedding(routing.refined_query)
            
            # Perform vector search on filtered IDs only
            vector_results = self.vector_store.search_by_ids(
                query_embedding,
                filtered_ids,
                top_k * 2  # Get extra for reranking
            )
        
        else:
            # STRATEGY B: Vector Search → MongoDB Validation (Inverted)
            strategy_used = "vector_search_first"
            
            # Generate query embedding
            query_embedding = self.vector_store.create_embedding(routing.refined_query)
            
            # Perform unrestricted vector search
            vector_results = self.vector_store.search(
                query="",
                query_embedding=query_embedding,
                top_k=top_k * 5  # Get more candidates for filtering
            )
            
            # Filter results by validating against MongoDB
            candidate_ids = [r["article_id"] for r in vector_results]
            
            # Add ID filter to existing MongoDB filter
            mongodb_filter["id"] = {"$in": candidate_ids}
            
            # Get valid IDs from MongoDB
            valid_ids = set(self.mongodb_store.filter_by_metadata(mongodb_filter))
            
            # Keep only valid results
            vector_results = [
                r for r in vector_results 
                if r["article_id"] in valid_ids
            ][:top_k * 2]  # Limit to reasonable size for reranking
        
        # Step 6: Fetch full articles from MongoDB
        article_ids = [r["article_id"] for r in vector_results]
        full_articles = self.mongodb_store.get_articles_by_ids(article_ids)
        
        # Step 7: Attach relevance scores to articles
        self._attach_scores(full_articles, vector_results)
        
        # Step 8: Rerank articles
        reranked_articles = self._rerank_articles(full_articles, routing)
        
        # Step 9: Return top_k articles + routing metadata
        final_articles = reranked_articles[:top_k]
        
        # Attach strategy metadata to routing for debugging
        routing.strategy_metadata = {
            "strategy_used": strategy_used,
            "filtered_count": filtered_count,
            "vector_candidates": len(vector_results),
            "mongodb_filter_applied": bool(mongodb_filter),
            "threshold": self.max_filter_ids
        }
        
        return final_articles, routing
    
    def _attach_scores(
        self,
        articles: List[NewsArticle],
        vector_results: List[Dict[str, Any]]
    ) -> None:
        """
        Attach relevance scores from vector search to articles.
        Modifies articles in-place.
        """
        # Create ID→similarity map
        score_map = {
            r["article_id"]: r["similarity"] 
            for r in vector_results
        }
        
        # Attach scores to articles
        for article in articles:
            article.relevance_score = score_map.get(article.id, 0.0)
    
    def _rerank_articles(
        self,
        articles: List[NewsArticle],
        routing: QueryRouting
    ) -> List[NewsArticle]:
        """
        Rerank articles combining semantic similarity with strategy scoring.
        """
        for article in articles:
            # Get base semantic score
            semantic_score = getattr(article, 'relevance_score', 0.0)
            
            # Calculate strategy-specific score
            strategy_score = self._calculate_strategy_score(article, routing)
            
            # Weighted combination (50% semantic, 50% strategy)
            base_score = (semantic_score * 0.5) + (strategy_score * 0.5)
            
            # Apply sentiment boost if applicable
            if article.has_sentiment():
                sentiment_data = article.get_sentiment()
                signal_strength = sentiment_data.signal_strength
                base_score = self._apply_sentiment_boost(base_score, signal_strength)
            
            # Store final score
            article.final_score = min(base_score, 1.0)
            article.strategy_score = strategy_score
        
        # Sort by final score
        return sorted(articles, key=lambda x: getattr(x, 'final_score', 0.0), reverse=True)
    
    def _calculate_strategy_score(
        self,
        article: NewsArticle,
        routing: QueryRouting
    ) -> float:
        """
        Calculate strategy-specific relevance score based on metadata match.
        """
        strategy_score = 0.0
        
        if routing.strategy == QueryIntent.DIRECT_ENTITY:
            # Check for entity matches
            article_companies = article.entities.get("Companies", []) if article.entities else []
            company_match = any(
                entity.lower() in [c.lower() for c in article_companies]
                for entity in routing.entities
            )
            
            # Check for stock symbol matches
            article_stocks = [
                s.get("symbol", "").upper() 
                for s in article.impacted_stocks
            ]
            stock_match = any(
                symbol.upper() in article_stocks
                for symbol in routing.stock_symbols
            )
            
            strategy_score = 1.0 if (company_match or stock_match) else 0.0
        
        elif routing.strategy == QueryIntent.SECTOR_WIDE:
            # Check for sector matches
            article_sectors = article.entities.get("Sectors", []) if article.entities else []
            sector_match = any(
                sector.lower() in [s.lower() for s in article_sectors]
                for sector in routing.sectors
            )
            strategy_score = 0.8 if sector_match else 0.0
        
        elif routing.strategy == QueryIntent.REGULATORY:
            # Check for regulator matches
            article_regulators = article.entities.get("Regulators", []) if article.entities else []
            regulator_match = any(
                regulator.lower() in [r.lower() for r in article_regulators]
                for regulator in routing.regulators
            )
            strategy_score = 1.0 if regulator_match else 0.0
        
        elif routing.strategy == QueryIntent.SENTIMENT_DRIVEN:
            # Check sentiment match
            if article.has_sentiment():
                sentiment_data = article.get_sentiment()
                sentiment_match = (
                    sentiment_data.classification == routing.sentiment_filter
                )
                strategy_score = 0.7 if sentiment_match else 0.0
                
                # Boost if sector also matches
                if sentiment_match and routing.sectors:
                    article_sectors = article.entities.get("Sectors", []) if article.entities else []
                    sector_match = any(
                        sector.lower() in [s.lower() for s in article_sectors]
                        for sector in routing.sectors
                    )
                    if sector_match:
                        strategy_score = 0.9
        
        elif routing.strategy == QueryIntent.CROSS_IMPACT:
            # Check for cross-impact data
            has_cross_impacts = article.has_cross_impacts() if hasattr(article, 'has_cross_impacts') else False
            strategy_score = 0.8 if has_cross_impacts else 0.5
        
        else:
            # Default score for other strategies
            strategy_score = 0.3
        
        return strategy_score
    
    def _apply_sentiment_boost(
        self,
        score: float,
        sentiment_signal: float
    ) -> float:
        """
        Amplify score based on sentiment signal strength.
        Formula: 1.0 + (signal_strength / 200.0). Max boost ~1.5x.
        """
        sentiment_boost = 1.0 + (sentiment_signal / 200.0)
        return score * sentiment_boost
    
    def process_query_with_routing(
        self,
        query: str,
        top_k: int = 10,
        sentiment_filter: Optional[str] = None
    ) -> Tuple[List[NewsArticle], QueryRouting]:
        """
        Alias for process_query to maintain compatibility with existing code.
        """
        return self.process_query(query, top_k, sentiment_filter)