"""
LLM Query Processor - Strategy Executor with Metadata Filtering
Executes query strategies based on LLM routing decisions.
File: app/agents/llm_query_processor.py
"""

from typing import List, Dict, Any, Optional
import json

from app.core.models import NewsArticle
from app.core.llm_schemas import QueryRouterSchema, QueryIntent
from app.services.vector_store import VectorStore
from app.agents.llm_query_router import LLMQueryRouter
from app.core.config_loader import get_config


class LLMQueryProcessor:
    """
    Strategy-based query processor using LLM routing decisions.
    Executes ChromaDB metadata filtering based on extracted entities.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        query_router: LLMQueryRouter,
        config: Optional[Any] = None
    ):
        """
        Initialize LLM query processor.
        
        Args:
            vector_store: ChromaDB vector store instance
            query_router: LLM query router for strategy selection
            config: Optional configuration override
        """
        self.vector_store = vector_store
        self.query_router = query_router
        self.config = config or get_config()
        
        print(f"✓ LLMQueryProcessor initialized")
        print(f"  - Strategy-based execution with metadata filtering")
    
    def _execute_direct_entity_strategy(
        self,
        routing: QueryRouterSchema,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Execute direct entity search using stock symbol filtering.
        Uses metadata filter: where={"impacted_stocks": {"$contains": "HDFCBANK"}}
        
        Args:
            routing: Query routing decision with extracted entities
            top_k: Number of results to retrieve
            
        Returns:
            List of matching articles
        """
        results = []
        
        # Search by stock symbols (highest precision)
        for symbol in routing.stock_symbols:
            # ChromaDB metadata filter for impacted_stocks
            # Note: impacted_stocks is stored as JSON string, so we search in the raw metadata
            where_filter = {
                "$or": [
                    {"impacted_stocks": {"$contains": f'"{symbol}"'}},
                    {"impacted_stocks": {"$contains": f"'{symbol}'"}}
                ]
            }
            
            symbol_results = self.vector_store.search(
                query=routing.refined_query,
                top_k=top_k,
                where=where_filter
            )
            results.extend(symbol_results)
        
        # Search by company names if no symbols or insufficient results
        if len(results) < top_k and routing.entities:
            for company in routing.entities:
                # Search in entities.Companies field (JSON)
                where_filter = {
                    "entities": {"$contains": f'"{company}"'}
                }
                
                company_results = self.vector_store.search(
                    query=routing.refined_query,
                    top_k=top_k,
                    where=where_filter
                )
                results.extend(company_results)
        
        # Deduplicate by article_id
        seen_ids = set()
        unique_results = []
        for result in results:
            article_id = result["article_id"]
            if article_id not in seen_ids:
                seen_ids.add(article_id)
                unique_results.append(result)
        
        return unique_results[:top_k]
    
    def _execute_sector_wide_strategy(
        self,
        routing: QueryRouterSchema,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Execute sector-wide search using sector metadata filtering.
        Uses metadata filter: where={"entities": {"$contains": "Banking"}}
        
        Args:
            routing: Query routing decision with extracted sectors
            top_k: Number of results to retrieve
            
        Returns:
            List of matching articles
        """
        results = []
        
        for sector in routing.sectors:
            # ChromaDB metadata filter for entities.Sectors (stored as JSON)
            where_filter = {
                "entities": {"$contains": f'"{sector}"'}
            }
            
            sector_results = self.vector_store.search(
                query=routing.refined_query,
                top_k=top_k,
                where=where_filter
            )
            results.extend(sector_results)
        
        # Deduplicate
        seen_ids = set()
        unique_results = []
        for result in results:
            article_id = result["article_id"]
            if article_id not in seen_ids:
                seen_ids.add(article_id)
                unique_results.append(result)
        
        return unique_results[:top_k]
    
    def _execute_regulatory_strategy(
        self,
        routing: QueryRouterSchema,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Execute regulatory search using regulator metadata filtering.
        Uses metadata filter: where={"entities": {"$contains": "RBI"}}
        
        Args:
            routing: Query routing decision with extracted regulators
            top_k: Number of results to retrieve
            
        Returns:
            List of matching articles
        """
        results = []
        
        for regulator in routing.regulators:
            # ChromaDB metadata filter for entities.Regulators (stored as JSON)
            where_filter = {
                "entities": {"$contains": f'"{regulator}"'}
            }
            
            regulator_results = self.vector_store.search(
                query=routing.refined_query,
                top_k=top_k,
                where=where_filter
            )
            results.extend(regulator_results)
        
        # Deduplicate
        seen_ids = set()
        unique_results = []
        for result in results:
            article_id = result["article_id"]
            if article_id not in seen_ids:
                seen_ids.add(article_id)
                unique_results.append(result)
        
        return unique_results[:top_k]
    
    def _execute_sentiment_strategy(
        self,
        routing: QueryRouterSchema,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Execute sentiment-driven search with sentiment and sector filtering.
        Uses metadata filter: where={"sentiment_classification": "Bullish", "entities": {"$contains": "Tech"}}
        
        Args:
            routing: Query routing decision with sentiment filter and sectors
            top_k: Number of results to retrieve
            
        Returns:
            List of matching articles
        """
        if not routing.sentiment_filter:
            # Fallback to semantic search if no sentiment specified
            return self._execute_semantic_strategy(routing, top_k)
        
        results = []
        
        if routing.sectors:
            # Combined sentiment + sector filter
            for sector in routing.sectors:
                where_filter = {
                    "$and": [
                        {"sentiment_classification": routing.sentiment_filter},
                        {"entities": {"$contains": f'"{sector}"'}}
                    ]
                }
                
                sentiment_results = self.vector_store.search(
                    query=routing.refined_query,
                    top_k=top_k,
                    where=where_filter
                )
                results.extend(sentiment_results)
        else:
            # Sentiment-only filter
            where_filter = {
                "sentiment_classification": routing.sentiment_filter
            }
            
            sentiment_results = self.vector_store.search(
                query=routing.refined_query,
                top_k=top_k,
                where=where_filter
            )
            results.extend(sentiment_results)
        
        # Deduplicate
        seen_ids = set()
        unique_results = []
        for result in results:
            article_id = result["article_id"]
            if article_id not in seen_ids:
                seen_ids.add(article_id)
                unique_results.append(result)
        
        return unique_results[:top_k]
    
    def _execute_semantic_strategy(
        self,
        routing: QueryRouterSchema,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Execute pure semantic search without metadata filters.
        Uses routing.refined_query for vector similarity search.
        
        Args:
            routing: Query routing decision with refined query
            top_k: Number of results to retrieve
            
        Returns:
            List of matching articles
        """
        results = self.vector_store.search(
            query=routing.refined_query,
            top_k=top_k,
            where=None  # No metadata filtering
        )
        
        return results
    
    def _execute_cross_impact_strategy(
        self,
        routing: QueryRouterSchema,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Execute cross-impact search for supply chain relationships.
        Searches for articles mentioning multiple sectors.
        
        Args:
            routing: Query routing decision with multiple sectors
            top_k: Number of results to retrieve
            
        Returns:
            List of matching articles
        """
        if len(routing.sectors) < 2:
            # Fallback to sector-wide if insufficient sectors
            return self._execute_sector_wide_strategy(routing, top_k)
        
        # Search for articles mentioning ANY of the sectors
        # (cross-impacts are typically identified post-ingestion)
        results = []
        
        for sector in routing.sectors:
            where_filter = {
                "entities": {"$contains": f'"{sector}"'}
            }
            
            sector_results = self.vector_store.search(
                query=routing.refined_query,
                top_k=top_k,
                where=where_filter
            )
            results.extend(sector_results)
        
        # Deduplicate and prioritize articles with cross_impacts
        seen_ids = set()
        unique_results = []
        for result in results:
            article_id = result["article_id"]
            if article_id not in seen_ids:
                seen_ids.add(article_id)
                # Boost articles with cross-impact data
                if result["metadata"].get("has_cross_impacts", False):
                    result["cross_impact_boost"] = 1.2
                unique_results.append(result)
        
        return unique_results[:top_k]
    
    def _execute_temporal_strategy(
        self,
        routing: QueryRouterSchema,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Execute temporal search with time-based filtering.
        Falls back to entity search with temporal query refinement.
        
        Args:
            routing: Query routing decision with temporal scope
            top_k: Number of results to retrieve
            
        Returns:
            List of matching articles
        """
        # For now, use entity-based search with temporal query
        # (ChromaDB timestamp filtering would require date parsing)
        if routing.entities or routing.stock_symbols:
            return self._execute_direct_entity_strategy(routing, top_k)
        else:
            return self._execute_semantic_strategy(routing, top_k)
    
    def _execute_strategy(
        self,
        routing: QueryRouterSchema,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Dispatch to appropriate strategy executor based on routing.strategy.
        
        Args:
            routing: Query routing decision
            top_k: Number of results to retrieve
            
        Returns:
            List of matching articles
        """
        strategy_map = {
            QueryIntent.DIRECT_ENTITY: self._execute_direct_entity_strategy,
            QueryIntent.SECTOR_WIDE: self._execute_sector_wide_strategy,
            QueryIntent.REGULATORY: self._execute_regulatory_strategy,
            QueryIntent.SENTIMENT_DRIVEN: self._execute_sentiment_strategy,
            QueryIntent.CROSS_IMPACT: self._execute_cross_impact_strategy,
            QueryIntent.TEMPORAL: self._execute_temporal_strategy,
            QueryIntent.SEMANTIC_SEARCH: self._execute_semantic_strategy
        }
        
        executor = strategy_map.get(routing.strategy)
        
        if not executor:
            print(f"⚠ Unknown strategy: {routing.strategy}, falling back to semantic search")
            return self._execute_semantic_strategy(routing, top_k)
        
        return executor(routing, top_k)
    
    def _calculate_strategy_score(
        self,
        result: Dict[str, Any],
        routing: QueryRouterSchema
    ) -> float:
        """
        Calculate strategy match score based on metadata alignment.
        
        Args:
            result: Search result with metadata
            routing: Query routing decision
            
        Returns:
            Strategy match score (0.0-1.0)
        """
        metadata = result["metadata"]
        entities = metadata.get("entities", {})
        
        # Parse entities JSON if string
        if isinstance(entities, str):
            try:
                entities = json.loads(entities)
            except json.JSONDecodeError:
                entities = {}
        
        # Parse impacted_stocks JSON if string
        impacted_stocks = metadata.get("impacted_stocks", [])
        if isinstance(impacted_stocks, str):
            try:
                impacted_stocks = json.loads(impacted_stocks)
            except json.JSONDecodeError:
                impacted_stocks = []
        
        strategy_score = 0.0
        
        if routing.strategy == QueryIntent.DIRECT_ENTITY:
            # Check company/stock match
            article_companies = entities.get("Companies", [])
            company_match = any(c in article_companies for c in routing.entities)
            
            stock_symbols = [s.get("symbol", "") for s in impacted_stocks]
            stock_match = any(sym in stock_symbols for sym in routing.stock_symbols)
            
            if company_match or stock_match:
                strategy_score = 1.0
            
        elif routing.strategy == QueryIntent.SECTOR_WIDE:
            # Check sector match
            article_sectors = entities.get("Sectors", [])
            sector_match = any(s in article_sectors for s in routing.sectors)
            
            if sector_match:
                strategy_score = 0.8
        
        elif routing.strategy == QueryIntent.REGULATORY:
            # Check regulator match
            article_regulators = entities.get("Regulators", [])
            regulator_match = any(r in article_regulators for r in routing.regulators)
            
            if regulator_match:
                strategy_score = 1.0
        
        elif routing.strategy == QueryIntent.SENTIMENT_DRIVEN:
            # Check sentiment + sector match
            sentiment_match = (
                metadata.get("sentiment_classification") == routing.sentiment_filter
            )
            
            if sentiment_match:
                strategy_score = 0.7
                
                # Boost if sector also matches
                article_sectors = entities.get("Sectors", [])
                sector_match = any(s in article_sectors for s in routing.sectors)
                if sector_match:
                    strategy_score = 0.9
        
        elif routing.strategy == QueryIntent.CROSS_IMPACT:
            # Boost if article has cross-impacts
            if metadata.get("has_cross_impacts", False):
                strategy_score = 0.8
            else:
                strategy_score = 0.5
        
        else:
            # Semantic search relies purely on similarity
            strategy_score = 0.3
        
        return strategy_score
    
    def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        routing: QueryRouterSchema
    ) -> List[Dict[str, Any]]:
        """
        Rerank results by combining semantic similarity, strategy match, and sentiment signal.
        
        Args:
            results: Raw search results
            routing: Query routing decision
            
        Returns:
            Reranked results with final_score
        """
        for result in results:
            semantic_score = result.get("similarity", 0.0)
            strategy_score = self._calculate_strategy_score(result, routing)
            
            # Base score: 50% semantic + 50% strategy
            final_score = (semantic_score * 0.5) + (strategy_score * 0.5)
            
            # Apply sentiment signal boost
            metadata = result["metadata"]
            signal_strength = float(metadata.get("sentiment_signal_strength", 0.0))
            sentiment_boost = 1.0 + (signal_strength / 200.0)  # Max 1.5x boost
            
            final_score *= sentiment_boost
            
            # Apply cross-impact boost if present
            cross_impact_boost = result.get("cross_impact_boost", 1.0)
            final_score *= cross_impact_boost
            
            # Cap at 1.0
            result["final_score"] = min(final_score, 1.0)
            result["strategy_score"] = strategy_score
            result["sentiment_boost"] = sentiment_boost
        
        # Sort by final score (descending)
        return sorted(results, key=lambda x: x["final_score"], reverse=True)
    
    def _results_to_articles(
        self,
        results: List[Dict[str, Any]]
    ) -> List[NewsArticle]:
        """
        Convert search results to NewsArticle objects with parsed metadata.
        
        Args:
            results: Reranked search results
            
        Returns:
            List of NewsArticle objects
        """
        articles = []
        
        for result in results:
            metadata = result["metadata"]
            
            # Parse JSON fields
            entities = metadata.get("entities", {})
            if isinstance(entities, str):
                try:
                    entities = json.loads(entities)
                except json.JSONDecodeError:
                    entities = {}
            
            impacted_stocks = metadata.get("impacted_stocks", [])
            if isinstance(impacted_stocks, str):
                try:
                    impacted_stocks = json.loads(impacted_stocks)
                except json.JSONDecodeError:
                    impacted_stocks = []
            
            sentiment = metadata.get("sentiment")
            if isinstance(sentiment, str):
                try:
                    sentiment = json.loads(sentiment) if sentiment != "null" else None
                except json.JSONDecodeError:
                    sentiment = None
            
            cross_impacts = metadata.get("cross_impacts", [])
            if isinstance(cross_impacts, str):
                try:
                    cross_impacts = json.loads(cross_impacts)
                except json.JSONDecodeError:
                    cross_impacts = []
            
            # Create NewsArticle
            article = NewsArticle(
                id=metadata["article_id"],
                title=metadata["title"],
                content=result["document"],
                source=metadata["source"],
                timestamp=metadata["timestamp"],
                raw_text=result["document"]
            )
            
            # Attach metadata
            article.entities = entities
            article.impacted_stocks = impacted_stocks
            article.sentiment = sentiment
            article.cross_impacts = cross_impacts
            
            # Attach query-specific scores
            article.relevance_score = result.get("final_score", result["similarity"])
            article.strategy_score = result.get("strategy_score", 0.0)
            article.sentiment_boost = result.get("sentiment_boost", 1.0)
            
            articles.append(article)
        
        return articles
    
    def process_query(
        self,
        query: str,
        top_k: int = 10,
        sentiment_filter: Optional[str] = None
    ) -> List[NewsArticle]:
        """
        Main entry point: route query → execute strategy → rerank → return results.
        
        Args:
            query: User's natural language query
            top_k: Number of results to return
            sentiment_filter: Optional sentiment filter override
            
        Returns:
            List of NewsArticle objects ranked by relevance
        """
        # Step 1: Route query using LLM
        routing = self.query_router.route_query(query)
        
        # Override sentiment filter if provided
        if sentiment_filter:
            routing.sentiment_filter = sentiment_filter
        
        # Step 2: Execute strategy-specific search
        results = self._execute_strategy(routing, top_k * 2)  # Retrieve extra for reranking
        
        # Step 3: Rerank results
        ranked_results = self._rerank_results(results, routing)
        
        # Step 4: Apply minimum similarity threshold
        min_similarity = self.config.query_processing.min_similarity
        filtered_results = [
            r for r in ranked_results 
            if r.get("final_score", 0.0) >= min_similarity
        ]
        
        # Step 5: Convert to NewsArticle objects
        articles = self._results_to_articles(filtered_results[:top_k])
        
        return articles