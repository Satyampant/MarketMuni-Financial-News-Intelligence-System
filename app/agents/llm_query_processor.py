from typing import List, Dict, Any, Optional, Tuple
import json

from app.core.models import NewsArticle
from app.core.llm_schemas import QueryRouterSchema, QueryIntent
from app.services.vector_store import VectorStore
from app.agents.llm_query_router import LLMQueryRouter
from app.core.config_loader import get_config


class LLMQueryProcessor:
    """
    Strategy-based query processor using LLM routing decisions and ChromaDB metadata filtering.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        query_router: LLMQueryRouter,
        config: Optional[Any] = None
    ):
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
        """Executes search prioritized by stock symbols, falling back to company names."""
        results = []
        
        # Priority 1: Search by stock symbols
        for symbol in routing.stock_symbols:
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
        
        # Priority 2: Search by company names if results are insufficient
        if len(results) < top_k and routing.entities:
            for company in routing.entities:
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
        """Executes search filtered by sector metadata."""
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
        """Executes search filtered by regulator entities (e.g., RBI, SEBI)."""
        results = []
        
        for regulator in routing.regulators:
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
        """Executes search based on sentiment classification, optionally combined with sectors."""
        if not routing.sentiment_filter:
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
        """Standard semantic search using the refined query."""
        results = self.vector_store.search(
            query=routing.refined_query,
            top_k=top_k,
            where=None
        )
        
        return results
    
    def _execute_cross_impact_strategy(
        self,
        routing: QueryRouterSchema,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Executes search for multi-sector relationships, boosting articles with cross-impact data."""
        if len(routing.sectors) < 2:
            return self._execute_sector_wide_strategy(routing, top_k)
        
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
        
        # Deduplicate and boost
        seen_ids = set()
        unique_results = []
        for result in results:
            article_id = result["article_id"]
            if article_id not in seen_ids:
                seen_ids.add(article_id)
                # Boost logic: 1.2x multiplier for cross-impact data
                if result["metadata"].get("has_cross_impacts", False):
                    result["cross_impact_boost"] = 1.2
                unique_results.append(result)
        
        return unique_results[:top_k]
    
    def _execute_temporal_strategy(
        self,
        routing: QueryRouterSchema,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Falls back to entity search with temporal query refinement if entities exist."""
        if routing.entities or routing.stock_symbols:
            return self._execute_direct_entity_strategy(routing, top_k)
        else:
            return self._execute_semantic_strategy(routing, top_k)
    
    def _execute_strategy(
        self,
        routing: QueryRouterSchema,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Dispatches execution to the appropriate strategy method."""
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
    
    def _merge_results(
        self,
        primary_results: List[Dict[str, Any]],
        context_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merges results, keeping the highest similarity score per article ID."""
        merged = {}
        all_results = primary_results + context_results
        
        for result in all_results:
            article_id = result["article_id"]
            current_similarity = result.get("similarity", 0.0)
            
            if article_id not in merged:
                merged[article_id] = result
            else:
                existing_similarity = merged[article_id].get("similarity", 0.0)
                if current_similarity > existing_similarity:
                    merged[article_id] = result
        
        return list(merged.values())
    
    def _calculate_strategy_score(
        self,
        result: Dict[str, Any],
        routing: QueryRouterSchema
    ) -> float:
        """Calculates a score (0.0 - 1.0) based on how well the article metadata matches the intent."""
        metadata = result["metadata"]
        
        # Safely parse JSON fields
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
        
        strategy_score = 0.0
        
        if routing.strategy == QueryIntent.DIRECT_ENTITY:
            article_companies = entities.get("Companies", [])
            company_match = any(
                entity.lower() in [c.lower() for c in article_companies]
                for entity in routing.entities
            )
            
            stock_symbols = [
                s.get("symbol", "").upper() if isinstance(s, dict) else str(s).upper()
                for s in impacted_stocks
            ]
            stock_match = any(
                symbol.upper() in stock_symbols
                for symbol in routing.stock_symbols
            )
            
            if company_match or stock_match:
                strategy_score = 1.0
            
        elif routing.strategy == QueryIntent.SECTOR_WIDE:
            article_sectors = entities.get("Sectors", [])
            sector_match = any(
                sector.lower() in [s.lower() for s in article_sectors]
                for sector in routing.sectors
            )
            if sector_match:
                strategy_score = 0.8
        
        elif routing.strategy == QueryIntent.REGULATORY:
            article_regulators = entities.get("Regulators", [])
            regulator_match = any(
                regulator.lower() in [r.lower() for r in article_regulators]
                for regulator in routing.regulators
            )
            if regulator_match:
                strategy_score = 1.0
        
        elif routing.strategy == QueryIntent.SENTIMENT_DRIVEN:
            sentiment_match = (
                metadata.get("sentiment_classification") == routing.sentiment_filter
            )
            
            if sentiment_match:
                strategy_score = 0.7
                # Boost if sector also matches
                article_sectors = entities.get("Sectors", [])
                sector_match = any(
                    sector.lower() in [s.lower() for s in article_sectors]
                    for sector in routing.sectors
                )
                if sector_match:
                    strategy_score = 0.9
        
        elif routing.strategy == QueryIntent.CROSS_IMPACT:
            if metadata.get("has_cross_impacts", False):
                strategy_score = 0.8
            else:
                strategy_score = 0.5
        
        else:
            strategy_score = 0.3
        
        return strategy_score
    
    def _apply_sentiment_boost(
        self,
        score: float,
        sentiment_signal: float
    ) -> float:
        """
        Amplifies score based on signal strength. 
        Formula: 1.0 + (signal_strength / 200.0). Max boost ~1.5x.
        """
        sentiment_boost = 1.0 + (sentiment_signal / 200.0)
        return score * sentiment_boost
    
    def _rerank_by_relevance(
        self,
        results: List[Dict[str, Any]],
        routing: QueryRouterSchema
    ) -> List[Dict[str, Any]]:
        """Reranks results combining semantic similarity, strategy match, and sentiment signal."""
        for result in results:
            semantic_score = result.get("similarity", 0.0)
            strategy_score = self._calculate_strategy_score(result, routing)
            
            # Weighted base score: 50% semantic, 50% strategy
            base_score = (semantic_score * 0.5) + (strategy_score * 0.5)
            
            metadata = result["metadata"]
            signal_strength = float(metadata.get("sentiment_signal_strength", 0.0))
            
            final_score = self._apply_sentiment_boost(base_score, signal_strength)
            
            # Apply Cross-Impact Boost if present
            cross_impact_boost = result.get("cross_impact_boost", 1.0)
            final_score *= cross_impact_boost
            
            result["final_score"] = min(final_score, 1.0)
            result["strategy_score"] = strategy_score
            result["sentiment_boost"] = 1.0 + (signal_strength / 200.0)
        
        return sorted(results, key=lambda x: x["final_score"], reverse=True)
    
    def _results_to_articles(
        self,
        results: List[Dict[str, Any]]
    ) -> List[NewsArticle]:
        """Converts raw ChromaDB results to NewsArticle objects with parsed metadata."""
        articles = []
        
        for result in results:
            metadata = result["metadata"]
            
            # Metadata Parsing
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
            
            article = NewsArticle(
                id=metadata["article_id"],
                title=metadata["title"],
                content=result["document"],
                source=metadata["source"],
                timestamp=metadata["timestamp"],
                raw_text=result["document"]
            )
            
            article.entities = entities
            article.impacted_stocks = impacted_stocks
            article.sentiment = sentiment
            article.cross_impacts = cross_impacts
            
            article.relevance_score = result.get("final_score", result["similarity"])
            article.strategy_score = result.get("strategy_score", 0.0)
            article.sentiment_boost = result.get("sentiment_boost", 1.0)
            
            articles.append(article)
        
        return articles
    
    def process_query_with_routing(
        self,
        query: str,
        top_k: int = 10,
        sentiment_filter: Optional[str] = None
    ) -> Tuple[List[NewsArticle], QueryRouterSchema]:
        """Main entry point: Returns both results and the routing decision."""
        
        # 1. Route query using LLM
        routing = self.query_router.route_query(query)
        
        if sentiment_filter:
            routing.sentiment_filter = sentiment_filter
        
        # 2. Execute strategy-specific search (fetching extra for reranking)
        results = self._execute_strategy(routing, top_k * 2)
        
        # 3. Rerank results
        ranked_results = self._rerank_by_relevance(results, routing)
        
        # 4. Filter by minimum similarity threshold
        min_similarity = self.config.query_processing.min_similarity
        filtered_results = [
            r for r in ranked_results 
            if r.get("final_score", 0.0) >= min_similarity
        ]
        
        # 5. Convert to objects
        articles = self._results_to_articles(filtered_results[:top_k])
        
        return articles, routing
    
    def process_query(
        self,
        query: str,
        top_k: int = 10,
        sentiment_filter: Optional[str] = None
    ) -> List[NewsArticle]:
        """Legacy wrapper for process_query_with_routing."""
        articles, _ = self.process_query_with_routing(query, top_k, sentiment_filter)
        return articles