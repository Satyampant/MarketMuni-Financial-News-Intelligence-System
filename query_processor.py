from enum import Enum
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re
from pathlib import Path
import json

from news_storage import NewsArticle
from vector_store import VectorStore
from entity_extraction import EntityExtractor
from stock_impact import StockImpactMapper

MODULE_DIR = Path(__file__).parent


class QueryStrategy(Enum):
    """Query processing strategies"""
    DIRECT_MENTION = "direct_mention"
    SECTOR_WIDE = "sector_wide"
    REGULATOR_FILTER = "regulator_filter"
    SEMANTIC_THEME = "semantic_theme"


@dataclass
class QueryContext:
    """Expanded query context with entities, strategies, and sentiment filter"""
    original_query: str
    expanded_terms: List[str]
    companies: List[str]
    sectors: List[str]
    regulators: List[str]
    stock_symbols: List[str]
    strategies: List[QueryStrategy]
    primary_query: str
    context_queries: List[str]
    sentiment_filter: Optional[str] = None  # "Bullish" | "Bearish" | "Neutral"


class QueryProcessor:
    def __init__(
        self,
        vector_store: VectorStore,
        entity_extractor: EntityExtractor,
        stock_mapper: StockImpactMapper,
        alias_path: Optional[Path] = None,
        sector_path: Optional[Path] = None,
    ):
        self.vector_store = vector_store
        self.entity_extractor = entity_extractor
        self.stock_mapper = stock_mapper
        
        # Load mappings
        alias_path = alias_path or MODULE_DIR / "company_aliases.json"
        sector_path = sector_path or MODULE_DIR / "sector_tickers.json"
        
        self.company_aliases = {}
        if alias_path.exists():
            self.company_aliases = json.loads(alias_path.read_text())
        
        self.sector_tickers = {}
        if sector_path.exists():
            self.sector_tickers = json.loads(sector_path.read_text())
        
        # Build reverse mappings
        self.company_to_sector = {}
        self.sector_to_companies = {}
        
        for company, meta in self.company_aliases.items():
            if isinstance(meta, dict):
                sector = meta.get("sector", "")
                if sector:
                    self.company_to_sector[company] = sector
                    if sector not in self.sector_to_companies:
                        self.sector_to_companies[sector] = []
                    self.sector_to_companies[sector].append(company)
    
    def expand_query(
        self, 
        query: str, 
        sentiment_filter: Optional[str] = None
    ) -> QueryContext:
        """
        Expand query with related entities and create multiple focused queries.
        
        Multi-Query Approach:
        - Primary Query: The original user query (preserves semantic meaning)
        - Context Queries: Additional targeted searches based on detected strategies
        - Sentiment Filter: Optional filter for bullish/bearish/neutral articles
        """
        # Extract entities from query
        entities = self.entity_extractor.extract_entities(query)
        
        companies = entities.get("Companies", [])
        sectors = entities.get("Sectors", [])
        regulators = entities.get("Regulators", [])
        
        # Determine query strategies
        strategies = self._determine_strategies(query, companies, sectors, regulators)
        
        # Build primary query (always the original query)
        primary_query = query
        
        # Build context queries based on strategies
        context_queries = []
        expanded_terms = [query]
        stock_symbols = []
        
        # Add company-related expansions
        for company in companies:
            expanded_terms.append(company)
            
            # Add sector context
            sector = self.company_to_sector.get(company)
            if sector and sector not in sectors:
                sectors.append(sector)
            
            # Add stock symbol
            meta = self.company_aliases.get(company, {})
            if isinstance(meta, dict):
                ticker = meta.get("ticker")
                if ticker:
                    stock_symbols.append(ticker)
        
        # SECTOR_WIDE strategy: Add dedicated sector searches
        if QueryStrategy.SECTOR_WIDE in strategies:
            for sector in sectors:
                # Add focused sector query
                context_queries.append(f"{sector} sector news")
                context_queries.append(f"{sector} industry update")
                expanded_terms.append(sector)
                
                # Add sector companies for filtering
                sector_companies = self.sector_to_companies.get(sector, [])
                for comp in sector_companies:
                    if comp not in companies:
                        companies.append(comp)
                
                # Add sector tickers
                sector_tickers = self.sector_tickers.get(sector, [])
                stock_symbols.extend(sector_tickers)
        
        # REGULATOR_FILTER strategy: Add dedicated regulator searches
        if QueryStrategy.REGULATOR_FILTER in strategies:
            for regulator in regulators:
                context_queries.append(f"{regulator} policy")
                context_queries.append(f"{regulator} announcement")
                expanded_terms.append(regulator)
        
        # DIRECT_MENTION strategy: Add company-specific searches if not already in query
        if QueryStrategy.DIRECT_MENTION in strategies:
            for company in companies:
                # Only add if company is not prominently in original query
                if company.lower() not in query.lower():
                    context_queries.append(f"{company} news")
        
        return QueryContext(
            original_query=query,
            expanded_terms=list(set(expanded_terms)),
            companies=companies,
            sectors=sectors,
            regulators=regulators,
            stock_symbols=list(set(stock_symbols)),
            strategies=strategies,
            primary_query=primary_query,
            context_queries=list(set(context_queries)),
            sentiment_filter=sentiment_filter
        )
    
    def _determine_strategies(
        self,
        query: str,
        companies: List[str],
        sectors: List[str],
        regulators: List[str]
    ) -> List[QueryStrategy]:
        """Determine which query strategies to apply"""
        strategies = []
        
        # Check for direct company mentions
        if companies:
            strategies.append(QueryStrategy.DIRECT_MENTION)
        
        # Check for sector queries
        if sectors or self._is_sector_query(query):
            strategies.append(QueryStrategy.SECTOR_WIDE)
        
        # Check for regulator queries
        if regulators or self._is_regulator_query(query):
            strategies.append(QueryStrategy.REGULATOR_FILTER)
        
        # Check for thematic queries
        if self._is_thematic_query(query):
            strategies.append(QueryStrategy.SEMANTIC_THEME)
        
        # Default to semantic theme if no specific strategy
        if not strategies:
            strategies.append(QueryStrategy.SEMANTIC_THEME)
        
        return strategies
    
    def _is_sector_query(self, query: str) -> bool:
        """Check if query is sector-related"""
        sector_keywords = ["sector", "industry", "industries", "segment"]
        query_lower = query.lower()
        return any(kw in query_lower for kw in sector_keywords)
    
    def _is_regulator_query(self, query: str) -> bool:
        """Check if query is regulator-related"""
        regulator_keywords = ["policy", "regulation", "regulatory", "central bank"]
        query_lower = query.lower()
        return any(kw in query_lower for kw in regulator_keywords)
    
    def _is_thematic_query(self, query: str) -> bool:
        """Check if query is thematic (e.g., 'interest rate impact')"""
        thematic_keywords = [
            "impact", "effect", "influence", "trend", "outlook",
            "rate", "inflation", "growth", "market", "economic"
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in thematic_keywords)
    
    def process_query(
        self,
        query_text: str,
        top_k: int = 10,
        min_similarity: float = 0.3,
        sentiment_filter: Optional[str] = None,
    ) -> List[NewsArticle]:
        """
        Process query using multi-query retrieval strategy with sentiment filtering.
        
        Multi-Query Retrieval:
        1. Primary Search: Search with original query (high precision)
        2. Context Searches: Additional targeted searches based on strategies
        3. Merge & Deduplicate: Combine results, remove duplicates
        4. Sentiment Filtering: Filter by bullish/bearish/neutral if specified
        5. Filter & Rank: Apply entity filtering and re-rank with sentiment boost
        
        Args:
            query_text: Natural language query
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold
            sentiment_filter: Optional "Bullish", "Bearish", or "Neutral" filter
            
        Returns:
            List of NewsArticle objects ranked by relevance
        """
        # Step 1: Expand query and determine strategies
        context = self.expand_query(query_text, sentiment_filter)
        
        # Step 2: Multi-Query Retrieval
        all_results = {}  # article_id -> result dict (for deduplication)
        
        # 2a. Primary search (original query - highest quality)
        primary_results = self.vector_store.search(
            context.primary_query,
            top_k=top_k * 2,
            sentiment_filter=sentiment_filter  # Apply sentiment filter in vector search
        )
        
        for result in primary_results:
            article_id = result["article_id"]
            result["query_source"] = "primary"
            all_results[article_id] = result
        
        # 2b. Context searches (strategy-specific queries)
        for context_query in context.context_queries[:3]:  # Limit to top 3 context queries
            context_results = self.vector_store.search(
                context_query,
                top_k=top_k,
                sentiment_filter=sentiment_filter  # Apply sentiment filter
            )
            
            for result in context_results:
                article_id = result["article_id"]
                # Only add if not already found, or if similarity is higher
                if article_id not in all_results:
                    result["query_source"] = "context"
                    all_results[article_id] = result
                elif result["similarity"] > all_results[article_id]["similarity"]:
                    # Update with better similarity score
                    result["query_source"] = "context_better"
                    all_results[article_id] = result
        
        # Step 3: Convert to list and apply filters
        merged_results = list(all_results.values())
        filtered_results = self._apply_filters(merged_results, context)
        
        # Step 4: Re-rank results with sentiment boosting
        ranked_results = self._rerank_results(filtered_results, context)
        
        # Step 5: Filter by minimum similarity
        final_results = [
            r for r in ranked_results
            if r["final_score"] >= min_similarity
        ]
        
        # Step 6: Convert to NewsArticle objects and return top_k
        articles = self._results_to_articles(final_results[:top_k])
        
        return articles
    
    def _apply_filters(
        self,
        results: List[Dict[str, Any]],
        context: QueryContext
    ) -> List[Dict[str, Any]]:
        """Apply strategy-based filters to results"""
        filtered = []
        
        for result in results:
            metadata = result["metadata"]
            entities = metadata.get("entities", {})
            if isinstance(entities, str):
                entities = json.loads(entities)
            
            impacted_stocks = metadata.get("impacted_stocks", [])
            if isinstance(impacted_stocks, str):
                impacted_stocks = json.loads(impacted_stocks)
            
            # Calculate match scores for different strategies
            match_scores = {
                "direct_mention": 0.0,
                "sector_wide": 0.0,
                "regulator_filter": 0.0,
                "semantic_theme": result["similarity"]
            }
            
            # Direct mention scoring
            if QueryStrategy.DIRECT_MENTION in context.strategies:
                article_companies = entities.get("Companies", [])
                company_match = any(c in article_companies for c in context.companies)
                
                stock_symbols = [s["symbol"] for s in impacted_stocks]
                stock_match = any(sym in stock_symbols for sym in context.stock_symbols)
                
                if company_match:
                    match_scores["direct_mention"] = 1.0
                elif stock_match:
                    direct_stocks = [s for s in impacted_stocks if s["impact_type"] == "direct"]
                    if any(s["symbol"] in context.stock_symbols for s in direct_stocks):
                        match_scores["direct_mention"] = 1.0
            
            # Sector-wide scoring
            if QueryStrategy.SECTOR_WIDE in context.strategies:
                article_sectors = entities.get("Sectors", [])
                sector_match = any(s in article_sectors for s in context.sectors)
                
                if sector_match:
                    match_scores["sector_wide"] = 0.8
                else:
                    # Check if article's companies belong to query sectors
                    article_companies = entities.get("Companies", [])
                    for company in article_companies:
                        if self.company_to_sector.get(company) in context.sectors:
                            match_scores["sector_wide"] = 0.7
                            break
            
            # Regulator filter scoring
            if QueryStrategy.REGULATOR_FILTER in context.strategies:
                article_regulators = entities.get("Regulators", [])
                regulator_match = any(r in article_regulators for r in context.regulators)
                
                if regulator_match:
                    match_scores["regulator_filter"] = 1.0
            
            # Calculate final filter score
            max_strategy_score = max(
                match_scores[s.value]
                for s in context.strategies
            )
            
            # Keep result if it matches at least one strategy or has high semantic similarity
            if max_strategy_score > 0 or result["similarity"] > 0.5:
                result["match_scores"] = match_scores
                result["max_strategy_score"] = max_strategy_score
                filtered.append(result)
        
        return filtered
    
    def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        context: QueryContext
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results with improved scoring and sentiment boosting.
        
        Scoring Strategy:
        - Primary query results get a boost
        - Direct matches get highest weight
        - Regulator matches get high precision weight
        - Sector/thematic use balanced weights
        - High signal_strength articles get sentiment boost (up to 1.5x)
        """
        for result in results:
            semantic_score = result["similarity"]
            strategy_score = result["max_strategy_score"]
            query_source = result.get("query_source", "context")
            
            # Boost for primary query results
            primary_boost = 1.1 if query_source == "primary" else 1.0
            
            # Strategy-based weighting
            if QueryStrategy.DIRECT_MENTION in context.strategies:
                if result["match_scores"]["direct_mention"] > 0:
                    # High weight on direct matches
                    final_score = 0.7 * strategy_score + 0.3 * semantic_score
                else:
                    # Balanced for non-direct but relevant
                    final_score = 0.4 * strategy_score + 0.6 * semantic_score
                    
            elif QueryStrategy.REGULATOR_FILTER in context.strategies:
                if result["match_scores"]["regulator_filter"] > 0:
                    # Very high precision for regulator matches
                    final_score = 0.8 * strategy_score + 0.2 * semantic_score
                else:
                    # Lower weight if no regulator match
                    final_score = 0.3 * strategy_score + 0.7 * semantic_score
                    
            else:
                # Sector or thematic queries - balanced approach
                final_score = 0.5 * strategy_score + 0.5 * semantic_score
            
            # Apply primary boost
            final_score *= primary_boost
            
            # SENTIMENT BOOST: Prioritize high-signal articles
            sentiment_boost = 1.0
            metadata = result["metadata"]
            
            if "sentiment_signal_strength" in metadata:
                signal_strength = float(metadata.get("sentiment_signal_strength", 0.0))
                
                # Boost articles with high signal strength (0-100 scale)
                # Max boost: 1.5x for signal_strength = 100
                sentiment_boost = 1.0 + (signal_strength / 200.0)
            
            # Apply sentiment boost
            final_score *= sentiment_boost
            
            result["final_score"] = min(final_score, 1.0)  # Cap at 1.0
            result["sentiment_boost"] = sentiment_boost
        
        # Sort by final score
        ranked = sorted(results, key=lambda x: x["final_score"], reverse=True)
        
        return ranked
    
    def _results_to_articles(self, results: List[Dict[str, Any]]) -> List[NewsArticle]:
        """Convert search results to NewsArticle objects"""
        articles = []
        
        for result in results:
            metadata = result["metadata"]
            
            # Parse entities
            entities = metadata.get("entities", {})
            if isinstance(entities, str):
                entities = json.loads(entities)
            
            # Parse impacted stocks
            impacted_stocks = metadata.get("impacted_stocks", [])
            if isinstance(impacted_stocks, str):
                impacted_stocks = json.loads(impacted_stocks)
            
            # Parse sentiment
            sentiment = metadata.get("sentiment")
            if isinstance(sentiment, str):
                sentiment = json.loads(sentiment) if sentiment != "null" else None
            
            # Create NewsArticle
            article = NewsArticle(
                id=metadata["article_id"],
                title=metadata["title"],
                content=result["document"],
                source=metadata["source"],
                timestamp=metadata["timestamp"],
                raw_text=result["document"]
            )
            
            # Add enriched data
            article.entities = entities
            article.impacted_stocks = impacted_stocks
            article.sentiment = sentiment
            
            # Add score metadata
            article.relevance_score = result.get("final_score", result["similarity"])
            article.query_source = result.get("query_source", "unknown")
            article.sentiment_boost = result.get("sentiment_boost", 1.0)
            
            articles.append(article)
        
        return articles
    
    def explain_query(self, query_text: str, sentiment_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Explain how a query will be processed (for debugging).
        
        Returns:
            Dictionary with query expansion and strategy information
        """
        context = self.expand_query(query_text, sentiment_filter)
        
        return {
            "original_query": context.original_query,
            "strategies": [s.value for s in context.strategies],
            "primary_query": context.primary_query,
            "context_queries": context.context_queries,
            "expanded_terms": context.expanded_terms,
            "identified_entities": {
                "companies": context.companies,
                "sectors": context.sectors,
                "regulators": context.regulators,
                "stock_symbols": context.stock_symbols
            },
            "sentiment_filter": context.sentiment_filter
        }