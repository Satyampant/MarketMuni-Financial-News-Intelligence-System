from app.core.config import Paths
from enum import Enum
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re
from pathlib import Path
import json

from app.core.models import NewsArticle
from app.services.vector_store import VectorStore
from app.agents.entity_extraction import EntityExtractor
from app.agents.stock_impact import StockImpactMapper
from app.core.config_loader import get_config


class QueryStrategy(Enum):
    DIRECT_MENTION = "direct_mention"
    SECTOR_WIDE = "sector_wide"
    REGULATOR_FILTER = "regulator_filter"
    SEMANTIC_THEME = "semantic_theme"


@dataclass
class QueryContext:
    original_query: str
    expanded_terms: List[str]
    companies: List[str]
    sectors: List[str]
    regulators: List[str]
    stock_symbols: List[str]
    strategies: List[QueryStrategy]
    primary_query: str
    context_queries: List[str]
    sentiment_filter: Optional[str] = None


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
        
        alias_path = alias_path or Paths.COMPANY_ALIASES
        sector_path = sector_path or Paths.SECTOR_TICKERS
        
        self.company_aliases = {}
        if alias_path.exists():
            self.company_aliases = json.loads(alias_path.read_text())
        
        self.sector_tickers = {}
        if sector_path.exists():
            self.sector_tickers = json.loads(sector_path.read_text())
        
        # Build reverse mappings for sector lookups
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
    
    def expand_query(self, query: str, sentiment_filter: Optional[str] = None) -> QueryContext:
        """
        Expands the user query into a primary query and targeted context queries
        based on extracted entities (Companies, Sectors, Regulators).
        """
        entities = self.entity_extractor.extract_entities(query)
        
        companies = entities.get("Companies", [])
        sectors = entities.get("Sectors", [])
        regulators = entities.get("Regulators", [])
        
        strategies = self._determine_strategies(query, companies, sectors, regulators)
        
        primary_query = query
        context_queries = []
        expanded_terms = [query]
        stock_symbols = []
        
        # Expand company context (sectors, tickers)
        for company in companies:
            expanded_terms.append(company)
            
            sector = self.company_to_sector.get(company)
            if sector and sector not in sectors:
                sectors.append(sector)
            
            meta = self.company_aliases.get(company, {})
            if isinstance(meta, dict):
                ticker = meta.get("ticker")
                if ticker:
                    stock_symbols.append(ticker)
        
        # Handle SECTOR_WIDE strategy: add industry updates and related tickers
        if QueryStrategy.SECTOR_WIDE in strategies:
            for sector in sectors:
                context_queries.append(f"{sector} sector news")
                context_queries.append(f"{sector} industry update")
                expanded_terms.append(sector)
                
                sector_companies = self.sector_to_companies.get(sector, [])
                for comp in sector_companies:
                    if comp not in companies:
                        companies.append(comp)
                
                sector_tickers = self.sector_tickers.get(sector, [])
                stock_symbols.extend(sector_tickers)
        
        # Handle REGULATOR_FILTER strategy
        if QueryStrategy.REGULATOR_FILTER in strategies:
            for regulator in regulators:
                context_queries.append(f"{regulator} policy")
                context_queries.append(f"{regulator} announcement")
                expanded_terms.append(regulator)
        
        # Handle DIRECT_MENTION strategy: ensure specific company news is fetched
        if QueryStrategy.DIRECT_MENTION in strategies:
            for company in companies:
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
        """Determine applicable query strategies based on entity presence and keywords."""
        strategies = []
        
        if companies:
            strategies.append(QueryStrategy.DIRECT_MENTION)
        
        if sectors or self._is_sector_query(query):
            strategies.append(QueryStrategy.SECTOR_WIDE)
        
        if regulators or self._is_regulator_query(query):
            strategies.append(QueryStrategy.REGULATOR_FILTER)
        
        if self._is_thematic_query(query):
            strategies.append(QueryStrategy.SEMANTIC_THEME)
        
        if not strategies:
            strategies.append(QueryStrategy.SEMANTIC_THEME)
        
        return strategies
    
    def _is_sector_query(self, query: str) -> bool:
        sector_keywords = ["sector", "industry", "industries", "segment"]
        query_lower = query.lower()
        return any(kw in query_lower for kw in sector_keywords)
    
    def _is_regulator_query(self, query: str) -> bool:
        regulator_keywords = ["policy", "regulation", "regulatory", "central bank"]
        query_lower = query.lower()
        return any(kw in query_lower for kw in regulator_keywords)
    
    def _is_thematic_query(self, query: str) -> bool:
        thematic_keywords = [
            "impact", "effect", "influence", "trend", "outlook",
            "rate", "inflation", "growth", "market", "economic"
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in thematic_keywords)
    
    def process_query(
        self,
        query_text: str,
        top_k: int = None,
        min_similarity: float = None,
        sentiment_filter: Optional[str] = None,
    ) -> List[NewsArticle]:
        """
        Executes multi-query retrieval: expands the query, searches the vector store 
        (merging primary and context results), filters by entity relevance, and reranks 
        based on sentiment signal strength.
        """
        config = get_config()
        top_k = top_k or config.query_processing.default_top_k
        min_similarity = min_similarity or config.query_processing.min_similarity

        # Expand query and determine strategies
        context = self.expand_query(query_text, sentiment_filter)
        
        # Multi-Query Retrieval
        all_results = {}  # article_id -> result dict (for deduplication)
        
        # 1. Primary search (high precision)
        primary_results = self.vector_store.search(
            context.primary_query,
            top_k=top_k * 2,
            sentiment_filter=sentiment_filter
        )
        
        for result in primary_results:
            article_id = result["article_id"]
            result["query_source"] = "primary"
            all_results[article_id] = result
        
        # 2. Context searches (strategy-specific)
        for context_query in context.context_queries[:3]:
            context_results = self.vector_store.search(
                context_query,
                top_k=top_k,
                sentiment_filter=sentiment_filter
            )
            
            for result in context_results:
                article_id = result["article_id"]
                # Upsert if new or if similarity score is better
                if article_id not in all_results:
                    result["query_source"] = "context"
                    all_results[article_id] = result
                elif result["similarity"] > all_results[article_id]["similarity"]:
                    result["query_source"] = "context_better"
                    all_results[article_id] = result
        
        merged_results = list(all_results.values())
        filtered_results = self._apply_filters(merged_results, context)
        ranked_results = self._rerank_results(filtered_results, context)
        
        final_results = [
            r for r in ranked_results
            if r["final_score"] >= min_similarity
        ]
        
        return self._results_to_articles(final_results[:top_k])
    
    def _apply_filters(self, results: List[Dict[str, Any]], context: QueryContext) -> List[Dict[str, Any]]:
        """Apply strategy-based scores (Direct Mention, Sector, Regulator) to results."""
        filtered = []
        
        for result in results:
            metadata = result["metadata"]
            entities = metadata.get("entities", {})
            if isinstance(entities, str):
                entities = json.loads(entities)
            
            impacted_stocks = metadata.get("impacted_stocks", [])
            if isinstance(impacted_stocks, str):
                impacted_stocks = json.loads(impacted_stocks)
            
            match_scores = {
                "direct_mention": 0.0,
                "sector_wide": 0.0,
                "regulator_filter": 0.0,
                "semantic_theme": result["similarity"]
            }
            
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
            
            if QueryStrategy.SECTOR_WIDE in context.strategies:
                article_sectors = entities.get("Sectors", [])
                sector_match = any(s in article_sectors for s in context.sectors)
                
                if sector_match:
                    match_scores["sector_wide"] = 0.8
                else:
                    # Indirect match: article companies belong to query sectors
                    article_companies = entities.get("Companies", [])
                    for company in article_companies:
                        if self.company_to_sector.get(company) in context.sectors:
                            match_scores["sector_wide"] = 0.7
                            break
            
            if QueryStrategy.REGULATOR_FILTER in context.strategies:
                article_regulators = entities.get("Regulators", [])
                regulator_match = any(r in article_regulators for r in context.regulators)
                if regulator_match:
                    match_scores["regulator_filter"] = 1.0
            
            max_strategy_score = max(match_scores[s.value] for s in context.strategies)
            
            # Keep result if it matches a strategy or has high semantic similarity
            if max_strategy_score > 0 or result["similarity"] > 0.5:
                result["match_scores"] = match_scores
                result["max_strategy_score"] = max_strategy_score
                filtered.append(result)
        
        return filtered
    
    def _rerank_results(self, results: List[Dict[str, Any]], context: QueryContext) -> List[Dict[str, Any]]:
        """
        Reranks results by combining semantic similarity, strategy alignment, 
        and sentiment signal strength.
        """
        config = get_config()
        for result in results:
            semantic_score = result["similarity"]
            strategy_score = result["max_strategy_score"]
            query_source = result.get("query_source", "context")
            
            primary_boost = 1.1 if query_source == "primary" else 1.0
            
            if QueryStrategy.DIRECT_MENTION in context.strategies:
                weights = config.query_processing.reranking_weights.get('direct_mention')
                if weights:
                    final_score = weights.strategy_weight * strategy_score + weights.semantic_weight * semantic_score
                    
            elif QueryStrategy.REGULATOR_FILTER in context.strategies:
                if result["match_scores"]["regulator_filter"] > 0:
                    # High precision weight for direct regulator matches
                    final_score = 0.8 * strategy_score + 0.2 * semantic_score
                else:
                    final_score = 0.3 * strategy_score + 0.7 * semantic_score
                    
            else:
                # Balanced approach for Sector/Thematic
                final_score = 0.5 * strategy_score + 0.5 * semantic_score
            
            final_score *= primary_boost
            
            # Boost articles with high sentiment signal strength (0-100 scale)
            sentiment_boost = 1.0
            metadata = result["metadata"]
            if "sentiment_signal_strength" in metadata:
                signal_strength = float(metadata.get("sentiment_signal_strength", 0.0))
                # Max boost: 1.5x for signal_strength = 100
                sentiment_boost = 1.0 + (signal_strength / 200.0)
            
            final_score *= sentiment_boost
            
            result["final_score"] = min(final_score, 1.0)
            result["sentiment_boost"] = sentiment_boost
        
        return sorted(results, key=lambda x: x["final_score"], reverse=True)
    
    def _results_to_articles(self, results: List[Dict[str, Any]]) -> List[NewsArticle]:
        """Convert search results to NewsArticle objects with parsed metadata."""
        articles = []
        
        for result in results:
            metadata = result["metadata"]
            
            entities = metadata.get("entities", {})
            if isinstance(entities, str):
                entities = json.loads(entities)
            
            impacted_stocks = metadata.get("impacted_stocks", [])
            if isinstance(impacted_stocks, str):
                impacted_stocks = json.loads(impacted_stocks)
            
            sentiment = metadata.get("sentiment")
            if isinstance(sentiment, str):
                sentiment = json.loads(sentiment) if sentiment != "null" else None
            
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
            article.query_source = result.get("query_source", "unknown")
            article.sentiment_boost = result.get("sentiment_boost", 1.0)
            
            articles.append(article)
        
        return articles
    
    def explain_query(self, query_text: str, sentiment_filter: Optional[str] = None) -> Dict[str, Any]:
        """Debug helper to explain query expansion and strategy selection."""
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