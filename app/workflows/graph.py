"""
Updated app/workflows/graph.py - With LLM Supply Chain Analysis
Full LLM pipeline: Entity Extraction → Stock Impact → Sentiment → Supply Chain (LLM-based)
"""

from typing import TypedDict, List, Annotated, Optional, Dict, Any
import operator
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from app.core.models import NewsArticle
from app.services.storage import NewsStorage
from app.agents.deduplication import DeduplicationAgent
from app.agents.llm_entity_extractor import LLMEntityExtractor
from app.agents.llm_stock_mapper import LLMStockImpactMapper
from app.agents.llm_sentiment import LLMSentimentAnalyzer
from app.agents.llm_supply_chain import LLMSupplyChainAnalyzer  # NEW: LLM supply chain
from app.services.vector_store import VectorStore
from app.agents.query_processor import QueryProcessor
from app.core.config_loader import get_config
from app.core.llm_schemas import EntityExtractionSchema, SentimentAnalysisSchema


class NewsIntelligenceState(TypedDict):
    """State definition for the news processing pipeline."""
    articles: Annotated[List[NewsArticle], operator.add]
    current_article: Optional[NewsArticle]
    duplicates: Annotated[List[str], operator.add]
    entities_schema: Optional[EntityExtractionSchema]
    entities: Optional[dict]
    impacted_stocks: Optional[List[dict]]
    sentiment_schema: Optional[SentimentAnalysisSchema]
    cross_impacts: Optional[List[Dict]]
    sentiment: Optional[dict]
    query_text: Optional[str]
    sentiment_filter: Optional[str]
    query_results: Annotated[List[NewsArticle], operator.add]
    error: Optional[str]
    stats: dict


class NewsIntelligenceGraph:
    """
    LangGraph-based multi-agent pipeline for financial news intelligence.
    Fully LLM-powered approach for all agents.
    """
    
    def __init__(
        self,
        storage: NewsStorage,
        vector_store: VectorStore,
        dedup_agent: Optional[DeduplicationAgent] = None,
        entity_extractor: Optional[LLMEntityExtractor] = None,
        stock_mapper: Optional[LLMStockImpactMapper] = None,
        sentiment_analyzer: Optional[LLMSentimentAnalyzer] = None,
        supply_chain_analyzer: Optional[LLMSupplyChainAnalyzer] = None  # NEW: LLM supply chain
    ):
        config = get_config()

        self.storage = storage
        self.vector_store = vector_store
        
        self.dedup_agent = dedup_agent or DeduplicationAgent()
        
        # LLM entity extraction
        self.entity_extractor = entity_extractor or LLMEntityExtractor(
            enable_caching=config.performance.cache_embeddings
        )
        
        # LLM stock impact mapper
        self.stock_mapper = stock_mapper or LLMStockImpactMapper()
        
        # LLM sentiment analyzer
        self.sentiment_analyzer = sentiment_analyzer or LLMSentimentAnalyzer(
            use_entity_context=True
        )
        
        # NEW: LLM supply chain analyzer (replaces rule-based approach)
        self.supply_chain_analyzer = supply_chain_analyzer or LLMSupplyChainAnalyzer()
        
        # Query processor (uses legacy entity extractor for query expansion)
        self.query_processor = QueryProcessor(
            vector_store=self.vector_store,
            entity_extractor=self.entity_extractor,
            stock_mapper=self.stock_mapper
        )
        
        self.app = self._build_ingestion_graph()
        self.query_app = self._build_query_graph()
        
        print("✓ NewsIntelligenceGraph initialized with full LLM pipeline")
        print("  - Entity extraction: LLM")
        print("  - Stock impact mapping: LLM")
        print("  - Sentiment analysis: LLM")
        print("  - Supply chain analysis: LLM")
    
    def _build_ingestion_graph(self):
        """Construct the main news ingestion and analysis pipeline."""
        graph_builder = StateGraph(NewsIntelligenceState)
        
        # Nodes
        graph_builder.add_node("ingestion", self._ingestion_agent)
        graph_builder.add_node("deduplication", self._deduplication_agent)
        graph_builder.add_node("entity_extraction", self._entity_extraction_agent)
        graph_builder.add_node("impact_mapper", self._impact_mapper_agent)
        graph_builder.add_node("sentiment_analysis", self._sentiment_analysis_agent)
        graph_builder.add_node("cross_impact", self._cross_impact_agent)
        graph_builder.add_node("indexing", self._indexing_agent)
        
        # Edges
        graph_builder.add_edge(START, "ingestion")
        graph_builder.add_edge("ingestion", "deduplication")
        graph_builder.add_edge("deduplication", "entity_extraction")
        graph_builder.add_edge("entity_extraction", "impact_mapper")
        graph_builder.add_edge("impact_mapper", "sentiment_analysis")
        graph_builder.add_edge("sentiment_analysis", "cross_impact")
        graph_builder.add_edge("cross_impact", "indexing")
        graph_builder.add_edge("indexing", END)
        
        return graph_builder.compile()

    def _build_query_graph(self):
        """Construct the retrieval pipeline."""
        query_graph = StateGraph(NewsIntelligenceState)
        query_graph.add_node("query", self._query_agent)
        query_graph.add_edge(START, "query")
        query_graph.add_edge("query", END)
        return query_graph.compile()
    
    def _ingestion_agent(self, state: NewsIntelligenceState) -> dict:
        """Validates and prepares the article object."""
        article = state.get("current_article")
        
        if not article:
            return {"error": "No article provided for ingestion"}
        
        if not all([article.id, article.title, article.content, article.source]):
            return {"error": "Article missing required fields"}
        
        if isinstance(article.timestamp, str):
            article.timestamp = datetime.fromisoformat(article.timestamp)
        
        return {
            "current_article": article,
            "stats": {
                "ingestion_time": datetime.now().isoformat(),
                "article_id": article.id
            }
        }
    
    def _deduplication_agent(self, state: NewsIntelligenceState) -> dict:
        """Checks for existing articles and consolidates if duplicates found."""
        article = state["current_article"]
        existing_articles = self.storage.get_all_articles()
        
        duplicate_ids = self.dedup_agent.find_duplicates(article, existing_articles)
        
        stats = state.get("stats", {})
        stats["duplicates_found"] = len(duplicate_ids)
        stats["is_duplicate"] = len(duplicate_ids) > 0
        
        if duplicate_ids:
            duplicates = [self.storage.get_by_id(dup_id) for dup_id in duplicate_ids]
            duplicates.append(article)
            consolidated = self.dedup_agent.consolidate_duplicates(duplicates)
            
            return {
                "current_article": consolidated,
                "duplicates": duplicate_ids,
                "stats": stats
            }
        
        return {"duplicates": [], "stats": stats}
    
    def _entity_extraction_agent(self, state: NewsIntelligenceState) -> dict:
        """Extracts entities using LLM with structured output."""
        article = state["current_article"]
        
        # LLM extraction returns EntityExtractionSchema
        entities_schema = self.entity_extractor.extract_entities(article)
        
        # Store rich data in article
        article.set_entities_rich(entities_schema)
        
        stats = state.get("stats", {})
        stats["entities_extracted"] = {
            "companies": len(entities_schema.companies),
            "sectors": len(entities_schema.sectors),
            "regulators": len(entities_schema.regulators),
            "people": len(entities_schema.people),
            "events": len(entities_schema.events)
        }
        
        tickers_extracted = [c.ticker_symbol for c in entities_schema.companies if c.ticker_symbol]
        stats["tickers_extracted"] = len(tickers_extracted)
        
        company_confidences = [c.confidence for c in entities_schema.companies]
        if company_confidences:
            stats["avg_company_confidence"] = round(
                sum(company_confidences) / len(company_confidences), 2
            )
        
        stats["entity_extraction_method"] = "llm"
        stats["extraction_reasoning"] = entities_schema.extraction_reasoning
        
        return {
            "current_article": article,
            "entities_schema": entities_schema,
            "entities": article.entities,
            "stats": stats
        }
    
    def _impact_mapper_agent(self, state: NewsIntelligenceState) -> dict:
        """Maps extracted entities to specific stock tickers using LLM reasoning."""
        article = state["current_article"]
        entities_schema = state["entities_schema"]
        
        if not entities_schema:
            # Fallback: reconstruct from article.entities_rich
            from app.core.llm_schemas import EntityExtractionSchema
            if hasattr(article, 'entities_rich') and article.entities_rich:
                entities_schema = EntityExtractionSchema.model_validate(article.entities_rich)
            else:
                raise ValueError("No entity schema available for impact mapping")
        
        # LLM-based impact mapping
        impact_result = self.stock_mapper.map_to_stocks(entities_schema, article)
        
        # Convert to legacy format and attach to article
        impact_dicts = [
            {
                "symbol": stock.symbol,
                "company_name": stock.company_name,
                "confidence": stock.confidence,
                "impact_type": stock.impact_type.value,
                "reasoning": stock.reasoning
            }
            for stock in impact_result.impacted_stocks
        ]
        article.impacted_stocks = impact_dicts
        
        stats = state.get("stats", {})
        stats["stocks_impacted"] = len(impact_result.impacted_stocks)
        stats["impact_breakdown"] = {
            "direct": sum(1 for s in impact_result.impacted_stocks if s.impact_type.value == "direct"),
            "sector": sum(1 for s in impact_result.impacted_stocks if s.impact_type.value == "sector"),
            "regulatory": sum(1 for s in impact_result.impacted_stocks if s.impact_type.value == "regulatory")
        }
        stats["stock_impact_method"] = "llm"
        stats["overall_market_impact"] = impact_result.overall_market_impact
        
        return {
            "current_article": article,
            "impacted_stocks": impact_dicts,
            "stats": stats
        }
    
    def _sentiment_analysis_agent(self, state: NewsIntelligenceState) -> dict:
        """Performs LLM-based sentiment analysis with entity context."""
        article = state["current_article"]
        entities_schema = state.get("entities_schema")
        
        # Reconstruct entity schema if not in state
        if not entities_schema and hasattr(article, 'entities_rich') and article.entities_rich:
            from app.core.llm_schemas import EntityExtractionSchema
            entities_schema = EntityExtractionSchema.model_validate(article.entities_rich)
        
        # LLM sentiment analysis with entity context
        sentiment_schema = self.sentiment_analyzer.analyze_sentiment(article, entities_schema)
        
        # Attach to article (convert to legacy format)
        self.sentiment_analyzer.analyze_and_attach(article, entities_schema)
        sentiment_data = article.get_sentiment()
        
        stats = state.get("stats", {})
        stats["sentiment_analyzed"] = True
        
        if sentiment_data:
            stats.update({
                "sentiment_classification": sentiment_data.classification,
                "sentiment_confidence": sentiment_data.confidence_score,
                "sentiment_signal_strength": sentiment_data.signal_strength,
                "sentiment_method": sentiment_data.analysis_method
            })
            
            # Extract key factors count
            key_factors = sentiment_data.sentiment_breakdown.get("key_factors", [])
            stats["sentiment_key_factors_count"] = len(key_factors)
        
        return {
            "current_article": article,
            "sentiment_schema": sentiment_schema,
            "sentiment": sentiment_data.to_dict() if sentiment_data else None,
            "stats": stats
        }
    
    def _cross_impact_agent(self, state: NewsIntelligenceState) -> dict:
        """
        Analyzes supply chain relationships using LLM reasoning.
        UPDATED: Uses LLMSupplyChainAnalyzer instead of rule-based approach.
        """
        article = state["current_article"]
        entities_schema = state.get("entities_schema")
        sentiment_schema = state.get("sentiment_schema")
        
        # Reconstruct schemas if not in state
        if not entities_schema and hasattr(article, 'entities_rich') and article.entities_rich:
            from app.core.llm_schemas import EntityExtractionSchema
            entities_schema = EntityExtractionSchema.model_validate(article.entities_rich)
        
        if not sentiment_schema and article.has_sentiment():
            from app.core.llm_schemas import SentimentAnalysisSchema, SentimentClassification
            sentiment_data = article.get_sentiment()
            sentiment_schema = SentimentAnalysisSchema(
                classification=SentimentClassification(sentiment_data.classification),
                confidence_score=sentiment_data.confidence_score,
                key_factors=sentiment_data.sentiment_breakdown.get("key_factors", ["N/A"]),
                signal_strength=sentiment_data.signal_strength,
                sentiment_breakdown=sentiment_data.sentiment_breakdown.get("sentiment_percentages"),
                entity_influence=sentiment_data.sentiment_breakdown.get("entity_influence")
            )
        
        # Check if we have required data for supply chain analysis
        if not entities_schema or not sentiment_schema:
            stats = state.get("stats", {})
            stats.update({
                "cross_impacts_found": 0,
                "upstream_dependencies": 0,
                "downstream_impacts": 0,
                "supply_chain_method": "llm",
                "supply_chain_skipped": "Missing entities or sentiment data"
            })
            return {
                "current_article": article,
                "cross_impacts": [],
                "stats": stats
            }
        
        # Check if sectors are present (required for supply chain analysis)
        if not entities_schema.sectors:
            stats = state.get("stats", {})
            stats.update({
                "cross_impacts_found": 0,
                "upstream_dependencies": 0,
                "downstream_impacts": 0,
                "supply_chain_method": "llm",
                "supply_chain_skipped": "No sectors identified"
            })
            return {
                "current_article": article,
                "cross_impacts": [],
                "stats": stats
            }
        
        # LLM-based supply chain analysis
        supply_chain_result = self.supply_chain_analyzer.generate_cross_impact_insights(
            article, entities_schema, sentiment_schema
        )
        
        # Convert to legacy format and attach to article
        all_impacts = (
            supply_chain_result.upstream_impacts +
            supply_chain_result.downstream_impacts
        )
        
        cross_impact_dicts = [
            {
                "source_sector": impact.source_sector,
                "target_sector": impact.target_sector,
                "relationship_type": impact.relationship_type.value,
                "impact_score": impact.impact_score,
                "dependency_weight": impact.dependency_weight,
                "reasoning": impact.reasoning,
                "impacted_stocks": impact.impacted_stocks,
                "time_horizon": impact.time_horizon
            }
            for impact in all_impacts
        ]
        
        article.set_cross_impacts(cross_impact_dicts)
        
        upstream_count = len(supply_chain_result.upstream_impacts)
        downstream_count = len(supply_chain_result.downstream_impacts)
        
        stats = state.get("stats", {})
        stats["cross_impacts_found"] = len(all_impacts)
        stats["upstream_dependencies"] = upstream_count
        stats["downstream_impacts"] = downstream_count
        stats["supply_chain_method"] = "llm"
        stats["total_sectors_impacted"] = supply_chain_result.total_sectors_impacted
        
        if all_impacts:
            top_impact = all_impacts[0]
            stats["top_cross_impact"] = {
                "target_sector": top_impact.get("target_sector"),
                "impact_score": top_impact.get("impact_score"),
                "relationship_type": top_impact.get("relationship_type")
            }
        
        return {
            "current_article": article,
            "cross_impacts": cross_impact_dicts,
            "stats": stats
        }
    
    def _indexing_agent(self, state: NewsIntelligenceState) -> dict:
        """Persists the article in storage and vector database."""
        article = state["current_article"]
        
        self.storage.add_article(article)
        self.vector_store.index_article(article)
        
        stats = state.get("stats", {})
        stats["indexed"] = True
        stats["total_articles"] = self.storage.article_count()
        stats["vector_store_count"] = self.vector_store.count()
        
        return {
            "articles": [article],
            "stats": stats
        }
    
    def _query_agent(self, state: NewsIntelligenceState) -> dict:
        """Retrieves relevant articles based on query and sentiment filter."""
        query_text = state.get("query_text")
        sentiment_filter = state.get("sentiment_filter")

        if not query_text:
            return {"error": "No query text provided"}
        
        if sentiment_filter:
            valid_sentiments = ["Bullish", "Bearish", "Neutral"]
            if sentiment_filter not in valid_sentiments:
                return {"error": f"Invalid sentiment filter: {sentiment_filter}. Must be {valid_sentiments}"}
        
        results = self.query_processor.process_query(
            query_text, 
            top_k=10, 
            sentiment_filter=sentiment_filter
        )
        
        stats = {
            "query_time": datetime.now().isoformat(),
            "results_count": len(results),
            "query": query_text,
            "sentiment_filter": sentiment_filter,
            "sentiment_filter_applied": sentiment_filter is not None
        }
        
        return {
            "query_results": results,
            "stats": stats
        }
    
    def run_pipeline(self, article: NewsArticle) -> dict:
        """Executes the full ingestion pipeline for a single article."""
        initial_state = {
            "articles": [],
            "current_article": article,
            "duplicates": [],
            "entities_schema": None,
            "entities": None,
            "impacted_stocks": None,
            "sentiment_schema": None,
            "cross_impacts": None,
            "sentiment": None,
            "query_text": None,
            "sentiment_filter": None,
            "query_results": [],
            "error": None,
            "stats": {}
        }
        
        return self.app.invoke(initial_state)
    
    def run_query(self, query_text: str, sentiment_filter: Optional[str] = None) -> dict:
        """Executes the query pipeline."""
        initial_state = {
            "articles": [],
            "current_article": None,
            "duplicates": [],
            "entities_schema": None,
            "entities": None,
            "impacted_stocks": None,
            "sentiment_schema": None,
            "cross_impacts": None,
            "sentiment": None,
            "query_text": query_text,
            "sentiment_filter": sentiment_filter,
            "query_results": [],
            "error": None,
            "stats": {}
        }
        
        return self.query_app.invoke(initial_state)
    
    def get_stats(self) -> dict:
        """Returns system statistics including storage counts and analysis metadata."""
        total_articles = self.storage.article_count()
        vector_count = self.vector_store.count()
        
        sentiment_stats = self.vector_store.get_sentiment_statistics()
        
        # Entity extraction stats
        entity_stats = self.entity_extractor.get_cache_stats()
        
        # Stock impact stats
        articles = self.storage.get_all_articles()
        stock_impact_stats = self.stock_mapper.get_impact_statistics(articles)
        
        # Sentiment statistics from analyzer
        sentiment_analyzer_stats = self.sentiment_analyzer.get_sentiment_statistics(articles)
        
        # NEW: Supply chain statistics from LLM analyzer
        supply_chain_stats = self.supply_chain_analyzer.get_impact_statistics(articles)
        
        return {
            "total_articles_stored": total_articles,
            "vector_store_count": vector_count,
            "dedup_threshold": {
                "bi_encoder": self.dedup_agent.bi_threshold,
                "cross_encoder": self.dedup_agent.cross_threshold
            },
            "entity_extraction": {
                "method": "llm",
                "cache_stats": entity_stats
            },
            "stock_impact_mapping": {
                "method": "llm",
                "statistics": stock_impact_stats
            },
            "sentiment_analysis": {
                "method": "llm",
                "vector_store_stats": sentiment_stats,
                "analyzer_stats": sentiment_analyzer_stats
            },
            "supply_chain_analysis": {
                "method": "llm",
                "statistics": supply_chain_stats
            },
            "status": "operational"
        }