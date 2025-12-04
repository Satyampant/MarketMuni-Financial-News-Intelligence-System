from typing import TypedDict, List, Annotated, Optional, Dict, Any
import operator
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from app.core.models import NewsArticle
from app.services.storage import NewsStorage
from app.agents.deduplication import DeduplicationAgent
from app.agents.entity_extraction import EntityExtractor
from app.agents.stock_impact import StockImpactMapper
from app.services.vector_store import VectorStore
from app.agents.query_processor import QueryProcessor
from app.agents.sentiment.hybrid import HybridSentimentClassifier
from app.agents.supply_chain import SupplyChainImpactMapper
from app.core.config_loader import get_config


class NewsIntelligenceState(TypedDict):
    """State definition for the news processing pipeline."""
    articles: Annotated[List[NewsArticle], operator.add]
    current_article: Optional[NewsArticle]
    duplicates: Annotated[List[str], operator.add]
    entities: Optional[dict]
    impacted_stocks: Optional[List[dict]]
    cross_impacts: Optional[List[Dict]]
    sentiment: Optional[dict]
    query_text: Optional[str]
    sentiment_filter: Optional[str]
    query_results: Annotated[List[NewsArticle], operator.add]
    error: Optional[str]
    stats: dict


class NewsIntelligenceGraph:
    def __init__(
        self,
        storage: NewsStorage,
        vector_store: VectorStore,
        dedup_agent: Optional[DeduplicationAgent] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        stock_mapper: Optional[StockImpactMapper] = None,
        sentiment_agent: Optional[HybridSentimentClassifier] = None,
        supply_chain_mapper: Optional[SupplyChainImpactMapper] = None,
        sentiment_method: str = None
    ):
        config = get_config()
        sentiment_method = sentiment_method or config.sentiment_analysis.method

        self.storage = storage
        self.vector_store = vector_store
        
        self.dedup_agent = dedup_agent or DeduplicationAgent()
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.stock_mapper = stock_mapper or StockImpactMapper()
        
        self.sentiment_agent = sentiment_agent or HybridSentimentClassifier(
            method=sentiment_method,
            entity_extractor=self.entity_extractor
        )
        
        self.supply_chain_mapper = supply_chain_mapper or SupplyChainImpactMapper()
        
        self.query_processor = QueryProcessor(
            vector_store=self.vector_store,
            entity_extractor=self.entity_extractor,
            stock_mapper=self.stock_mapper
        )
        
        self.app = self._build_ingestion_graph()
        self.query_app = self._build_query_graph()
    
    def _build_ingestion_graph(self):
        """Construct the main news ingestion and analysis pipeline."""
        graph_builder = StateGraph(NewsIntelligenceState)
        
        # Nodes
        graph_builder.add_node("ingestion", self._ingestion_agent)
        graph_builder.add_node("deduplication", self._deduplication_agent)
        graph_builder.add_node("entity_extraction", self._entity_extraction_agent)
        graph_builder.add_node("impact_mapper", self._impact_mapper_agent)
        graph_builder.add_node("cross_impact", self._cross_impact_agent)
        graph_builder.add_node("sentiment_analysis", self._sentiment_analysis_agent)
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
        """Extracts key entities (companies, sectors, etc.) from the content."""
        article = state["current_article"]
        entities = self.entity_extractor.extract_entities(article)
        article.entities = entities
        
        stats = state.get("stats", {})
        stats["entities_extracted"] = {
            "companies": len(entities.get("Companies", [])),
            "sectors": len(entities.get("Sectors", [])),
            "regulators": len(entities.get("Regulators", [])),
            "people": len(entities.get("People", [])),
            "events": len(entities.get("Events", []))
        }
        
        return {
            "current_article": article,
            "entities": entities,
            "stats": stats
        }
    
    def _impact_mapper_agent(self, state: NewsIntelligenceState) -> dict:
        """Maps extracted entities to specific stock tickers."""
        article = state["current_article"]
        entities = state["entities"]
        
        impacts = self.stock_mapper.map_to_stocks(entities)
        impact_dicts = [imp.to_dict() for imp in impacts]
        article.impacted_stocks = impact_dicts
        
        stats = state.get("stats", {})
        stats["stocks_impacted"] = len(impacts)
        stats["impact_breakdown"] = {
            "direct": sum(1 for i in impacts if i.impact_type == "direct"),
            "sector": sum(1 for i in impacts if i.impact_type == "sector"),
            "regulatory": sum(1 for i in impacts if i.impact_type == "regulatory")
        }
        
        return {
            "current_article": article,
            "impacted_stocks": impact_dicts,
            "stats": stats
        }
    
    def _cross_impact_agent(self, state: NewsIntelligenceState) -> dict:
        """Analyzes supply chain relationships for upstream/downstream effects."""
        article = state["current_article"]
        entities = state["entities"]
        sectors = entities.get("Sectors", [])
        
        if not sectors:
            stats = state.get("stats", {})
            stats.update({
                "cross_impacts_found": 0,
                "upstream_dependencies": 0,
                "downstream_impacts": 0
            })
            return {
                "current_article": article,
                "cross_impacts": [],
                "stats": stats
            }
        
        cross_impacts = self.supply_chain_mapper.generate_cross_impact_insights(
            article, 
            entities
        )
        
        cross_impact_dicts = [impact.to_dict() for impact in cross_impacts]
        article.cross_impacts = cross_impact_dicts
        
        upstream_count = sum(1 for i in cross_impacts if i.relationship_type == "upstream_demand_shock")
        downstream_count = sum(1 for i in cross_impacts if i.relationship_type == "downstream_supply_impact")
        
        stats = state.get("stats", {})
        stats["cross_impacts_found"] = len(cross_impacts)
        stats["upstream_dependencies"] = upstream_count
        stats["downstream_impacts"] = downstream_count
        
        if cross_impacts:
            top_impact = cross_impacts[0]
            stats["top_cross_impact"] = {
                "target_sector": top_impact.target_sector,
                "impact_score": top_impact.impact_score,
                "relationship_type": top_impact.relationship_type
            }
        
        return {
            "current_article": article,
            "cross_impacts": cross_impact_dicts,
            "stats": stats
        }
    
    def _sentiment_analysis_agent(self, state: NewsIntelligenceState) -> dict:
        """Performs hybrid sentiment classification."""
        article = state["current_article"]
        
        self.sentiment_agent.analyze_and_attach(article)
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
            
            if "agreement_score" in sentiment_data.sentiment_breakdown:
                stats["sentiment_agreement"] = sentiment_data.sentiment_breakdown["agreement_score"]
        
        return {
            "current_article": article,
            "sentiment": sentiment_data.to_dict() if sentiment_data else None,
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
            "entities": None,
            "impacted_stocks": None,
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
            "entities": None,
            "impacted_stocks": None,
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
        sentiment_method_info = self.sentiment_agent.get_method_info()
        
        supply_chain_info = {
            "sectors_mapped": len(self.supply_chain_mapper.supply_chain_graph),
            "total_relationships": sum(
                len(sector_data.get("depends_on", [])) + len(sector_data.get("impacts", []))
                for sector_data in self.supply_chain_mapper.supply_chain_graph.values()
            )
        }
        
        return {
            "total_articles_stored": total_articles,
            "vector_store_count": vector_count,
            "dedup_threshold": {
                "bi_encoder": self.dedup_agent.bi_threshold,
                "cross_encoder": self.dedup_agent.cross_threshold
            },
            "sentiment_analysis": {
                "method": sentiment_method_info,
                "statistics": sentiment_stats
            },
            "supply_chain_analysis": supply_chain_info,
            "status": "operational"
        }