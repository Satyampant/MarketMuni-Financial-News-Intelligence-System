from typing import TypedDict, List, Annotated, Optional, Dict, Any
import operator
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from news_storage import NewsArticle, NewsStorage
from deduplication import DeduplicationAgent
from entity_extraction import EntityExtractor
from stock_impact import StockImpactMapper
from vector_store import VectorStore
from query_processor import QueryProcessor
from sentiment_hybrid import HybridSentimentClassifier
from supply_chain_mapper import SupplyChainImpactMapper


class NewsIntelligenceState(TypedDict):
    """Enhanced state with sentiment analysis and cross-impact support"""
    articles: Annotated[List[NewsArticle], operator.add]
    current_article: Optional[NewsArticle]
    duplicates: Annotated[List[str], operator.add]
    entities: Optional[dict]
    impacted_stocks: Optional[List[dict]]
    cross_impacts: Optional[List[Dict]]  # NEW: Supply chain cross-impacts
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
        sentiment_method: str = "hybrid"
    ):
        self.storage = storage
        self.vector_store = vector_store
        
        self.dedup_agent = dedup_agent or DeduplicationAgent()
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.stock_mapper = stock_mapper or StockImpactMapper()
        
        # Initialize Sentiment Agent
        self.sentiment_agent = sentiment_agent or HybridSentimentClassifier(
            method=sentiment_method,
            entity_extractor=self.entity_extractor
        )
        
        # Initialize Supply Chain Impact Mapper
        self.supply_chain_mapper = supply_chain_mapper or SupplyChainImpactMapper()
        
        # Initialize QueryProcessor once
        self.query_processor = QueryProcessor(
            vector_store=self.vector_store,
            entity_extractor=self.entity_extractor,
            stock_mapper=self.stock_mapper
        )
        
        # Build graphs
        self.app = self._build_ingestion_graph()
        self.query_app = self._build_query_graph()
    
    def _build_ingestion_graph(self):
        """Construct the main ingestion pipeline with cross-impact analysis"""
        graph_builder = StateGraph(NewsIntelligenceState)
        
        # Add all agent nodes
        graph_builder.add_node("ingestion", self._ingestion_agent)
        graph_builder.add_node("deduplication", self._deduplication_agent)
        graph_builder.add_node("entity_extraction", self._entity_extraction_agent)
        graph_builder.add_node("impact_mapper", self._impact_mapper_agent)
        graph_builder.add_node("cross_impact", self._cross_impact_agent)  # NEW
        graph_builder.add_node("sentiment_analysis", self._sentiment_analysis_agent)
        graph_builder.add_node("indexing", self._indexing_agent)
        
        # Define pipeline flow with cross-impact analysis
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
        """Construct the query-specific pipeline"""
        query_graph = StateGraph(NewsIntelligenceState)
        query_graph.add_node("query", self._query_agent)
        query_graph.add_edge(START, "query")
        query_graph.add_edge("query", END)
        return query_graph.compile()
    
    def _ingestion_agent(self, state: NewsIntelligenceState) -> dict:
        """Agent 1: News Ingestion - Validates and prepares article"""
        article = state.get("current_article")
        
        if not article:
            return {"error": "No article provided for ingestion"}
        
        # Validate required fields
        if not all([article.id, article.title, article.content, article.source]):
            return {"error": "Article missing required fields"}
        
        # Ensure timestamp is datetime object
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
        """Agent 2: Deduplication - Identifies and consolidates duplicates"""
        article = state["current_article"]
        existing_articles = self.storage.get_all_articles()
        
        # Find duplicates
        duplicate_ids = self.dedup_agent.find_duplicates(article, existing_articles)
        
        stats = state.get("stats", {})
        stats["duplicates_found"] = len(duplicate_ids)
        stats["is_duplicate"] = len(duplicate_ids) > 0
        
        if duplicate_ids:
            # Consolidate with existing duplicates
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
        """Agent 3: Entity Extraction - Extracts companies, sectors, regulators, etc."""
        article = state["current_article"]
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(article)
        
        # Attach to article
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
        """Agent 4: Stock Impact Mapper - Maps entities to impacted stocks"""
        article = state["current_article"]
        entities = state["entities"]
        
        # Map to stocks
        impacts = self.stock_mapper.map_to_stocks(entities)
        
        # Convert to dict format and attach to article
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
        """
        Agent 4.5: Cross-Impact Analysis
        Analyzes supply chain relationships and cross-sectoral effects
        """
        article = state["current_article"]
        entities = state["entities"]
        
        # Check if article has sectors to analyze
        sectors = entities.get("Sectors", [])
        
        if not sectors:
            # No sectors found, skip cross-impact analysis
            stats = state.get("stats", {})
            stats["cross_impacts_found"] = 0
            stats["upstream_dependencies"] = 0
            stats["downstream_impacts"] = 0
            
            return {
                "current_article": article,
                "cross_impacts": [],
                "stats": stats
            }
        
        # Generate cross-impact insights
        cross_impacts = self.supply_chain_mapper.generate_cross_impact_insights(
            article, 
            entities
        )
        
        # Convert to dict format and attach to article
        cross_impact_dicts = [impact.to_dict() for impact in cross_impacts]
        article.cross_impacts = cross_impact_dicts
        
        # Count upstream vs downstream relationships
        upstream_count = sum(
            1 for impact in cross_impacts 
            if impact.relationship_type == "upstream_demand_shock"
        )
        downstream_count = sum(
            1 for impact in cross_impacts 
            if impact.relationship_type == "downstream_supply_impact"
        )
        
        # Update stats
        stats = state.get("stats", {})
        stats["cross_impacts_found"] = len(cross_impacts)
        stats["upstream_dependencies"] = upstream_count
        stats["downstream_impacts"] = downstream_count
        
        if cross_impacts:
            # Add top impact details for debugging
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
        """
        Agent 5: Sentiment Analysis - Analyzes article sentiment
        
        Performs sentiment classification using hybrid approach
        Attaches sentiment data to article for downstream processing
        """
        article = state["current_article"]
        
        # Analyze sentiment and attach to article using the classifier's method
        self.sentiment_agent.analyze_and_attach(article)
        
        # Get the attached sentiment for stats
        sentiment_data = article.get_sentiment()
        
        # Update stats
        stats = state.get("stats", {})
        stats["sentiment_analyzed"] = True
        
        if sentiment_data:
            stats["sentiment_classification"] = sentiment_data.classification
            stats["sentiment_confidence"] = sentiment_data.confidence_score
            stats["sentiment_signal_strength"] = sentiment_data.signal_strength
            stats["sentiment_method"] = sentiment_data.analysis_method
            
            # Add agreement score if available in breakdown (hybrid mode)
            if "agreement_score" in sentiment_data.sentiment_breakdown:
                stats["sentiment_agreement"] = sentiment_data.sentiment_breakdown["agreement_score"]
        
        return {
            "current_article": article,
            "sentiment": sentiment_data.to_dict() if sentiment_data else None,
            "stats": stats
        }
    
    def _indexing_agent(self, state: NewsIntelligenceState) -> dict:
        """Agent 6: Storage & Indexing - Stores article and indexes in vector store"""
        article = state["current_article"]
        
        # Store in NewsStorage
        self.storage.add_article(article)
        
        # Index in VectorStore (now includes sentiment and cross-impact data)
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
        """Agent 7: Query Processing - Retrieves relevant articles with sentiment filtering"""
        query_text = state.get("query_text")
        sentiment_filter = state.get("sentiment_filter")

        if not query_text:
            return {"error": "No query text provided"}
        
        # Validate sentiment filter if provided
        if sentiment_filter:
            valid_sentiments = ["Bullish", "Bearish", "Neutral"]
            if sentiment_filter not in valid_sentiments:
                return {"error": f"Invalid sentiment filter: {sentiment_filter}. Must be one of {valid_sentiments}"}
        
        # Process query with sentiment filter
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
        """
        Execute full ingestion pipeline for a single article.
        
        Flow: ingestion → deduplication → entity_extraction → 
              impact_mapper → cross_impact → sentiment_analysis → indexing
        
        Args:
            article: NewsArticle to process
            
        Returns:
            Dictionary with final state and processing stats
        """
        initial_state = {
            "articles": [],
            "current_article": article,
            "duplicates": [],
            "entities": None,
            "impacted_stocks": None,
            "cross_impacts": None,  # NEW
            "sentiment": None,
            "query_text": None,
            "sentiment_filter": None,
            "query_results": [],
            "error": None,
            "stats": {}
        }
        
        # Execute pipeline
        final_state = self.app.invoke(initial_state)
        
        return final_state
    
    def run_query(self, query_text: str, sentiment_filter: Optional[str] = None) -> dict:
        """
        Execute query pipeline with optional sentiment filtering
        
        Args:
            query_text: Natural language query
            sentiment_filter: Optional "Bullish", "Bearish", or "Neutral" filter
            
        Returns:
            Dictionary with query results and stats
        """
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
        
        final_state = self.query_app.invoke(initial_state)
        
        return final_state
    
    def get_stats(self) -> dict:
        """Get system statistics including sentiment analysis and cross-impact info"""
        total_articles = self.storage.article_count()
        vector_count = self.vector_store.count()
        
        # Get sentiment statistics from vector store
        sentiment_stats = self.vector_store.get_sentiment_statistics()
        
        # Get sentiment agent info
        sentiment_method_info = self.sentiment_agent.get_method_info()
        
        # Get supply chain mapper info
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