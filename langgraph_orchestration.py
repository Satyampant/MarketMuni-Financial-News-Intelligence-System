from typing import TypedDict, List, Annotated, Optional
import operator
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from news_storage import NewsArticle, NewsStorage
from deduplication import DeduplicationAgent
from entity_extraction import EntityExtractor
from stock_impact import StockImpactMapper
from vector_store import VectorStore
# Moved import to top level
from query_processor import QueryProcessor

class NewsIntelligenceState(TypedDict):
    # ... (Same as original) ...
    articles: Annotated[List[NewsArticle], operator.add]
    current_article: Optional[NewsArticle]
    duplicates: Annotated[List[str], operator.add]
    entities: Optional[dict]
    impacted_stocks: Optional[List[dict]]
    query_text: Optional[str]
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
        stock_mapper: Optional[StockImpactMapper] = None
    ):
        self.storage = storage
        self.vector_store = vector_store
        
        self.dedup_agent = dedup_agent or DeduplicationAgent()
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.stock_mapper = stock_mapper or StockImpactMapper()

        # OPTIMIZATION 1: Initialize QueryProcessor once to avoid re-reading JSONs
        self.query_processor = QueryProcessor(
            vector_store=self.vector_store,
            entity_extractor=self.entity_extractor,
            stock_mapper=self.stock_mapper
        )
        
        # Build graphs ONCE during initialization
        self.app = self._build_ingestion_graph()
        self.query_app = self._build_query_graph() # OPTIMIZATION 2
    
    def _build_ingestion_graph(self):
        """Construct the main ingestion pipeline"""
        graph_builder = StateGraph(NewsIntelligenceState)
        
        graph_builder.add_node("ingestion", self._ingestion_agent)
        graph_builder.add_node("deduplication", self._deduplication_agent)
        graph_builder.add_node("entity_extraction", self._entity_extraction_agent)
        graph_builder.add_node("impact_mapper", self._impact_mapper_agent)
        graph_builder.add_node("indexing", self._indexing_agent)
        
        graph_builder.add_edge(START, "ingestion")
        graph_builder.add_edge("ingestion", "deduplication")
        # Potential for conditional edge here: if perfect duplicate, go to END
        graph_builder.add_edge("deduplication", "entity_extraction")
        graph_builder.add_edge("entity_extraction", "impact_mapper")
        graph_builder.add_edge("impact_mapper", "indexing")
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
        """
        Agent 1: News Ingestion
        Validates and prepares article for processing
        """
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
        """
        Agent 2: Deduplication
        Identifies and consolidates duplicate articles using semantic similarity
        """
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
        """
        Agent 3: Entity Extraction
        Extracts companies, sectors, regulators, people, and events using NER
        """
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
        """
        Agent 4: Stock Impact Analysis
        Maps extracted entities to impacted stocks with confidence scores
        """
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
    
    def _indexing_agent(self, state: NewsIntelligenceState) -> dict:
        """
        Agent 5: Storage & Indexing
        Stores article in NewsStorage and indexes in VectorStore
        """
        article = state["current_article"]
        
        # Store in NewsStorage
        self.storage.add_article(article)
        
        # Index in VectorStore
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
        """
        Agent 6: Query Processing
        Retrieves relevant articles based on natural language query
        """
        query_text = state.get("query_text")
        
        if not query_text:
            return {"error": "No query text provided"}
        
        # Import here to avoid circular dependency
        from query_processor import QueryProcessor
        
        # Initialize query processor
        query_processor = QueryProcessor(
            vector_store=self.vector_store,
            entity_extractor=self.entity_extractor,
            stock_mapper=self.stock_mapper
        )
        
        # Process query
        results = query_processor.process_query(query_text, top_k=10)
        
        stats = {
            "query_time": datetime.now().isoformat(),
            "results_count": len(results),
            "query": query_text
        }
        
        return {
            "query_results": results,
            "stats": stats
        }
    
    def run_pipeline(self, article: NewsArticle) -> dict:
        """
        Execute full ingestion pipeline for a single article.
        
        Flow: ingestion → deduplication → entity_extraction → 
              impact_mapper → indexing
        
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
            "query_text": None,
            "query_results": [],
            "error": None,
            "stats": {}
        }
        
        # Execute pipeline (follows ingestion path)
        final_state = self.app.invoke(initial_state)
        
        return final_state
    
    def run_query(self, query_text: str) -> dict:
        
        initial_state = {
            "articles": [],
            "current_article": None,
            "duplicates": [],
            "entities": None,
            "impacted_stocks": None,
            "query_text": query_text,
            "query_results": [],
            "error": None,
            "stats": {}
        }
        
        final_state = self.query_app.invoke(initial_state)
        
        return final_state
    
    def get_stats(self) -> dict:
        """
        Get system statistics.
        
        Returns:
            Dictionary with article count, deduplication accuracy, etc.
        """
        total_articles = self.storage.article_count()
        vector_count = self.vector_store.count()
        
        return {
            "total_articles_stored": total_articles,
            "vector_store_count": vector_count,
            "dedup_threshold": {
                "bi_encoder": self.dedup_agent.bi_threshold,
                "cross_encoder": self.dedup_agent.cross_threshold
            },
            "status": "operational"
        }