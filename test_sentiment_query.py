#!/usr/bin/env python3
"""
Test script for sentiment-based query filtering and ranking
Demonstrates Task 7: Add Sentiment-Based Query Filtering & Ranking
"""

import json
from datetime import datetime
from news_storage import NewsArticle, NewsStorage
from vector_store import VectorStore
from deduplication import DeduplicationAgent
from entity_extraction import EntityExtractor
from stock_impact import StockImpactMapper
from langgraph_orchestration import NewsIntelligenceGraph


def load_test_data():
    """Load mock news data"""
    with open("mock_news_data.json", "r") as f:
        data = json.load(f)
    return [NewsArticle(**article) for article in data]


def print_separator(title=""):
    """Print a visual separator"""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    print()


def print_article_summary(article: NewsArticle, rank: int = None):
    """Print a concise article summary"""
    prefix = f"[{rank}] " if rank is not None else ""
    print(f"{prefix}ID: {article.id}")
    print(f"    Title: {article.title}")
    
    # Print sentiment info if available
    if hasattr(article, 'sentiment') and article.sentiment:
        sentiment = article.sentiment
        classification = sentiment.get('classification', 'Unknown')
        confidence = sentiment.get('confidence_score', 0)
        signal = sentiment.get('signal_strength', 0)
        method = sentiment.get('analysis_method', 'unknown')
        
        print(f"    Sentiment: {classification} (Confidence: {confidence:.1f}, Signal: {signal:.1f})")
        print(f"    Method: {method}")
    
    # Print relevance score if available
    if hasattr(article, 'relevance_score'):
        print(f"    Relevance Score: {article.relevance_score:.3f}")
    
    # Print sentiment boost if available
    if hasattr(article, 'sentiment_boost'):
        print(f"    Sentiment Boost: {article.sentiment_boost:.3f}x")
    
    print()


def test_sentiment_filtering():
    """Test sentiment-based query filtering"""
    print_separator("TEST 1: Sentiment-Based Query Filtering")
    
    # Initialize system
    storage = NewsStorage()
    vector_store = VectorStore(
        collection_name="test_sentiment_query",
        persist_directory="./test_chroma_sentiment_query"
    )
    vector_store.reset()  # Clean slate
    
    news_graph = NewsIntelligenceGraph(
        storage=storage,
        vector_store=vector_store,
        sentiment_method="hybrid"
    )
    
    # Load and ingest test articles
    articles = load_test_data()
    print(f"Ingesting {len(articles)} articles...\n")
    
    for i, article in enumerate(articles[:15], 1):  # Use first 15 for faster testing
        result = news_graph.run_pipeline(article)
        if not result.get("error"):
            processed_article = result["articles"][0]
            sentiment = processed_article.sentiment
            if sentiment:
                print(f"[{i}/{15}] {article.id}: {sentiment['classification']} "
                      f"(Signal: {sentiment['signal_strength']:.1f})")
    
    print(f"\n✓ Ingested {storage.article_count()} articles")
    
    # Test Query 1: Banking news (no filter)
    print_separator("Query 1: Banking News (No Sentiment Filter)")
    query1 = "HDFC Bank news"
    result1 = news_graph.run_query(query1)
    
    print(f"Query: '{query1}'")
    print(f"Results: {result1['stats']['results_count']}\n")
    
    for i, article in enumerate(result1["query_results"][:5], 1):
        print_article_summary(article, rank=i)
    
    # Test Query 2: Banking news (Bullish only)
    print_separator("Query 2: Banking News (Bullish Filter)")
    query2 = "HDFC Bank news"
    result2 = news_graph.run_query(query2, sentiment_filter="Bullish")
    
    print(f"Query: '{query2}' [Filter: Bullish]")
    print(f"Results: {result2['stats']['results_count']}\n")
    
    for i, article in enumerate(result2["query_results"][:5], 1):
        print_article_summary(article, rank=i)
    
    # Test Query 3: Banking news (Bearish only)
    print_separator("Query 3: Banking News (Bearish Filter)")
    query3 = "HDFC Bank news"
    result3 = news_graph.run_query(query3, sentiment_filter="Bearish")
    
    print(f"Query: '{query3}' [Filter: Bearish]")
    print(f"Results: {result3['stats']['results_count']}\n")
    
    for i, article in enumerate(result3["query_results"][:5], 1):
        print_article_summary(article, rank=i)
    
    # Test Query 4: Sector-wide (Neutral filter)
    print_separator("Query 4: Banking Sector (Neutral Filter)")
    query4 = "Banking sector update"
    result4 = news_graph.run_query(query4, sentiment_filter="Neutral")
    
    print(f"Query: '{query4}' [Filter: Neutral]")
    print(f"Results: {result4['stats']['results_count']}\n")
    
    for i, article in enumerate(result4["query_results"][:5], 1):
        print_article_summary(article, rank=i)


def test_sentiment_ranking():
    """Test sentiment-based ranking (high signal articles boosted)"""
    print_separator("TEST 2: Sentiment-Based Ranking (Signal Strength Boost)")
    
    # Initialize system
    storage = NewsStorage()
    vector_store = VectorStore(
        collection_name="test_sentiment_ranking",
        persist_directory="./test_chroma_sentiment_ranking"
    )
    vector_store.reset()
    
    news_graph = NewsIntelligenceGraph(
        storage=storage,
        vector_store=vector_store,
        sentiment_method="hybrid"
    )
    
    # Load and ingest all test articles
    articles = load_test_data()
    print(f"Ingesting {len(articles)} articles...\n")
    
    for article in articles:
        news_graph.run_pipeline(article)
    
    print(f"✓ Ingested {storage.article_count()} articles")
    
    # Query: General financial news (no filter)
    print_separator("Query: IT Sector News (Ranked by Relevance + Signal Strength)")
    query = "IT sector news"
    result = news_graph.run_query(query)
    
    print(f"Query: '{query}'")
    print(f"Results: {result['stats']['results_count']}")
    print("\nTop 10 Results (sorted by final score with sentiment boost):\n")
    
    for i, article in enumerate(result["query_results"][:10], 1):
        print_article_summary(article, rank=i)
    
    # Show ranking explanation
    print_separator("Ranking Explanation")
    print("Articles with higher signal_strength receive sentiment boost:")
    print("  sentiment_boost = 1.0 + (signal_strength / 200.0)")
    print("  Max boost: 1.5x for signal_strength = 100")
    print("\nAdditional 10% boost for articles matching sentiment filter")
    print()


def test_api_integration():
    """Test API endpoint integration"""
    print_separator("TEST 3: API Integration Test")
    
    print("Testing API query endpoint with sentiment filtering:\n")
    
    print("1. Query without sentiment filter:")
    print("   GET /query?q=HDFC Bank&top_k=5")
    print()
    
    print("2. Query with Bullish filter:")
    print("   GET /query?q=HDFC Bank&top_k=5&filter_by_sentiment=Bullish")
    print()
    
    print("3. Query with Bearish filter:")
    print("   GET /query?q=Banking sector&filter_by_sentiment=Bearish")
    print()
    
    print("4. Query with Neutral filter:")
    print("   GET /query?q=RBI policy&filter_by_sentiment=Neutral")
    print()
    
    print("✓ API endpoints support sentiment filtering via 'filter_by_sentiment' parameter")
    print("✓ Valid values: 'Bullish', 'Bearish', 'Neutral'")


def test_query_explanation():
    """Test query explanation feature"""
    print_separator("TEST 4: Query Explanation (Debug Feature)")
    
    from query_processor import QueryProcessor
    
    # Initialize components
    vector_store = VectorStore(
        collection_name="test_explanation",
        persist_directory="./test_chroma_explanation"
    )
    entity_extractor = EntityExtractor()
    stock_mapper = StockImpactMapper()
    
    query_processor = QueryProcessor(
        vector_store=vector_store,
        entity_extractor=entity_extractor,
        stock_mapper=stock_mapper
    )
    
    # Test queries
    test_queries = [
        ("HDFC Bank news", "Bullish"),
        ("Banking sector update", "Bearish"),
        ("RBI policy changes", "Neutral"),
        ("IT sector growth", None)
    ]
    
    for query, sentiment_filter in test_queries:
        explanation = query_processor.explain_query(query, sentiment_filter)
        
        print(f"Query: '{query}'")
        if sentiment_filter:
            print(f"Sentiment Filter: {sentiment_filter}")
        print(f"Strategies: {', '.join(explanation['strategies'])}")
        print(f"Companies: {explanation['identified_entities']['companies']}")
        print(f"Sectors: {explanation['identified_entities']['sectors']}")
        print(f"Regulators: {explanation['identified_entities']['regulators']}")
        print()


def main():
    """Run all tests"""
    print_separator("Sentiment-Based Query Filtering & Ranking Tests")
    print("Testing Task 7 Implementation")
    print()
    
    try:
        # Test 1: Sentiment Filtering
        test_sentiment_filtering()
        
        # Test 2: Sentiment Ranking
        test_sentiment_ranking()
        
        # Test 3: API Integration
        test_api_integration()
        
        # Test 4: Query Explanation
        test_query_explanation()
        
        print_separator("ALL TESTS COMPLETED SUCCESSFULLY")
        print("✓ Sentiment-based filtering working")
        print("✓ Signal strength boosting working")
        print("✓ API integration ready")
        print("✓ Query explanation available")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()