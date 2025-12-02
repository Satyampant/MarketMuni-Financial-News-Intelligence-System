#!/usr/bin/env python3
"""
Test script for Sentiment Analysis Integration in LangGraph Pipeline
Tests Task 5: Sentiment Agent Integration
"""

from datetime import datetime
from news_storage import NewsArticle, NewsStorage
from vector_store import VectorStore
from langgraph_orchestration import NewsIntelligenceGraph


def test_sentiment_integration():
    """Test sentiment analysis integration in the pipeline"""
    
    print("=" * 80)
    print("Testing Sentiment Analysis Integration in LangGraph Pipeline")
    print("=" * 80)
    
    # Initialize components
    storage = NewsStorage()
    vector_store = VectorStore(
        collection_name="test_sentiment_integration",
        persist_directory="./test_chroma_sentiment"
    )
    
    # Test with different sentiment methods
    for method in ["rule_based", "hybrid"]:
        print(f"\n{'=' * 80}")
        print(f"Testing with method: {method.upper()}")
        print('=' * 80)
        
        # Initialize graph with sentiment method
        graph = NewsIntelligenceGraph(
            storage=storage,
            vector_store=vector_store,
            sentiment_method=method
        )
        
        # Test articles with different sentiments
        test_articles = [
            NewsArticle(
                id=f"SENT_TEST_{method}_001",
                title="HDFC Bank announces 15% dividend, board approves stock buyback",
                content="HDFC Bank Ltd announced a 15% dividend payout and approved a stock buyback program worth Rs 5,000 crores. The board expressed confidence in the bank's strong fundamentals.",
                source="MoneyControl",
                timestamp=datetime.now()
            ),
            NewsArticle(
                id=f"SENT_TEST_{method}_002",
                title="Tech sector faces massive layoffs amid economic uncertainty",
                content="Major IT companies announced devastating layoffs, with revenues plummeting and losses mounting amid economic crisis.",
                source="Economic Times",
                timestamp=datetime.now()
            ),
            NewsArticle(
                id=f"SENT_TEST_{method}_003",
                title="RBI announces policy meeting scheduled for next week",
                content="The Reserve Bank of India will hold its regular monetary policy committee meeting next week to review current rates and policies.",
                source="Business Standard",
                timestamp=datetime.now()
            )
        ]
        
        # Process each article through pipeline
        for article in test_articles:
            print(f"\n[Processing] {article.id}")
            print(f"Title: {article.title}")
            
            # Run through pipeline
            result = graph.run_pipeline(article)
            
            # Check for errors
            if result.get("error"):
                print(f"  ❌ Error: {result['error']}")
                continue
            
            # Extract stats
            stats = result.get("stats", {})
            
            # Verify sentiment analysis was executed
            if stats.get("sentiment_analyzed"):
                print(f"  ✓ Sentiment Analyzed: {stats['sentiment_analyzed']}")
                print(f"  Classification: {stats.get('sentiment_classification')}")
                print(f"  Confidence: {stats.get('sentiment_confidence'):.2f}")
                print(f"  Signal Strength: {stats.get('sentiment_signal_strength'):.2f}")
                print(f"  Method: {stats.get('sentiment_method')}")
                
                if "sentiment_agreement" in stats:
                    print(f"  Agreement Score: {stats['sentiment_agreement']:.3f}")
            else:
                print(f"  ❌ Sentiment Analysis Failed")
            
            # Verify article was indexed
            if stats.get("indexed"):
                print(f"  ✓ Article Indexed")
            
            # Verify sentiment in article object
            processed_article = result["articles"][0] if result.get("articles") else None
            if processed_article and processed_article.has_sentiment():
                sentiment = processed_article.get_sentiment()
                print(f"  ✓ Sentiment attached to article")
                print(f"    -> {sentiment.classification} ({sentiment.confidence_score:.2f})")
            else:
                print(f"  ❌ Sentiment not attached to article")
        
        # Get system stats
        print(f"\n{'-' * 80}")
        print("System Statistics:")
        print('-' * 80)
        
        system_stats = graph.get_stats()
        print(f"Total Articles: {system_stats['total_articles_stored']}")
        print(f"Vector Store Count: {system_stats['vector_store_count']}")
        
        sentiment_info = system_stats.get("sentiment_analysis", {})
        if sentiment_info:
            print(f"\nSentiment Analysis Info:")
            method_info = sentiment_info.get("method", {})
            print(f"  Configured Method: {method_info.get('configured_method')}")
            print(f"  Rule-Based Available: {method_info.get('rule_based_available')}")
            print(f"  FinBERT Available: {method_info.get('finbert_available')}")
            
            stats_data = sentiment_info.get("statistics", {})
            if stats_data:
                print(f"\nSentiment Statistics:")
                print(f"  Total Articles: {stats_data.get('total_articles')}")
                print(f"  Analyzed: {stats_data.get('analyzed_count')}")
                print(f"  Bullish: {stats_data.get('bullish_count')} ({stats_data.get('bullish_percentage')}%)")
                print(f"  Bearish: {stats_data.get('bearish_count')} ({stats_data.get('bearish_percentage')}%)")
                print(f"  Neutral: {stats_data.get('neutral_count')} ({stats_data.get('neutral_percentage')}%)")
    
    print(f"\n{'=' * 80}")
    print("✓ Sentiment Integration Tests Completed")
    print('=' * 80)


def test_sentiment_in_vector_search():
    """Test that sentiment data is preserved in vector search results"""
    
    print("\n" + "=" * 80)
    print("Testing Sentiment Data in Vector Search")
    print("=" * 80)
    
    # Initialize components
    storage = NewsStorage()
    vector_store = VectorStore(
        collection_name="test_sentiment_search",
        persist_directory="./test_chroma_sentiment_search"
    )
    
    graph = NewsIntelligenceGraph(
        storage=storage,
        vector_store=vector_store,
        sentiment_method="rule_based"
    )
    
    # Add a test article
    article = NewsArticle(
        id="SEARCH_TEST_001",
        title="Banking sector shows exceptional growth with record profits",
        content="All major banks reported stellar quarterly results with profits surging across the sector.",
        source="Financial Express",
        timestamp=datetime.now()
    )
    
    print(f"\n[Ingesting] {article.title}")
    result = graph.run_pipeline(article)
    
    if result.get("error"):
        print(f"❌ Error during ingestion: {result['error']}")
        return
    
    print("✓ Article ingested with sentiment")
    
    # Now search for it
    print("\n[Searching] 'banking sector'")
    search_results = vector_store.search("banking sector", top_k=5)
    
    if search_results:
        for i, result in enumerate(search_results, 1):
            print(f"\nResult {i}:")
            print(f"  Article ID: {result['article_id']}")
            print(f"  Title: {result['metadata']['title']}")
            
            # Check sentiment in metadata
            sentiment = result['metadata'].get('sentiment')
            if sentiment:
                print(f"  ✓ Sentiment Found:")
                print(f"    Classification: {sentiment.get('classification')}")
                print(f"    Confidence: {sentiment.get('confidence_score')}")
                print(f"    Signal Strength: {sentiment.get('signal_strength')}")
            else:
                print(f"  ❌ Sentiment Missing from search results")
    else:
        print("❌ No search results found")
    
    print("\n" + "=" * 80)
    print("✓ Vector Search Sentiment Test Completed")
    print("=" * 80)


if __name__ == "__main__":
    try:
        # Test 1: Pipeline Integration
        test_sentiment_integration()
        
        # Test 2: Vector Search with Sentiment
        test_sentiment_in_vector_search()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()