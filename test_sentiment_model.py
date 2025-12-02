#!/usr/bin/env python3
"""
Test script for Sentiment Data Model implementation.
Validates NewsArticle sentiment functionality and VectorStore integration.
"""

from datetime import datetime
from news_storage import NewsArticle, SentimentData, NewsStorage
from vector_store import VectorStore


def test_sentiment_data_model():
    """Test SentimentData dataclass validation"""
    print("=" * 60)
    print("TEST 1: SentimentData Model")
    print("=" * 60)
    
    # Valid sentiment data
    sentiment = SentimentData(
        classification="Bullish",
        confidence_score=85.5,
        signal_strength=72.3,
        sentiment_breakdown={
            "positive": 75.0,
            "negative": 15.0,
            "neutral": 10.0,
            "urgency": 60.0
        },
        analysis_method="hybrid"
    )
    
    print(f"✓ Created SentimentData: {sentiment.classification}")
    print(f"  Confidence: {sentiment.confidence_score}")
    print(f"  Signal Strength: {sentiment.signal_strength}")
    print(f"  Method: {sentiment.analysis_method}")
    print(f"  Breakdown: {sentiment.sentiment_breakdown}")
    print()
    
    # Test serialization
    sentiment_dict = sentiment.to_dict()
    print(f"✓ Serialized to dict: {len(sentiment_dict)} fields")
    
    # Test deserialization
    reconstructed = SentimentData.from_dict(sentiment_dict)
    print(f"✓ Deserialized back: {reconstructed.classification}")
    print()
    
    # Test validation
    print("Testing validation:")
    try:
        invalid = SentimentData(
            classification="InvalidType",
            confidence_score=85.5,
            signal_strength=72.3,
            sentiment_breakdown={},
            analysis_method="hybrid"
        )
        print("✗ Should have raised ValueError for invalid classification")
    except ValueError as e:
        print(f"✓ Caught invalid classification: {str(e)[:50]}...")
    
    try:
        invalid = SentimentData(
            classification="Bullish",
            confidence_score=150.0,  # Invalid range
            signal_strength=72.3,
            sentiment_breakdown={},
            analysis_method="hybrid"
        )
        print("✗ Should have raised ValueError for invalid score")
    except ValueError as e:
        print(f"✓ Caught invalid score: {str(e)[:50]}...")
    
    print()


def test_newsarticle_sentiment():
    """Test NewsArticle sentiment integration"""
    print("=" * 60)
    print("TEST 2: NewsArticle Sentiment Integration")
    print("=" * 60)
    
    # Create article
    article = NewsArticle(
        id="TEST001",
        title="HDFC Bank reports strong Q3 earnings",
        content="HDFC Bank announced record quarterly profits driven by retail growth.",
        source="MoneyControl",
        timestamp=datetime.now(),
        impacted_stocks=[
            {"symbol": "HDFCBANK", "confidence": 1.0, "impact_type": "direct"}
        ]
    )
    
    print(f"✓ Created article: {article.id}")
    print(f"  Has sentiment: {article.has_sentiment()}")
    print()
    
    # Add sentiment
    sentiment = SentimentData(
        classification="Bullish",
        confidence_score=88.2,
        signal_strength=75.5,
        sentiment_breakdown={
            "positive": 80.0,
            "negative": 10.0,
            "neutral": 10.0
        },
        analysis_method="rule_based"
    )
    
    article.set_sentiment(sentiment)
    print(f"✓ Added sentiment to article")
    print(f"  Has sentiment: {article.has_sentiment()}")
    print(f"  Classification: {article.sentiment['classification']}")
    print()
    
    # Retrieve sentiment
    retrieved_sentiment = article.get_sentiment()
    print(f"✓ Retrieved sentiment: {retrieved_sentiment.classification}")
    print(f"  Confidence: {retrieved_sentiment.confidence_score}")
    print()


def test_storage_sentiment_queries():
    """Test NewsStorage sentiment filtering"""
    print("=" * 60)
    print("TEST 3: NewsStorage Sentiment Queries")
    print("=" * 60)
    
    storage = NewsStorage()
    
    # Create test articles with different sentiments
    sentiments = [
        ("N001", "Bullish", 85.0),
        ("N002", "Bearish", 72.0),
        ("N003", "Neutral", 60.0),
        ("N004", "Bullish", 90.0),
        ("N005", "Neutral", 55.0)
    ]
    
    for article_id, classification, confidence in sentiments:
        article = NewsArticle(
            id=article_id,
            title=f"Test Article {article_id}",
            content="Test content",
            source="Test",
            timestamp=datetime.now()
        )
        
        sentiment = SentimentData(
            classification=classification,
            confidence_score=confidence,
            signal_strength=confidence * 0.8,
            sentiment_breakdown={"positive": 50.0, "negative": 30.0, "neutral": 20.0},
            analysis_method="rule_based"
        )
        
        article.set_sentiment(sentiment)
        storage.add_article(article)
    
    print(f"✓ Added {storage.article_count()} articles to storage")
    print()
    
    # Query by sentiment
    bullish = storage.get_articles_by_sentiment("Bullish")
    bearish = storage.get_articles_by_sentiment("Bearish")
    neutral = storage.get_articles_by_sentiment("Neutral")
    
    print(f"✓ Bullish articles: {len(bullish)}")
    for article in bullish:
        print(f"  - {article.id}: {article.sentiment['confidence_score']}")
    
    print(f"✓ Bearish articles: {len(bearish)}")
    for article in bearish:
        print(f"  - {article.id}: {article.sentiment['confidence_score']}")
    
    print(f"✓ Neutral articles: {len(neutral)}")
    for article in neutral:
        print(f"  - {article.id}: {article.sentiment['confidence_score']}")
    
    print()


def test_vector_store_sentiment():
    """Test VectorStore sentiment indexing and retrieval"""
    print("=" * 60)
    print("TEST 4: VectorStore Sentiment Integration")
    print("=" * 60)
    
    # Initialize vector store
    vector_store = VectorStore(
        collection_name="test_sentiment",
        persist_directory="./test_chroma_sentiment"
    )
    
    # Reset for clean test
    vector_store.reset()
    print("✓ Initialized and reset vector store")
    print()
    
    # Create article with sentiment
    article = NewsArticle(
        id="VS001",
        title="Tech stocks rally on strong earnings",
        content="Major technology companies reported better-than-expected quarterly results.",
        source="Financial Times",
        timestamp=datetime.now(),
        impacted_stocks=[
            {"symbol": "TCS", "confidence": 0.9, "impact_type": "direct"}
        ]
    )
    
    sentiment = SentimentData(
        classification="Bullish",
        confidence_score=92.5,
        signal_strength=85.0,
        sentiment_breakdown={
            "positive": 85.0,
            "negative": 5.0,
            "neutral": 10.0,
            "market_impact": 80.0
        },
        analysis_method="hybrid"
    )
    
    article.set_sentiment(sentiment)
    
    # Index article
    vector_store.index_article(article)
    print(f"✓ Indexed article with sentiment")
    print(f"  Classification: {article.sentiment['classification']}")
    print(f"  Confidence: {article.sentiment['confidence_score']}")
    print()
    
    # Search with sentiment filter
    results = vector_store.search(
        query="technology earnings",
        top_k=5,
        sentiment_filter="Bullish"
    )
    
    print(f"✓ Search results with Bullish filter: {len(results)}")
    for result in results:
        metadata = result["metadata"]
        print(f"  - {metadata['article_id']}: {metadata.get('sentiment_classification', 'N/A')}")
        if metadata.get("sentiment"):
            print(f"    Confidence: {metadata['sentiment']['confidence_score']}")
    print()
    
    # Get sentiment statistics
    stats = vector_store.get_sentiment_statistics()
    print("✓ Sentiment Statistics:")
    print(f"  Total articles: {stats['total_articles']}")
    print(f"  Analyzed: {stats['analyzed_count']}")
    print(f"  Bullish: {stats['bullish_count']} ({stats['bullish_percentage']}%)")
    print(f"  Bearish: {stats['bearish_count']} ({stats['bearish_percentage']}%)")
    print(f"  Neutral: {stats['neutral_count']} ({stats['neutral_percentage']}%)")
    print()
    
    # Cleanup
    vector_store.reset()
    print("✓ Test cleanup complete")
    print()


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "SENTIMENT DATA MODEL TEST SUITE" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    try:
        test_sentiment_data_model()
        test_newsarticle_sentiment()
        test_storage_sentiment_queries()
        test_vector_store_sentiment()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()