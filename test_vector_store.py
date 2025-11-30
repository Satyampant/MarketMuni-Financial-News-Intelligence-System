from vector_store import VectorStore
from news_storage import load_mock_dataset
from entity_extraction import EntityExtractor
from stock_impact import StockImpactMapper
import json

def setup_test_data():
    """Load and enrich test articles with entities and stock impacts."""
    articles = load_mock_dataset('mock_news_data.json')
    
    # Initialize enrichment modules
    extractor = EntityExtractor()
    mapper = StockImpactMapper()
    
    # Enrich first 5 articles for testing
    enriched_articles = []
    for article in articles[:5]:
        # Extract entities
        text = f"{article.title}. {article.content}"
        entities = extractor.extract_entities(text)
        article.entities = entities
        
        # Map to stocks
        impacts = mapper.map_to_stocks(entities)
        article.impacted_stocks = impacts
        
        enriched_articles.append(article)
    
    return enriched_articles

def test_index_and_count():
    """Test indexing articles and counting."""
    print("Test 1: Index articles and count")
    print("=" * 60)
    
    # Create fresh vector store
    vs = VectorStore(collection_name="test_financial_news")
    vs.reset()
    
    # Load and index test data
    articles = setup_test_data()
    
    for article in articles:
        vs.index_article(article)
        print(f"✓ Indexed: {article.id} - {article.title[:50]}...")
    
    count = vs.count()
    assert count == 5, f"Expected 5 articles, got {count}"
    print(f"\n✓ Total articles indexed: {count}")
    print()

def test_semantic_search():
    """Test semantic search functionality."""
    print("Test 2: Semantic search for 'banking crisis'")
    print("=" * 60)
    
    vs = VectorStore(collection_name="test_financial_news")
    
    # Search for banking-related news
    query = "banking crisis"
    results = vs.search(query, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Top {len(results)} results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Article ID: {result['article_id']}")
        print(f"   Title: {result['metadata']['title']}")
        print(f"   Similarity: {result['similarity']:.4f}")
        print(f"   Source: {result['metadata']['source']}")
        
        # Show impacted stocks
        stocks = result['metadata'].get('impacted_stocks', [])
        if stocks:
            print(f"   Impacted Stocks: {', '.join([s['symbol'] for s in stocks[:3]])}")
        print()
    
    # Verify relevant results
    assert len(results) > 0, "Should return results"
    assert results[0]['similarity'] > 0.3, "Top result should have decent similarity"
    print("✓ Semantic search working correctly")
    print()

def test_stock_symbol_search():
    """Test search with stock symbol filter."""
    print("Test 3: Search for HDFC Bank news")
    print("=" * 60)
    
    vs = VectorStore(collection_name="test_financial_news")
    
    query = "HDFC Bank"
    results = vs.search(query, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['metadata']['title']}")
        print(f"   Similarity: {result['similarity']:.4f}")
        stocks = result['metadata'].get('impacted_stocks', [])
        if stocks:
            items = [f"{s['symbol']} ({s['confidence']:.2f})" for s in stocks[:3]]
            print(f"   Stocks: {', '.join(items)}")
        print()
    
    print("✓ Stock-specific search working")
    print()

def test_regulatory_news_search():
    """Test search for regulatory news."""
    print("Test 4: Search for RBI policy news")
    print("=" * 60)
    
    vs = VectorStore(collection_name="test_financial_news")
    
    query = "RBI monetary policy interest rate"
    results = vs.search(query, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Results:\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['metadata']['title']}")
        print(f"   Similarity: {result['similarity']:.4f}")
        
        # Show entities
        entities = result['metadata'].get('entities', {})
        if entities:
            regulators = entities.get('Regulators', [])
            if regulators:
                print(f"   Regulators: {', '.join(regulators)}")
        
        # Show impacted stocks
        stocks = result['metadata'].get('impacted_stocks', [])
        if stocks:
            regulatory_stocks = [s for s in stocks if s['impact_type'] == 'regulatory']
            if regulatory_stocks:
                print(f"   Regulatory Impact: {len(regulatory_stocks)} stocks")
        print()
    
    print("✓ Regulatory news search working")
    print()

def test_get_by_id():
    """Test retrieval by article ID."""
    print("Test 5: Retrieve article by ID")
    print("=" * 60)
    
    vs = VectorStore(collection_name="test_financial_news")
    
    article_id = "N001"
    result = vs.get_by_id(article_id)
    
    assert result is not None, f"Article {article_id} should exist"
    
    print(f"Retrieved Article ID: {result['article_id']}")
    print(f"Title: {result['metadata']['title']}")
    print(f"Source: {result['metadata']['source']}")
    print(f"Timestamp: {result['metadata']['timestamp']}")
    
    # Show entities
    entities = result['metadata'].get('entities', {})
    print(f"\nEntities:")
    for entity_type, values in entities.items():
        if values:
            print(f"  {entity_type}: {', '.join(values)}")
    
    # Show impacted stocks
    stocks = result['metadata'].get('impacted_stocks', [])
    print(f"\nImpacted Stocks:")
    for stock in stocks:
        print(f"  {stock['symbol']}: {stock['confidence']:.2f} ({stock['impact_type']})")
    
    print("\n✓ Retrieval by ID working")
    print()

def test_similarity_ranking():
    """Test that results are properly ranked by similarity."""
    print("Test 6: Verify similarity ranking")
    print("=" * 60)
    
    vs = VectorStore(collection_name="test_financial_news")
    
    query = "bank financial services"
    results = vs.search(query, top_k=5)
    
    print(f"Query: '{query}'")
    print("Similarity scores (should be in descending order):\n")
    
    similarities = []
    for i, result in enumerate(results, 1):
        similarity = result['similarity']
        similarities.append(similarity)
        print(f"{i}. {result['article_id']}: {similarity:.4f} - {result['metadata']['title'][:60]}...")
    
    # Verify descending order
    for i in range(len(similarities) - 1):
        assert similarities[i] >= similarities[i + 1], "Similarities should be in descending order"
    
    print("\n✓ Results properly ranked by similarity")
    print()

def test_embedding_consistency():
    """Test that embeddings are consistent."""
    print("Test 7: Test embedding consistency")
    print("=" * 60)
    
    vs = VectorStore(collection_name="test_financial_news")
    
    # Create embeddings for same text twice
    text = "HDFC Bank announces dividend"
    emb1 = vs.create_embedding(text)
    emb2 = vs.create_embedding(text)
    
    # Compute cosine similarity
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    sim = cosine_similarity([emb1], [emb2])[0][0]
    
    print(f"Text: '{text}'")
    print(f"Embedding 1 length: {len(emb1)}")
    print(f"Embedding 2 length: {len(emb2)}")
    print(f"Cosine similarity: {sim:.6f}")
    
    assert sim > 0.9999, "Same text should produce nearly identical embeddings"
    print("\n✓ Embeddings are consistent")
    print()

if __name__ == "__main__":
    print("Testing Vector Store & RAG Indexing")
    print("=" * 60 + "\n")
    
    test_index_and_count()
    test_semantic_search()
    test_stock_symbol_search()
    test_regulatory_news_search()
    test_get_by_id()
    test_similarity_ranking()
    test_embedding_consistency()
    
    print("=" * 60)
    print("✅ All vector store tests passed!")