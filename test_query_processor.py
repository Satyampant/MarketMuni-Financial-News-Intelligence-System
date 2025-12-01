from query_processor import QueryProcessor, QueryStrategy
from vector_store import VectorStore
from entity_extraction import EntityExtractor
from stock_impact import StockImpactMapper
from news_storage import load_mock_dataset
from deduplication import DeduplicationAgent


def setup_test_system():
    """Setup complete system with indexed data"""
    print("Setting up test system...")
    
    # Load articles
    articles = load_mock_dataset('mock_news_data.json')
    
    # Initialize components
    extractor = EntityExtractor()
    mapper = StockImpactMapper()
    dedup_agent = DeduplicationAgent()
    
    # Use shared embedding model
    shared_model = dedup_agent.embedding_model
    vector_store = VectorStore(
        collection_name="test_query_processor",
        embedding_model=shared_model
    )
    vector_store.reset()
    
    # Enrich and index articles
    for article in articles:
        text = f"{article.title}. {article.content}"
        entities = extractor.extract_entities(text)
        article.entities = entities
        
        impacts = mapper.map_to_stocks(entities)
        article.impacted_stocks = impacts
        
        vector_store.index_article(article)
    
    # Create query processor
    query_processor = QueryProcessor(
        vector_store=vector_store,
        entity_extractor=extractor,
        stock_mapper=mapper
    )
    
    print(f"✓ Indexed {vector_store.count()} articles")
    return query_processor


def test_query_expansion():
    """Test query expansion functionality with multi-query approach"""
    print("\n" + "=" * 60)
    print("Test 1: Query Expansion (Multi-Query Approach)")
    print("=" * 60)
    
    qp = setup_test_system()
    
    # Test 1: Company query
    print("\n[Query 1] HDFC Bank")
    context = qp.expand_query("HDFC Bank")
    print(f"  Original: {context.original_query}")
    print(f"  Primary Query: {context.primary_query}")
    print(f"  Context Queries: {context.context_queries}")
    print(f"  Companies: {context.companies}")
    print(f"  Sectors: {context.sectors}")
    print(f"  Stock Symbols: {context.stock_symbols}")
    print(f"  Strategies: {[s.value for s in context.strategies]}")
    
    assert context.primary_query == "HDFC Bank"
    assert "HDFC Bank" in context.companies
    assert "Banking" in context.sectors or any("Banking" in q for q in context.context_queries)
    assert "HDFCBANK" in context.stock_symbols
    assert QueryStrategy.DIRECT_MENTION in context.strategies
    print("✓ Company query expansion working")
    
    # Test 2: Sector query
    print("\n[Query 2] Banking sector")
    context = qp.expand_query("Banking sector")
    print(f"  Original: {context.original_query}")
    print(f"  Primary Query: {context.primary_query}")
    print(f"  Context Queries: {context.context_queries[:3]}")
    print(f"  Sectors: {context.sectors}")
    print(f"  Companies: {context.companies[:3]}...")
    print(f"  Stock Symbols: {context.stock_symbols[:3]}...")
    print(f"  Strategies: {[s.value for s in context.strategies]}")
    
    assert context.primary_query == "Banking sector"
    assert "Banking" in context.sectors
    assert QueryStrategy.SECTOR_WIDE in context.strategies
    assert len(context.context_queries) > 0, "Should have context queries for sector expansion"
    assert len(context.companies) > 0
    print("✓ Sector query expansion working")
    
    # Test 3: Regulator query
    print("\n[Query 3] RBI policy")
    context = qp.expand_query("RBI policy")
    print(f"  Original: {context.original_query}")
    print(f"  Primary Query: {context.primary_query}")
    print(f"  Context Queries: {context.context_queries}")
    print(f"  Regulators: {context.regulators}")
    print(f"  Strategies: {[s.value for s in context.strategies]}")
    
    assert context.primary_query == "RBI policy"
    assert "RBI" in context.regulators
    assert QueryStrategy.REGULATOR_FILTER in context.strategies
    assert len(context.context_queries) > 0, "Should have context queries for regulator"
    print("✓ Regulator query expansion working")


def test_direct_mention_query():
    """Test: 'HDFC Bank news' should return N1 + sector news"""
    print("\n" + "=" * 60)
    print("Test 2: Direct Mention Query - 'HDFC Bank news'")
    print("=" * 60)
    print("Expected: N001 (direct) + N004 (sector-wide) + N002 (RBI/Banking)")
    
    qp = setup_test_system()
    results = qp.process_query("HDFC Bank news", top_k=5)
    
    print(f"\nFound {len(results)} results:")
    for i, article in enumerate(results, 1):
        score = getattr(article, 'relevance_score', 0)
        print(f"{i}. [{article.id}] {article.title}")
        print(f"   Score: {score:.3f}")
        companies = article.entities.get("Companies", [])
        sectors = article.entities.get("Sectors", [])
        print(f"   Companies: {companies}")
        print(f"   Sectors: {sectors}")
    
    # Verify N001 (HDFC Bank direct mention) is in results
    result_ids = [a.id for a in results]
    assert "N001" in result_ids, "N001 (HDFC Bank dividend) should be in results"
    
    # Verify banking sector articles are included
    banking_articles = [a for a in results if "Banking" in a.entities.get("Sectors", [])]
    assert len(banking_articles) > 0, "Should include Banking sector articles"
    
    print("\n✓ Direct mention query working correctly")


def test_sector_wide_query():
    """Test: 'Banking sector update' should return N1, N2, N3, N4"""
    print("\n" + "=" * 60)
    print("Test 3: Sector-Wide Query - 'Banking sector update'")
    print("=" * 60)
    print("Expected: All banking-related articles (N001, N002, N003, N004, N011)")
    
    qp = setup_test_system()
    results = qp.process_query("Banking sector update", top_k=10)
    
    print(f"\nFound {len(results)} results:")
    for i, article in enumerate(results, 1):
        score = getattr(article, 'relevance_score', 0)
        print(f"{i}. [{article.id}] {article.title}")
        print(f"   Score: {score:.3f}")
        sectors = article.entities.get("Sectors", [])
        companies = article.entities.get("Companies", [])
        print(f"   Sectors: {sectors}")
        print(f"   Companies: {companies}")
    
    result_ids = [a.id for a in results]
    
    # Verify key banking articles are present
    expected_ids = ["N001", "N002", "N003", "N004"]
    found_expected = [eid for eid in expected_ids if eid in result_ids]
    
    print(f"\n✓ Found {len(found_expected)}/{len(expected_ids)} expected articles")
    assert len(found_expected) >= 3, "Should find at least 3 core banking articles"
    print("✓ Sector-wide query working correctly")


def test_regulator_filter_query():
    """Test: 'RBI policy changes' should return N2 primarily"""
    print("\n" + "=" * 60)
    print("Test 4: Regulator Filter Query - 'RBI policy changes'")
    print("=" * 60)
    print("Expected: N002 (RBI rate hike) as top result")
    
    qp = setup_test_system()
    results = qp.process_query("RBI policy changes", top_k=5)
    
    print(f"\nFound {len(results)} results:")
    for i, article in enumerate(results, 1):
        score = getattr(article, 'relevance_score', 0)
        print(f"{i}. [{article.id}] {article.title}")
        print(f"   Score: {score:.3f}")
        regulators = article.entities.get("Regulators", [])
        print(f"   Regulators: {regulators}")
    
    # Verify N002 is in top results
    result_ids = [a.id for a in results[:3]]
    assert "N002" in result_ids, "N002 (RBI rate hike) should be in top 3"
    
    # Verify RBI is mentioned in top result
    top_regulators = results[0].entities.get("Regulators", [])
    assert "RBI" in top_regulators, "Top result should mention RBI"
    
    print("\n✓ Regulator filter query working correctly")


def test_semantic_theme_query():
    """Test: 'Interest rate impact' should return related articles"""
    print("\n" + "=" * 60)
    print("Test 5: Semantic Theme Query - 'Interest rate impact'")
    print("=" * 60)
    print("Expected: N002 (RBI rate hike) + semantically similar articles")
    
    qp = setup_test_system()
    results = qp.process_query("interest rate impact financial markets", top_k=5)
    
    print(f"\nFound {len(results)} results:")
    for i, article in enumerate(results, 1):
        score = getattr(article, 'relevance_score', 0)
        print(f"{i}. [{article.id}] {article.title}")
        print(f"   Score: {score:.3f}")
        events = article.entities.get("Events", [])
        if events:
            print(f"   Events: {events}")
    
    # Verify we get rate-related articles
    result_ids = [a.id for a in results]
    
    # N002 should be highly ranked
    assert "N002" in result_ids[:3], "N002 should be in top 3 for rate impact query"
    
    print("\n✓ Semantic theme query working correctly")


def test_it_sector_query():
    """Test: IT sector query"""
    print("\n" + "=" * 60)
    print("Test 6: IT Sector Query - 'IT sector technology'")
    print("=" * 60)
    
    qp = setup_test_system()
    results = qp.process_query("IT sector technology companies", top_k=5)
    
    print(f"\nFound {len(results)} results:")
    for i, article in enumerate(results, 1):
        score = getattr(article, 'relevance_score', 0)
        print(f"{i}. [{article.id}] {article.title}")
        print(f"   Score: {score:.3f}")
        companies = article.entities.get("Companies", [])
        sectors = article.entities.get("Sectors", [])
        print(f"   Companies: {companies}")
        print(f"   Sectors: {sectors}")
    
    # Verify IT sector articles
    it_articles = [a for a in results if "IT" in a.entities.get("Sectors", [])]
    assert len(it_articles) > 0, "Should return IT sector articles"
    
    print("\n✓ IT sector query working correctly")


def test_explain_query():
    """Test query explanation feature with multi-query approach"""
    print("\n" + "=" * 60)
    print("Test 7: Query Explanation (Multi-Query)")
    print("=" * 60)
    
    qp = setup_test_system()
    
    queries = [
        "HDFC Bank news",
        "Banking sector update",
        "RBI policy changes",
        "interest rate impact"
    ]
    
    for query in queries:
        print(f"\n[Query] {query}")
        explanation = qp.explain_query(query)
        print(f"  Strategies: {explanation['strategies']}")
        print(f"  Primary Query: {explanation['primary_query']}")
        print(f"  Context Queries ({len(explanation['context_queries'])}): {explanation['context_queries'][:2]}")
        print(f"  Companies: {explanation['identified_entities']['companies']}")
        print(f"  Sectors: {explanation['identified_entities']['sectors']}")
        print(f"  Regulators: {explanation['identified_entities']['regulators']}")
        
        # Verify multi-query structure
        assert explanation['primary_query'] == query, "Primary query should match original"
    
    print("\n✓ Query explanation working with multi-query approach")


def test_multi_company_query():
    """Test query with multiple companies"""
    print("\n" + "=" * 60)
    print("Test 8: Multi-Company Query")
    print("=" * 60)
    
    qp = setup_test_system()
    results = qp.process_query("TCS Infosys technology deals", top_k=5)
    
    print(f"\nFound {len(results)} results:")
    for i, article in enumerate(results, 1):
        score = getattr(article, 'relevance_score', 0)
        print(f"{i}. [{article.id}] {article.title}")
        print(f"   Score: {score:.3f}")
        companies = article.entities.get("Companies", [])
        print(f"   Companies: {companies}")
    
    # Should find TCS and Infosys articles
    result_ids = [a.id for a in results]
    tcs_found = any("N008" in result_ids for _ in [1])  # N008 is TCS
    infosys_found = any("N009" in result_ids for _ in [1])  # N009 is Infosys
    
    assert tcs_found or infosys_found, "Should find TCS or Infosys articles"
    print("\n✓ Multi-company query working")


def test_query_with_min_similarity():
    """Test minimum similarity threshold"""
    print("\n" + "=" * 60)
    print("Test 9: Minimum Similarity Threshold")
    print("=" * 60)
    
    qp = setup_test_system()
    
    # High threshold
    results_high = qp.process_query("HDFC Bank", top_k=10, min_similarity=0.6)
    print(f"\nHigh threshold (0.6): {len(results_high)} results")
    
    # Low threshold
    results_low = qp.process_query("HDFC Bank", top_k=10, min_similarity=0.2)
    print(f"Low threshold (0.2): {len(results_low)} results")
    
    assert len(results_high) <= len(results_low), "High threshold should return fewer results"
    
    print("\n✓ Similarity threshold working")


if __name__ == "__main__":
    print("Testing Context-Aware Query Processing")
    print("=" * 60)
    
    test_query_expansion()
    test_direct_mention_query()
    test_sector_wide_query()
    test_regulator_filter_query()
    test_semantic_theme_query()
    test_it_sector_query()
    test_explain_query()
    test_multi_company_query()
    test_query_with_min_similarity()
    
    print("\n" + "=" * 60)
    print("✅ All query processing tests passed!")