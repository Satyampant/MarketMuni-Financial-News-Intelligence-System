
from news_storage import load_mock_dataset, NewsStorage
from deduplication import DeduplicationAgent
from entity_extraction import EntityExtractor
from stock_impact import StockImpactMapper
from vector_store import VectorStore

def process_and_index_news():
    """
    Complete pipeline: Load → Deduplicate → Extract → Map → Index
    """
    print("Financial News Intelligence Pipeline")
    print("=" * 60)
    
    # Step 1: Load mock news data
    print("\n[Step 1] Loading news articles...")
    articles = load_mock_dataset('mock_news_data.json')
    print(f"✓ Loaded {len(articles)} articles")
    
    # Step 2: Initialize modules
    print("\n[Step 2] Initializing modules...")
    storage = NewsStorage()
    dedup_agent = DeduplicationAgent()
    extractor = EntityExtractor()
    mapper = StockImpactMapper()
    shared_model = dedup_agent.embedding_model
    vector_store = VectorStore(collection_name="financial_news_production", embedding_model=shared_model)
    vector_store.reset()  # Start fresh
    print("✓ All modules initialized")
    
    # Step 3: Process articles with deduplication
    print("\n[Step 3] Processing articles with deduplication...")
    unique_articles = []
    processed_ids = set()
    
    for article in articles:
        if article.id in processed_ids:
            continue
        
        # Check for duplicates
        existing = [a for a in unique_articles]
        duplicates = dedup_agent.find_duplicates(article, existing)
        
        if duplicates:
            print(f"  ⚠ Duplicate found: {article.id} matches {duplicates}")
            processed_ids.add(article.id)
        else:
            unique_articles.append(article)
            storage.add_article(article)
            processed_ids.add(article.id)
    
    print(f"✓ Identified {len(unique_articles)} unique stories (removed {len(articles) - len(unique_articles)} duplicates)")
    
    # Step 4: Extract entities and map stocks
    print("\n[Step 4] Extracting entities and mapping stock impacts...")
    for article in unique_articles:
        # Extract entities
        text = f"{article.title}. {article.content}"
        entities = extractor.extract_entities(text)
        article.entities = entities
        
        # Map to stocks
        impacts = mapper.map_to_stocks(entities)
        article.impacted_stocks = impacts
        
        print(f"  ✓ {article.id}: {len(entities.get('Companies', []))} companies, {len(impacts)} stocks impacted")
    
    # Step 5: Index in vector store
    print("\n[Step 5] Indexing articles in vector database...")
    for article in unique_articles:
        cached_emb = dedup_agent.get_cached_embedding(article.id)
        vector_store.index_article(article)
    
    print(f"✓ Indexed {vector_store.count()} articles in vector store")
    
    return vector_store, unique_articles

def demonstrate_queries(vector_store):
    """
    Demonstrate various query patterns on indexed data.
    """
    print("\n" + "=" * 60)
    print("Query Demonstration")
    print("=" * 60)
    
    # Query 1: Company-specific news
    print("\n[Query 1] HDFC Bank news")
    print("-" * 40)
    results = vector_store.search("HDFC Bank news", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['metadata']['title']}")
        print(f"   Similarity: {r['similarity']:.3f}")
        stocks = [s['symbol'] for s in r['metadata'].get('impacted_stocks', [])]
        print(f"   Stocks: {', '.join(stocks[:5])}")
        print()
    
    # Query 2: Sector-wide news
    print("\n[Query 2] Banking sector update")
    print("-" * 40)
    results = vector_store.search("Banking sector update", top_k=4)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['metadata']['title']}")
        print(f"   Similarity: {r['similarity']:.3f}")
        entities = r['metadata'].get('entities', {})
        sectors = entities.get('Sectors', [])
        print(f"   Sectors: {', '.join(sectors) if sectors else 'N/A'}")
        print()
    
    # Query 3: Regulatory news
    print("\n[Query 3] RBI policy changes")
    print("-" * 40)
    results = vector_store.search("RBI policy changes", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['metadata']['title']}")
        print(f"   Similarity: {r['similarity']:.3f}")
        entities = r['metadata'].get('entities', {})
        regulators = entities.get('Regulators', [])
        print(f"   Regulators: {', '.join(regulators) if regulators else 'N/A'}")
        stocks = r['metadata'].get('impacted_stocks', [])
        reg_stocks = [s for s in stocks if s['impact_type'] == 'regulatory']
        print(f"   Regulatory Impact: {len(reg_stocks)} stocks")
        print()
    
    # Query 4: Thematic search
    print("\n[Query 4] Interest rate impact")
    print("-" * 40)
    results = vector_store.search("interest rate impact financial markets", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['metadata']['title']}")
        print(f"   Similarity: {r['similarity']:.3f}")
        entities = r['metadata'].get('entities', {})
        events = entities.get('Events', [])
        print(f"   Events: {', '.join(events) if events else 'N/A'}")
        print()
    
    # Query 5: IT sector news
    print("\n[Query 5] IT sector technology companies")
    print("-" * 40)
    results = vector_store.search("IT sector technology companies", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['metadata']['title']}")
        print(f"   Similarity: {r['similarity']:.3f}")
        stocks = r['metadata'].get('impacted_stocks', [])
        it_stocks = [s['symbol'] for s in stocks if 'TCS' in s['symbol'] or 'INFOSYS' in s['symbol'] or 'WIPRO' in s['symbol']]
        print(f"   IT Stocks: {', '.join(it_stocks) if it_stocks else 'Various'}")
        print()

def show_statistics(vector_store, articles):
    """
    Display statistics about indexed data.
    """
    print("\n" + "=" * 60)
    print("Pipeline Statistics")
    print("=" * 60)
    
    total_articles = len(articles)
    indexed_count = vector_store.count()
    
    # Count entities
    total_companies = sum(len(a.entities.get('Companies', [])) for a in articles)
    total_sectors = sum(len(a.entities.get('Sectors', [])) for a in articles)
    total_regulators = sum(len(a.entities.get('Regulators', [])) for a in articles)
    
    # Count stock impacts
    total_impacts = sum(len(a.impacted_stocks) for a in articles)
    direct_impacts = sum(len([s for s in a.impacted_stocks if s.impact_type == 'direct']) for a in articles)
    sector_impacts = sum(len([s for s in a.impacted_stocks if s.impact_type == 'sector']) for a in articles)
    regulatory_impacts = sum(len([s for s in a.impacted_stocks if s.impact_type == 'regulatory']) for a in articles)
    
    print(f"\nArticles:")
    print(f"  Total unique articles: {total_articles}")
    print(f"  Indexed in vector store: {indexed_count}")
    
    print(f"\nEntities Extracted:")
    print(f"  Companies: {total_companies}")
    print(f"  Sectors: {total_sectors}")
    print(f"  Regulators: {total_regulators}")
    
    print(f"\nStock Impacts:")
    print(f"  Total impacts: {total_impacts}")
    print(f"  Direct mentions: {direct_impacts}")
    print(f"  Sector-wide: {sector_impacts}")
    print(f"  Regulatory: {regulatory_impacts}")
    
    print()

if __name__ == "__main__":
    # Run complete pipeline
    vector_store, articles = process_and_index_news()
    
    # Demonstrate queries
    demonstrate_queries(vector_store)
    
    # Show statistics
    show_statistics(vector_store, articles)
    
    print("=" * 60)
    print("✅ Pipeline completed successfully!")