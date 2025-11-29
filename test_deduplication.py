from deduplication import DeduplicationAgent
from news_storage import NewsArticle, load_mock_dataset
from datetime import datetime

def test_duplicate_detection():
    # Lower threshold (0.75) for better duplicate detection
    agent = DeduplicationAgent(similarity_threshold=0.75)
    
    # Test with RBI rate hike articles (N002, N005, N006)
    articles = load_mock_dataset('mock_news_data.json')
    
    n002 = next(a for a in articles if a.id == "N002")
    n005 = next(a for a in articles if a.id == "N005")
    n006 = next(a for a in articles if a.id == "N006")
    
    print("Testing duplicate detection on RBI rate hike articles:")
    print(f"  N002: {n002.title}")
    print(f"  N005: {n005.title}")
    print(f"  N006: {n006.title}\n")
    
    # Test finding duplicates for N005 against N002
    duplicates = agent.find_duplicates(n005, [n002])
    print(f"✓ Duplicates for N005 against N002: {duplicates}")
    assert len(duplicates) > 0, "Should detect N002 as duplicate of N005"
    
    # Test finding duplicates for N006 against N002 and N005
    duplicates = agent.find_duplicates(n006, [n002, n005])
    print(f"✓ Duplicates for N006 against [N002, N005]: {duplicates}")
    assert len(duplicates) > 0, "Should detect duplicates for N006"
    
    print()

def test_consolidation():
    agent = DeduplicationAgent()
    articles = load_mock_dataset('mock_news_data.json')
    
    n002 = next(a for a in articles if a.id == "N002")
    n005 = next(a for a in articles if a.id == "N005")
    n006 = next(a for a in articles if a.id == "N006")
    
    # Consolidate duplicates (should keep earliest timestamp)
    consolidated = agent.consolidate_duplicates([n005, n002, n006])
    assert consolidated.id == "N002", "Should keep earliest article (N002)"
    assert "Economic Times" in consolidated.source, "Should preserve source info"
    print(f"✓ Consolidated to article {consolidated.id} from {consolidated.timestamp}")
    print(f"  Sources: {consolidated.source}\n")

def test_non_duplicates():
    agent = DeduplicationAgent(similarity_threshold=0.75)
    articles = load_mock_dataset('mock_news_data.json')
    
    # Test unrelated articles (HDFC dividend vs TCS deal)
    n001 = next(a for a in articles if a.id == "N001")
    n008 = next(a for a in articles if a.id == "N008")
    
    duplicates = agent.find_duplicates(n001, [n008])
    assert len(duplicates) == 0, "Should not detect unrelated articles as duplicates"
    print(f"✓ Correctly identified non-duplicates (N001 vs N008)\n")

def test_similarity_scores():
    """Debug helper to see actual similarity scores"""
    agent = DeduplicationAgent(similarity_threshold=0.75)
    articles = load_mock_dataset('mock_news_data.json')
    
    print("--- Similarity Score Analysis ---")
    
    # RBI articles (should be high similarity)
    n002 = next(a for a in articles if a.id == "N002")
    n005 = next(a for a in articles if a.id == "N005")
    n006 = next(a for a in articles if a.id == "N006")
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    emb_002 = agent._get_embedding(n002).reshape(1, -1)
    emb_005 = agent._get_embedding(n005).reshape(1, -1)
    emb_006 = agent._get_embedding(n006).reshape(1, -1)
    
    sim_002_005 = cosine_similarity(emb_002, emb_005)[0][0]
    sim_002_006 = cosine_similarity(emb_002, emb_006)[0][0]
    sim_005_006 = cosine_similarity(emb_005, emb_006)[0][0]
    
    print(f"RBI Articles (Expected: High Similarity)")
    print(f"  N002 vs N005: {sim_002_005:.4f}")
    print(f"  N002 vs N006: {sim_002_006:.4f}")
    print(f"  N005 vs N006: {sim_005_006:.4f}")
    
    # Unrelated articles (should be low similarity)
    n001 = next(a for a in articles if a.id == "N001")
    n008 = next(a for a in articles if a.id == "N008")
    
    emb_001 = agent._get_embedding(n001).reshape(1, -1)
    emb_008 = agent._get_embedding(n008).reshape(1, -1)
    
    sim_001_008 = cosine_similarity(emb_001, emb_008)[0][0]
    
    print(f"\nUnrelated Articles (Expected: Low Similarity)")
    print(f"  N001 (HDFC) vs N008 (TCS): {sim_001_008:.4f}")
    print()

def test_full_corpus_deduplication():
    agent = DeduplicationAgent(similarity_threshold=0.75)
    articles = load_mock_dataset('mock_news_data.json')
    
    print("--- Testing Full Corpus Deduplication ---")
    duplicate_groups = []
    processed_ids = set()
    
    for article in articles:
        if article.id in processed_ids:
            continue
        
        remaining = [a for a in articles if a.id != article.id and a.id not in processed_ids]
        duplicates = agent.find_duplicates(article, remaining)
        
        if duplicates:
            group = [article.id] + duplicates
            duplicate_groups.append(group)
            processed_ids.update(group)
            print(f"✓ Found duplicate group: {group}")
        else:
            processed_ids.add(article.id)
    
    print(f"\n✓ Found {len(duplicate_groups)} duplicate group(s) in corpus")
    print(f"✓ Unique stories: {len(articles) - sum(len(g)-1 for g in duplicate_groups)}")
    print()

if __name__ == "__main__":
    print("Testing Semantic Deduplication Engine")
    print("=" * 50 + "\n")
    
    test_similarity_scores()
    test_duplicate_detection()
    test_consolidation()
    test_non_duplicates()
    test_full_corpus_deduplication()
    
    print("=" * 50)
    print("✅ All deduplication tests passed!")