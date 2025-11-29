from deduplication import DeduplicationAgent
from news_storage import NewsArticle, load_mock_dataset
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_duplicate_detection():
    # We no longer need to lower the threshold manually. 
    # The Cross-Encoder is smart enough to handle semantic similarity at high precision.
    agent = DeduplicationAgent() 
    
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
    print(f"  Merged Sources: {consolidated.source}\n")

def test_non_duplicates():
    agent = DeduplicationAgent()
    articles = load_mock_dataset('mock_news_data.json')
    
    # Test unrelated articles (HDFC dividend vs TCS deal)
    n001 = next(a for a in articles if a.id == "N001")
    n008 = next(a for a in articles if a.id == "N008")
    
    duplicates = agent.find_duplicates(n001, [n008])
    assert len(duplicates) == 0, "Should not detect unrelated articles as duplicates"
    print(f"✓ Correctly identified non-duplicates (N001 vs N008)\n")

def test_similarity_analysis():
    """
    Debug helper to analyze the Two-Stage Deduplication Process:
    1. Bi-Encoder Score (Candidate Retrieval)
    2. Cross-Encoder Score (Final Verification)
    """
    agent = DeduplicationAgent()
    articles = load_mock_dataset('mock_news_data.json')
    
    print("--- Two-Stage Similarity Analysis ---")
    
    # Helper to get cross-encoder score
    def get_cross_score(a1, a2):
        # Replicating the text formatting from DeduplicationAgent
        text1 = f"{a1.title}. {a1.content}"
        text2 = f"{a2.title}. {a2.content}"
        return agent.reranker.predict([[text1, text2]])[0]

    # Helper to get bi-encoder score
    def get_bi_score(a1, a2):
        emb1 = agent._get_embedding(a1).reshape(1, -1)
        emb2 = agent._get_embedding(a2).reshape(1, -1)
        return cosine_similarity(emb1, emb2)[0][0]

    # RBI articles (True Duplicates)
    n002 = next(a for a in articles if a.id == "N002")
    n005 = next(a for a in articles if a.id == "N005")
    
    bi_score_rbi = get_bi_score(n002, n005)
    cross_score_rbi = get_cross_score(n002, n005)
    
    print(f"RBI Articles (N002 vs N005):")
    print(f"  Bi-Encoder (Cosine):   {bi_score_rbi:.4f} (Filters candidates)")
    print(f"  Cross-Encoder (Final): {cross_score_rbi:.4f} (Decides duplicate)")
    print(f"  Result: {'DUPLICATE' if cross_score_rbi > 0.5 else 'DISTINCT'}")

    # Related article 2
    n006 = next(a for a in articles if a.id == "N006")
    
    bi_score_rbi = get_bi_score(n006, n005)
    cross_score_rbi = get_cross_score(n006, n005)
    
    print(f"RBI Articles (N006 vs N005):")
    print(f"  Bi-Encoder (Cosine):   {bi_score_rbi:.4f} (Filters candidates)")
    print(f"  Cross-Encoder (Final): {cross_score_rbi:.4f} (Decides duplicate)")
    print(f"  Result: {'DUPLICATE' if cross_score_rbi > 0.8 else 'DISTINCT'}")
    
    # Unrelated articles (Distinct)
    n001 = next(a for a in articles if a.id == "N001")
    n008 = next(a for a in articles if a.id == "N008")
    
    bi_score_diff = get_bi_score(n001, n008)
    cross_score_diff = get_cross_score(n001, n008)
    
    print(f"\nUnrelated Articles (N001 vs N008):")
    print(f"  Bi-Encoder (Cosine):   {bi_score_diff:.4f}")
    print(f"  Cross-Encoder (Final): {cross_score_diff:.4f}")
    print(f"  Result: {'DUPLICATE' if cross_score_diff > 0.8 else 'DISTINCT'}")
    print()

def test_full_corpus_deduplication():
    agent = DeduplicationAgent()
    articles = load_mock_dataset('mock_news_data.json')
    
    print("--- Testing Full Corpus Deduplication ---")
    duplicate_groups = []
    processed_ids = set()
    
    for article in articles:
        if article.id in processed_ids:
            continue
        
        # In a real scenario, we check all, but here we just pass the rest
        remaining = [a for a in articles if a.id != article.id and a.id not in processed_ids]
        
        # DeduplicationAgent now handles the two-stage process internally
        duplicates = agent.find_duplicates(article, remaining)
        
        if duplicates:
            group = [article.id] + duplicates
            duplicate_groups.append(group)
            processed_ids.update(group)
            print(f"✓ Found duplicate group: {group}")
        else:
            processed_ids.add(article.id)
    
    total_found = len(articles) - sum(len(g)-1 for g in duplicate_groups)
    print(f"\n✓ Found {len(duplicate_groups)} duplicate group(s) in corpus")
    print(f"✓ Unique stories count: {total_found}")
    print()

if __name__ == "__main__":
    print("Testing Semantic Deduplication Engine (Bi-Encoder + Cross-Encoder)")
    print("=" * 60 + "\n")
    
    test_similarity_analysis()
    test_duplicate_detection()
    test_consolidation()
    test_non_duplicates()
    test_full_corpus_deduplication()
    
    print("=" * 60)
    print("✅ All deduplication tests passed!")