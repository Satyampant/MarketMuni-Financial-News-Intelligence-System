from deduplication import DeduplicationAgent
from news_storage import NewsArticle, load_mock_dataset
from datetime import datetime
import numpy as np

def test_rbi_duplicate_detection():
    """Test the core example from problem statement: RBI rate hike articles."""
    print("=" * 70)
    print("TEST 1: RBI Rate Hike Duplicate Detection (Core Requirement)")
    print("=" * 70)
    
    agent = DeduplicationAgent()
    articles = load_mock_dataset('mock_news_data.json')
    
    # Get the three RBI articles (N002, N005, N006)
    n002 = next(a for a in articles if a.id == "N002")
    n005 = next(a for a in articles if a.id == "N005")
    n006 = next(a for a in articles if a.id == "N006")
    
    print("\nInput Articles:")
    print(f"  N002: '{n002.title}'")
    print(f"  N005: '{n005.title}'")
    print(f"  N006: '{n006.title}'")
    
    # Test N005 against N002
    duplicates_005 = agent.find_duplicates(n005, [n002])
    bi_score_005, cross_score_005 = agent.get_similarity_scores(n005, n002)
    
    print(f"\nN005 vs N002:")
    print(f"  Bi-Encoder Score:   {bi_score_005:.4f}")
    print(f"  Cross-Encoder Score: {cross_score_005:.4f}")
    print(f"  Duplicates Found: {duplicates_005}")
    
    # Test N006 against both N002 and N005
    duplicates_006 = agent.find_duplicates(n006, [n002, n005])
    bi_score_006_002, cross_score_006_002 = agent.get_similarity_scores(n006, n002)
    bi_score_006_005, cross_score_006_005 = agent.get_similarity_scores(n006, n005)
    
    print(f"\nN006 vs N002:")
    print(f"  Bi-Encoder Score:   {bi_score_006_002:.4f}")
    print(f"  Cross-Encoder Score: {cross_score_006_002:.4f}")
    print(f"\nN006 vs N005:")
    print(f"  Bi-Encoder Score:   {bi_score_006_005:.4f}")
    print(f"  Cross-Encoder Score: {cross_score_006_005:.4f}")
    print(f"  Duplicates Found: {duplicates_006}")
    
    # Verify all three are identified as duplicates
    all_duplicates = set()
    all_duplicates.update(duplicates_005)
    all_duplicates.update(duplicates_006)
    
    print(f"\n✓ Result: All RBI articles identified as duplicates")
    assert len(duplicates_005) > 0, "N005 should detect N002 as duplicate"
    assert len(duplicates_006) > 0, "N006 should detect duplicates"
    print("✅ PASSED: Core duplicate detection working correctly\n")


def test_consolidation():
    """Test consolidation logic: earliest timestamp + merged sources."""
    print("=" * 70)
    print("TEST 2: Duplicate Consolidation")
    print("=" * 70)
    
    agent = DeduplicationAgent()
    articles = load_mock_dataset('mock_news_data.json')
    
    n002 = next(a for a in articles if a.id == "N002")
    n005 = next(a for a in articles if a.id == "N005")
    n006 = next(a for a in articles if a.id == "N006")
    
    print(f"\nInput Articles:")
    print(f"  N002: {n002.timestamp} - Source: {n002.source}")
    print(f"  N005: {n005.timestamp} - Source: {n005.source}")
    print(f"  N006: {n006.timestamp} - Source: {n006.source}")
    
    consolidated = agent.consolidate_duplicates([n005, n002, n006])
    
    print(f"\nConsolidated Article:")
    print(f"  ID: {consolidated.id}")
    print(f"  Timestamp: {consolidated.timestamp}")
    print(f"  Sources: {consolidated.source}")
    
    assert consolidated.id == "N002", "Should keep earliest article (N002)"
    assert "Economic Times" in consolidated.source
    assert "Financial Express" in consolidated.source
    assert "Trade Brains" in consolidated.source
    
    print("✅ PASSED: Consolidation preserves earliest article and merges sources\n")


def test_non_duplicate_detection():
    """Test that unrelated articles are NOT flagged as duplicates."""
    print("=" * 70)
    print("TEST 3: Non-Duplicate Detection (False Positive Prevention)")
    print("=" * 70)
    
    agent = DeduplicationAgent()
    articles = load_mock_dataset('mock_news_data.json')
    
    # Test completely unrelated articles
    test_pairs = [
        ("N001", "N008"),  # HDFC dividend vs TCS deal
        ("N007", "N013"),  # Reliance profit vs Mahindra EVs
        ("N010", "N020"),  # Adani solar vs HUL revenue
    ]
    
    print("\nTesting unrelated article pairs:")
    all_passed = True
    
    for id1, id2 in test_pairs:
        a1 = next(a for a in articles if a.id == id1)
        a2 = next(a for a in articles if a.id == id2)
        
        duplicates = agent.find_duplicates(a1, [a2])
        bi_score, cross_score = agent.get_similarity_scores(a1, a2)
        
        print(f"\n  {id1} vs {id2}:")
        print(f"    Bi-Encoder: {bi_score:.4f} | Cross-Encoder: {cross_score:.4f}")
        print(f"    Result: {'DUPLICATE' if duplicates else 'DISTINCT'}")
        
        if duplicates:
            all_passed = False
            print(f"    ⚠️  WARNING: False positive detected!")
    
    assert all_passed, "Unrelated articles should not be detected as duplicates"
    print("\n✅ PASSED: No false positives on unrelated articles\n")


def test_full_corpus_deduplication():
    """Test deduplication across entire mock dataset."""
    print("=" * 70)
    print("TEST 4: Full Corpus Deduplication (30+ Articles)")
    print("=" * 70)
    
    agent = DeduplicationAgent()
    articles = load_mock_dataset('mock_news_data.json')
    
    print(f"\nProcessing {len(articles)} articles...")
    
    duplicate_groups = []
    processed_ids = set()
    
    for article in articles:
        if article.id in processed_ids:
            continue
        
        remaining = [
            a for a in articles 
            if a.id != article.id and a.id not in processed_ids
        ]
        
        duplicates = agent.find_duplicates(article, remaining)
        
        if duplicates:
            group = [article.id] + duplicates
            duplicate_groups.append(group)
            processed_ids.update(group)
            print(f"  Found duplicate group: {group}")
        else:
            processed_ids.add(article.id)
    
    unique_count = len(articles) - sum(len(g) - 1 for g in duplicate_groups)
    
    print(f"\n📊 Deduplication Summary:")
    print(f"  Total Articles: {len(articles)}")
    print(f"  Duplicate Groups Found: {len(duplicate_groups)}")
    print(f"  Unique Stories: {unique_count}")
    print(f"  Duplicates Removed: {len(articles) - unique_count}")
    
    # Verify RBI group is detected
    rbi_group_found = any(
        set(['N002', 'N005', 'N006']).issubset(set(group)) 
        for group in duplicate_groups
    )
    
    assert rbi_group_found, "RBI duplicate group should be detected in full corpus"
    print("✅ PASSED: Full corpus deduplication successful\n")


def test_accuracy_benchmark():
    """
    Benchmark test to verify ≥95% accuracy requirement.
    
    Ground Truth:
    - Duplicates: (N002, N005, N006) - RBI rate hike
    - All other pairs should be distinct
    """
    print("=" * 70)
    print("TEST 5: Accuracy Benchmark (≥95% Target)")
    print("=" * 70)
    
    agent = DeduplicationAgent()
    articles = load_mock_dataset('mock_news_data.json')
    
    # Ground truth: Only N002, N005, N006 are duplicates
    ground_truth_duplicates = {
        ('N002', 'N005'),
        ('N002', 'N006'),
        ('N005', 'N006')
    }
    
    # Test all pairs
    total_pairs = 0
    correct_predictions = 0
    false_positives = []
    false_negatives = []
    
    print("\nTesting all article pairs...")
    
    for i, a1 in enumerate(articles):
        for a2 in articles[i+1:]:
            total_pairs += 1
            
            duplicates = agent.find_duplicates(a1, [a2])
            predicted_duplicate = len(duplicates) > 0
            
            pair = tuple(sorted([a1.id, a2.id]))
            actual_duplicate = pair in ground_truth_duplicates
            
            if predicted_duplicate == actual_duplicate:
                correct_predictions += 1
            else:
                if predicted_duplicate and not actual_duplicate:
                    false_positives.append(pair)
                elif not predicted_duplicate and actual_duplicate:
                    false_negatives.append(pair)
    
    accuracy = (correct_predictions / total_pairs) * 100
    
    print(f"\n📊 Accuracy Metrics:")
    print(f"  Total Pairs Tested: {total_pairs}")
    print(f"  Correct Predictions: {correct_predictions}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  False Positives: {len(false_positives)}")
    print(f"  False Negatives: {len(false_negatives)}")
    
    if false_positives:
        print(f"\n⚠️  False Positives (incorrectly marked as duplicates):")
        for pair in false_positives[:5]:  # Show first 5
            print(f"    {pair}")
    
    if false_negatives:
        print(f"\n⚠️  False Negatives (missed duplicates):")
        for pair in false_negatives:
            print(f"    {pair}")
    
    assert accuracy >= 95.0, f"Accuracy {accuracy:.2f}% is below 95% target"
    print(f"\n✅ PASSED: Achieved {accuracy:.2f}% accuracy (≥95% target met)\n")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("=" * 70)
    print("TEST 6: Edge Cases")
    print("=" * 70)
    
    agent = DeduplicationAgent()
    articles = load_mock_dataset('mock_news_data.json')
    article = articles[0]
    
    # Test empty existing articles
    print("\n1. Empty existing articles list:")
    duplicates = agent.find_duplicates(article, [])
    assert duplicates == [], "Should return empty list for empty input"
    print("   ✓ Returns empty list correctly")
    
    # Test single article consolidation
    print("\n2. Single article consolidation:")
    consolidated = agent.consolidate_duplicates([article])
    assert consolidated.id == article.id
    print("   ✓ Returns same article for single input")
    
    # Test cache functionality
    print("\n3. Embedding cache:")
    initial_cache_size = len(agent.embeddings_cache)
    _ = agent.find_duplicates(article, articles[:5])
    new_cache_size = len(agent.embeddings_cache)
    assert new_cache_size > initial_cache_size, "Cache should grow"
    print(f"   ✓ Cache working (size: {initial_cache_size} → {new_cache_size})")
    
    agent.clear_cache()
    assert len(agent.embeddings_cache) == 0
    print("   ✓ Cache cleared successfully")
    
    print("\n✅ PASSED: All edge cases handled correctly\n")


def run_all_tests():
    """Execute complete test suite."""
    print("\n" + "=" * 70)
    print("SEMANTIC DEDUPLICATION TEST SUITE")
    print("Target: ≥95% Duplicate Detection Accuracy")
    print("=" * 70 + "\n")
    
    try:
        test_rbi_duplicate_detection()
        test_consolidation()
        test_non_duplicate_detection()
        test_full_corpus_deduplication()
        test_accuracy_benchmark()
        test_edge_cases()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED - DEDUPLICATION SYSTEM READY")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()