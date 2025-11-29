from news_storage import NewsArticle, NewsStorage, load_mock_dataset
from datetime import datetime

def test_news_storage():
    # Test NewsStorage operations
    storage = NewsStorage()
    
    article = NewsArticle(
        id="TEST001",
        title="Test Article",
        content="Test content",
        source="Test Source",
        timestamp=datetime.now(),
        raw_text="Test raw text"
    )
    
    storage.add_article(article)
    assert storage.get_by_id("TEST001") == article
    assert len(storage.get_all_articles()) == 1
    print("✓ Basic storage operations working")

def test_mock_dataset():
    # Test loading mock dataset
    articles = load_mock_dataset('mock_news_data.json')
    assert len(articles) >= 30
    assert all(isinstance(a, NewsArticle) for a in articles)
    print(f"✓ Loaded {len(articles)} articles from mock dataset")
    
    # Test retrieval by ID
    storage = NewsStorage()
    for article in articles:
        storage.add_article(article)
    
    retrieved = storage.get_by_id("N001")
    assert retrieved is not None
    assert retrieved.title == "HDFC Bank announces 15% dividend, board approves stock buyback"
    print(f"✓ Retrieved article by ID: {retrieved.title}")
    
    all_articles = storage.get_all_articles()
    assert len(all_articles) == len(articles)
    print(f"✓ Retrieved all {len(all_articles)} articles")

if __name__ == "__main__":
    test_news_storage()
    test_mock_dataset()
    print("\n✅ All tests passed!")