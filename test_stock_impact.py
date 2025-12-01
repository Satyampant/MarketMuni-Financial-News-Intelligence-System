from stock_impact import StockImpactMapper, StockImpact
from entity_extraction import EntityExtractor
from news_storage import NewsArticle
from datetime import datetime


def test_direct_company_mention():
    """Test direct company mention with 100% confidence."""
    print("Test 1: Direct Company Mention")
    mapper = StockImpactMapper()
    
    entities = {
        "Companies": ["HDFC Bank"],
        "Sectors": [],
        "Regulators": [],
        "People": [],
        "Events": ["dividend", "buyback"]
    }
    
    impacts = mapper.map_to_stocks(entities)
    
    # Should have direct mention of HDFC Bank
    direct_impacts = [i for i in impacts if i.impact_type == "direct"]
    assert len(direct_impacts) == 1, f"Expected 1 direct impact, got {len(direct_impacts)}"
    assert direct_impacts[0].symbol == "HDFCBANK", f"Expected HDFCBANK, got {direct_impacts[0].symbol}"
    assert direct_impacts[0].confidence == 1.0, f"Expected confidence 1.0, got {direct_impacts[0].confidence}"
    
    print(f"  ✓ Direct mention: {direct_impacts[0].symbol} with confidence {direct_impacts[0].confidence}")
    print(f"    Source: {direct_impacts[0].source_entity}")
    print(f"    Rule: {direct_impacts[0].rule}\n")


def test_sector_wide_impact():
    """Test sector-wide impact with 70% confidence."""
    print("Test 2: Sector-Wide Impact")
    mapper = StockImpactMapper()
    
    entities = {
        "Companies": [],
        "Sectors": ["Banking"],
        "Regulators": [],
        "People": [],
        "Events": []
    }
    
    impacts = mapper.map_to_stocks(entities)
    
    # Should have all Banking sector stocks with 0.7 confidence
    sector_impacts = [i for i in impacts if i.impact_type == "sector"]
    assert len(sector_impacts) == 3, f"Expected 3 banking stocks, got {len(sector_impacts)}"
    
    symbols = {i.symbol for i in sector_impacts}
    expected_symbols = {"HDFCBANK", "ICICIBANK", "SBIN"}
    assert symbols == expected_symbols, f"Expected {expected_symbols}, got {symbols}"
    
    for impact in sector_impacts:
        assert impact.confidence == 0.7, f"Expected 0.7, got {impact.confidence}"
    
    print(f"  ✓ Sector-wide impact: Found {len(sector_impacts)} banking stocks")
    for impact in sector_impacts:
        print(f"    - {impact.symbol}: {impact.confidence} ({impact.impact_type})")
    print()


def test_regulatory_impact():
    """Test regulatory impact with 50% confidence."""
    print("Test 3: Regulatory Impact")
    mapper = StockImpactMapper()
    
    entities = {
        "Companies": [],
        "Sectors": [],
        "Regulators": ["RBI"],
        "People": [],
        "Events": []
    }
    
    impacts = mapper.map_to_stocks(entities)
    
    # RBI impacts Banking and Finance sectors
    regulatory_impacts = [i for i in impacts if i.impact_type == "regulatory"]
    assert len(regulatory_impacts) > 0, "Expected regulatory impacts"
    
    for impact in regulatory_impacts:
        assert impact.confidence == 0.5, f"Expected 0.5, got {impact.confidence}"
        assert "RBI" in impact.source_entity, f"Expected RBI in source, got {impact.source_entity}"
    
    print(f"  ✓ Regulatory impact: Found {len(regulatory_impacts)} stocks")
    for impact in regulatory_impacts[:5]:  # Show first 5
        print(f"    - {impact.symbol}: {impact.confidence} ({impact.source_entity})")
    print()


def test_combined_impact():
    """Test combining direct and sector impacts (should not duplicate)."""
    print("Test 4: Combined Direct + Sector Impact")
    mapper = StockImpactMapper()
    
    entities = {
        "Companies": ["HDFC Bank"],
        "Sectors": ["Banking"],
        "Regulators": [],
        "People": [],
        "Events": []
    }
    
    impacts = mapper.map_to_stocks(entities)
    
    # HDFC Bank should appear only once with combined confidence
    hdfcbank_impacts = [i for i in impacts if i.symbol == "HDFCBANK"]
    assert len(hdfcbank_impacts) == 1, f"Expected 1 HDFCBANK entry, got {len(hdfcbank_impacts)}"
    
    hdfcbank = hdfcbank_impacts[0]
    # Should combine 1.0 (direct) + 0.7 (sector) = 1 - (1-1.0)*(1-0.7) = 1.0
    assert hdfcbank.impact_type == "direct", "Direct should take precedence"
    assert hdfcbank.confidence == 1.0, f"Expected combined confidence 1.0, got {hdfcbank.confidence}"
    
    # Other banking stocks should appear with sector confidence only
    other_banking = [i for i in impacts if i.symbol in ["ICICIBANK", "SBIN"]]
    assert len(other_banking) == 2, f"Expected 2 other banks, got {len(other_banking)}"
    
    for impact in other_banking:
        assert impact.confidence == 0.7, f"Expected 0.7, got {impact.confidence}"
        assert impact.impact_type == "sector", f"Expected sector, got {impact.impact_type}"
    
    print(f"  ✓ Combined impact: No duplicates, proper precedence")
    for impact in impacts:
        print(f"    - {impact.symbol}: {impact.confidence} ({impact.impact_type})")
    print()


def test_confidence_combination():
    """Test that multiple impacts combine confidences correctly."""
    print("Test 5: Confidence Combination Logic")
    mapper = StockImpactMapper()
    
    # Banking stock mentioned directly + sector impact
    entities = {
        "Companies": ["ICICI Bank"],
        "Sectors": ["Banking"],
        "Regulators": ["RBI"],
        "People": [],
        "Events": []
    }
    
    impacts = mapper.map_to_stocks(entities)
    
    icici = [i for i in impacts if i.symbol == "ICICIBANK"][0]
    
    # Should combine: direct(1.0) + sector(0.7) + regulatory(0.5)
    # P(A or B or C) = 1 - (1-1.0)*(1-0.7)*(1-0.5) = 1.0
    print(f"  ✓ ICICIBANK combined confidence: {icici.confidence}")
    print(f"    Impact type: {icici.impact_type} (should be 'direct')")
    print(f"    Sources: {icici.source_entity}")
    print(f"    Rules: {icici.rule}")
    
    assert icici.impact_type == "direct", "Direct should take precedence"
    assert icici.confidence == 1.0, f"Expected 1.0, got {icici.confidence}"
    print()


def test_integration_with_entity_extractor():
    """Test full pipeline: text -> entities -> stock impacts."""
    print("Test 6: Integration with Entity Extractor")
    extractor = EntityExtractor()
    mapper = StockImpactMapper()
    
    # Test with actual article text
    text = "HDFC Bank announces 15% dividend, board approves stock buyback"
    entities = extractor.extract_entities(text)
    impacts = mapper.map_to_stocks(entities)
    
    print(f"  Text: {text}")
    print(f"  Extracted Entities:")
    for key, values in entities.items():
        if values:
            print(f"    {key}: {values}")
    
    print(f"  Stock Impacts:")
    for impact in impacts:
        print(f"    - {impact.symbol}: {impact.confidence} ({impact.impact_type})")
    
    assert len(impacts) > 0, "Should have at least one impact"
    
    direct = [i for i in impacts if i.impact_type == "direct"]
    assert len(direct) > 0, "Should have direct impacts"
    print()


def test_rbi_article():
    """Test regulatory article (RBI rate change)."""
    print("Test 7: RBI Regulatory Article")
    extractor = EntityExtractor()
    mapper = StockImpactMapper()
    
    text = "RBI raises repo rate by 25bps to 6.75%, citing inflation concerns"
    entities = extractor.extract_entities(text)
    impacts = mapper.map_to_stocks(entities)
    
    print(f"  Text: {text}")
    print(f"  Extracted Entities:")
    for key, values in entities.items():
        if values:
            print(f"    {key}: {values}")
    
    print(f"  Stock Impacts: {len(impacts)} stocks affected")
    for impact in impacts[:5]:  # Show first 5
        print(f"    - {impact.symbol}: {impact.confidence} ({impact.impact_type})")
    
    # RBI news should have regulatory impact
    assert len(impacts) > 0, "Should have impacts"
    
    regulatory = [i for i in impacts if i.impact_type == "regulatory"]
    assert len(regulatory) > 0, "Should have regulatory impacts"
    
    for impact in regulatory:
        assert impact.confidence == 0.5, f"Regulatory confidence should be 0.5, got {impact.confidence}"
        assert "RBI" in impact.source_entity, "Source should mention RBI"
    print()


def test_map_article_method():
    """Test the map_article method that updates NewsArticle objects."""
    print("Test 8: map_article() Method")
    extractor = EntityExtractor()
    mapper = StockImpactMapper()
    
    article = NewsArticle(
        id="TEST001",
        title="HDFC Bank announces dividend",
        content="HDFC Bank announced a 15% dividend payout today.",
        source="TestSource",
        timestamp=datetime.now(),
        raw_text="HDFC Bank announces dividend"
    )
    
    # Map the article
    updated_article = mapper.map_article(article, extractor)
    
    print(f"  Article: {updated_article.title}")
    print(f"  Impacted Stocks: {len(updated_article.impacted_stocks)}")
    
    for impact in updated_article.impacted_stocks:
        print(f"    - {impact['symbol']}: {impact['confidence']} ({impact['impact_type']})")
    
    assert len(updated_article.impacted_stocks) > 0, "Should have stock impacts"
    assert isinstance(updated_article.impacted_stocks[0], dict), "Should be dict format"
    assert "symbol" in updated_article.impacted_stocks[0], "Should have symbol key"
    print()


def test_case_insensitive_matching():
    """Test that sector and company matching is case-insensitive."""
    print("Test 9: Case-Insensitive Matching")
    mapper = StockImpactMapper()
    
    entities = {
        "Companies": ["hdfc bank", "ICICI BANK"],  # Different cases
        "Sectors": ["banking", "IT"],  # Lowercase
        "Regulators": ["rbi"],  # Lowercase
        "People": [],
        "Events": []
    }
    
    impacts = mapper.map_to_stocks(entities)
    
    symbols = {i.symbol for i in impacts}
    print(f"  Found symbols: {symbols}")
    
    # Should find HDFC and ICICI despite case differences
    assert "HDFCBANK" in symbols, "Should find HDFCBANK"
    assert "ICICIBANK" in symbols, "Should find ICICIBANK"
    
    # Should find IT sector stocks
    it_stocks = {"TCS", "INFOSYS", "WIPRO", "TECHM"}
    assert any(stock in symbols for stock in it_stocks), "Should find IT sector stocks"
    
    print(f"  ✓ Case-insensitive matching works correctly")
    print()


def test_sorted_output():
    """Test that output is sorted by confidence (descending)."""
    print("Test 10: Sorted Output")
    mapper = StockImpactMapper()
    
    entities = {
        "Companies": ["HDFC Bank"],  # 1.0
        "Sectors": ["Banking"],  # 0.7
        "Regulators": ["RBI"],  # 0.5
        "People": [],
        "Events": []
    }
    
    impacts = mapper.map_to_stocks(entities)
    
    # Check that impacts are sorted by confidence (descending)
    confidences = [i.confidence for i in impacts]
    assert confidences == sorted(confidences, reverse=True), "Should be sorted by confidence"
    
    print(f"  ✓ Output sorted by confidence:")
    for impact in impacts[:5]:
        print(f"    - {impact.symbol}: {impact.confidence}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("STOCK IMPACT MAPPING - COMPREHENSIVE TEST SUITE")
    print("=" * 70 + "\n")
    
    try:
        test_direct_company_mention()
        test_sector_wide_impact()
        test_regulatory_impact()
        test_combined_impact()
        test_confidence_combination()
        test_integration_with_entity_extractor()
        test_rbi_article()
        test_map_article_method()
        test_case_insensitive_matching()
        test_sorted_output()
        
        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise