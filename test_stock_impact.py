from stock_impact import StockImpactMapper, StockImpact
from entity_extraction import EntityExtractor

def test_direct_company_mention():
    mapper = StockImpactMapper()
    entities = {
        "Companies": ["HDFC Bank"],
        "Sectors": ["Banking"],
        "Regulators": [],
        "People": [],
        "Events": ["dividend", "buyback"]
    }
    
    impacts = mapper.map_to_stocks(entities)
    
    # Should have direct mention of HDFC Bank
    direct_impacts = [i for i in impacts if i.impact_type == "direct"]
    assert len(direct_impacts) == 1
    assert direct_impacts[0].symbol == "HDFCBANK"
    assert direct_impacts[0].confidence == 1.0
    print(f"✓ Direct company mention: {direct_impacts[0]}")

def test_sector_wide_impact():
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
    assert len(sector_impacts) == 3  # HDFCBANK, ICICIBANK, SBIN
    assert all(i.confidence == 0.7 for i in sector_impacts)
    print(f"✓ Sector-wide impact: Found {len(sector_impacts)} banking stocks")
    for impact in sector_impacts:
        print(f"  - {impact.symbol}: {impact.confidence}")

def test_regulatory_impact():
    mapper = StockImpactMapper()
    entities = {
        "Companies": [],
        "Sectors": [],
        "Regulators": ["RBI"],
        "People": [],
        "Events": []
    }
    
    impacts = mapper.map_to_stocks(entities)
    
    # Should have all stocks with 0.5 confidence for regulatory news
    regulatory_impacts = [i for i in impacts if i.impact_type == "regulatory"]
    assert len(regulatory_impacts) > 0
    assert all(i.confidence == 0.5 for i in regulatory_impacts)
    print(f"✓ Regulatory impact: Found {len(regulatory_impacts)} stocks")

def test_combined_impact():
    mapper = StockImpactMapper()
    entities = {
        "Companies": ["HDFC Bank"],
        "Sectors": ["Banking"],
        "Regulators": [],
        "People": [],
        "Events": []
    }
    
    impacts = mapper.map_to_stocks(entities)
    
    # HDFC Bank should appear only once (direct), not in sector
    hdfcbank_impacts = [i for i in impacts if i.symbol == "HDFCBANK"]
    assert len(hdfcbank_impacts) == 1
    assert hdfcbank_impacts[0].impact_type == "direct"
    assert hdfcbank_impacts[0].confidence == 1.0
    
    # Other banking stocks should appear with sector confidence
    other_banking = [i for i in impacts if i.symbol in ["ICICIBANK", "SBIN"]]
    assert all(i.confidence == 0.7 for i in other_banking)
    print(f"✓ Combined impact: Direct + Sector (no duplicates)")
    for impact in impacts:
        print(f"  - {impact.symbol}: {impact.confidence} ({impact.impact_type})")

def test_integration_with_entity_extractor():
    extractor = EntityExtractor()
    mapper = StockImpactMapper()
    
    # Test with actual article text
    text = "HDFC Bank announces 15% dividend, board approves stock buyback"
    entities = extractor.extract_entities(text)
    impacts = mapper.map_to_stocks(entities)
    
    print(f"\n✓ Integration Test:")
    print(f"  Text: {text}")
    print(f"  Extracted Entities: {entities}")
    print(f"  Stock Impacts:")
    for impact in impacts:
        print(f"    - {impact.symbol}: {impact.confidence} ({impact.impact_type})")
    
    assert len(impacts) > 0
    direct = [i for i in impacts if i.impact_type == "direct"]
    assert len(direct) > 0

def test_rbi_article():
    extractor = EntityExtractor()
    mapper = StockImpactMapper()
    
    text = "RBI raises repo rate by 25bps to 6.75%, citing inflation concerns"
    entities = extractor.extract_entities(text)
    impacts = mapper.map_to_stocks(entities)
    
    print(f"\n✓ RBI Article Test:")
    print(f"  Text: {text}")
    print(f"  Extracted Entities: {entities}")
    print(f"  Stock Impacts: {len(impacts)} stocks affected")
    
    # RBI news should have regulatory impact on all stocks
    assert len(impacts) > 0
    assert all(i.impact_type == "regulatory" for i in impacts)
    assert all(i.confidence == 0.5 for i in impacts)

if __name__ == "__main__":
    print("Testing Stock Impact Mapping")
    print("=" * 60 + "\n")
    
    test_direct_company_mention()
    print()
    test_sector_wide_impact()
    print()
    test_regulatory_impact()
    print()
    test_combined_impact()
    print()
    test_integration_with_entity_extractor()
    test_rbi_article()
    
    print("\n" + "=" * 60)
    print("✅ All stock impact mapping tests passed!")