# test_entity_extraction.py
from entity_extraction import EntityExtractor
e = EntityExtractor(alias_path="company_aliases.json", sector_path="sector_tickers.json")
out = e.extract_entities("RBI raises rates affecting HDFC Bank")
assert "HDFC Bank" in out["Companies"], out
assert "RBI" in out["Regulators"], out
print("Smoke test passed:", out)


rbi = "The Reserve Bank of India increased the repo rate by 25 basis points to 6.75% in its monetary policy meeting, citing persistent inflation concerns."
assert rbi is not None, "RBI test article not found in mock dataset"
res = e.extract_entities(rbi)
print("RBI extraction:", res)
assert "RBI" in res.get("Regulators", [])