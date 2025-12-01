from __future__ import annotations
from pathlib import Path
import json
import re
from typing import Dict, List, Union, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from news_storage import NewsArticle

try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

MODULE_DIR = Path(__file__).parent


@dataclass
class EntityConfidence:
    """Track entity extraction with confidence scores"""
    entity: str
    entity_type: str
    confidence: float
    source: str  # "matcher", "ner", "regex", "inference"


class EntityExtractor:
    def __init__(
        self,
        alias_path: Optional[Union[str, Path]] = None,
        sector_path: Optional[Union[str, Path]] = None,
        regulator_path: Optional[Union[str, Path]] = None,
        model_name: str = "en_core_web_sm",
        event_keywords: Optional[List[str]] = None,
    ):
        # Portable defaults (module-relative)
        alias_path = Path(alias_path) if alias_path else MODULE_DIR / "company_aliases.json"
        sector_path = Path(sector_path) if sector_path else MODULE_DIR / "sector_tickers.json"
        regulator_path = Path(regulator_path) if regulator_path else MODULE_DIR / "regulators.json"

        # Load company aliases
        self.alias_table = {}
        if alias_path.exists():
            self.alias_table = json.loads(alias_path.read_text())
        
        # Build reverse lookup: alias -> canonical name
        self.alias_to_canonical = {}
        for canonical, meta in self.alias_table.items():
            for alias in meta.get("aliases", [canonical]):
                self.alias_to_canonical[alias.lower()] = canonical

        # Load sector tickers
        self.sector_tickers = {}
        if sector_path.exists():
            self.sector_tickers = json.loads(sector_path.read_text())

        # Load regulators - create bidirectional mapping
        self.regulator_map = {}  # alias -> canonical
        self.canonical_regulators = set()  # all canonical regulator names
        if regulator_path.exists():
            reg_data = json.loads(regulator_path.read_text())
            for canonical, aliases in reg_data.items():
                self.canonical_regulators.add(canonical)
                for alias in aliases:
                    self.regulator_map[alias.lower()] = canonical

        # Event keywords (lowercased)
        self.event_keywords = [k.lower() for k in (event_keywords or [
            "dividend", "buyback", "stock buyback", "merger", "acquisition", 
            "ipo", "rates", "repo rate", "interest rate", "policy rate",
            "profit", "loss", "revenue", "earnings", "quarterly results"
        ])]

        # spaCy loading with graceful degradation
        self.nlp = None
        self.use_transformer = False
        
        if SPACY_AVAILABLE:
            # Try transformer model first, then small, then blank
            for model in [model_name, "en_core_web_sm", "en_core_web_lg"]:
                try:
                    self.nlp = spacy.load(model)
                    self.use_transformer = ("trf" in model)
                    break
                except Exception:
                    continue
            
            # Fallback to blank if all else fails
            if self.nlp is None:
                self.nlp = spacy.blank("en")

        # PhraseMatcher for precise alias detection
        self.matcher = None
        if self.nlp is not None and self.alias_table:
            try:
                self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
                for canon, meta in self.alias_table.items():
                    aliases = meta.get("aliases", [canon])
                    patterns = [self.nlp.make_doc(a) for a in aliases]
                    self.matcher.add(canon, patterns)
            except Exception:
                pass

    def _as_text(self, item: Union[NewsArticle, str]) -> str:
        """Convert input to text string"""
        if isinstance(item, NewsArticle):
            return f"{item.title}. {item.content}"
        return item

    def _normalize_company_name(self, name: str) -> Optional[str]:
        """Map any company mention to canonical name"""
        normalized = name.strip()
        
        # Remove common prefixes
        for prefix in ["the ", "The "]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        # Check alias lookup
        canonical = self.alias_to_canonical.get(normalized.lower())
        if canonical:
            return canonical
        
        # Try partial matching for known companies (more conservative)
        normalized_lower = normalized.lower()
        for alias, canon in self.alias_to_canonical.items():
            # Only match if the alias is a complete word match
            if normalized_lower == alias:
                return canon
        
        return normalized  # Return as-is if no mapping found

    def _extract_regulators(self, text: str) -> Set[Tuple[str, float, str]]:
        """Extract regulators with confidence scores"""
        regulators = set()
        lower_text = text.lower()
        
        for alias, canonical in self.regulator_map.items():
            # Word boundary matching for precision
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, lower_text):
                # Higher confidence for exact canonical name matches
                confidence = 1.0 if alias == canonical.lower() else 0.95
                regulators.add((canonical, confidence, "regex"))
        
        return regulators

    def _extract_companies_matcher(self, doc) -> Set[Tuple[str, float, str]]:
        """Extract companies using PhraseMatcher (highest precision)"""
        companies = set()
        
        if self.matcher is None:
            return companies
        
        matched_spans = set()
        for match_id, start, end in self.matcher(doc):
            canonical = self.nlp.vocab.strings[match_id]
            
            # Verify this is a valid word-boundary match
            span_text = doc[start:end].text
            
            # Additional validation: check if this matched text is a known alias
            is_valid = False
            for alias in self.alias_table[canonical].get("aliases", [canonical]):
                if span_text.lower() == alias.lower():
                    is_valid = True
                    break
            
            if is_valid:
                matched_spans.add((start, end))
                companies.add((canonical, 1.0, "matcher"))
        
        return companies

    def _extract_companies_ner(self, doc) -> Set[Tuple[str, float, str]]:
        """Extract companies using spaCy NER"""
        companies = set()
        
        for ent in getattr(doc, "ents", []):
            if ent.label_ == "ORG":
                # Normalize to canonical name
                canonical = self._normalize_company_name(ent.text)
                
                # Skip if it's actually a regulator
                if canonical in self.canonical_regulators:
                    continue
                if canonical.lower() in self.regulator_map:
                    continue
                
                # Lower confidence for NER-only matches
                confidence = 0.85 if self.use_transformer else 0.75
                companies.add((canonical, confidence, "ner"))
        
        return companies

    def _extract_companies_regex(self, text: str) -> Set[Tuple[str, float, str]]:
        """Regex-based company detection with strict word boundaries"""
        companies = set()
        
        for canonical, meta in self.alias_table.items():
            for alias in meta.get("aliases", [canonical]):
                # Strict word boundary matching
                # Use \b for word boundaries to avoid partial matches
                pattern = rf"\b{re.escape(alias)}\b"
                if re.search(pattern, text, flags=re.IGNORECASE):
                    companies.add((canonical, 0.95, "regex"))
                    break  # Found this company, move to next
        
        return companies

    def _extract_people(self, doc) -> Set[str]:
        """Extract person names from NER"""
        people = set()
        
        for ent in getattr(doc, "ents", []):
            if ent.label_ == "PERSON":
                # Filter out single-word names (often false positives)
                name = ent.text.strip()
                if len(name.split()) >= 2:
                    people.add(name)
        
        return people

    def _infer_sectors_from_companies(self, companies: Set[str]) -> Set[str]:
        """Infer sectors from identified companies"""
        sectors = set()
        
        for company in companies:
            meta = self.alias_table.get(company)
            if meta and isinstance(meta, dict):
                sector = meta.get("sector")
                if sector:
                    sectors.add(sector)
        
        return sectors

    def _extract_sectors_explicit(self, text: str) -> Set[str]:
        """Detect explicit sector mentions"""
        sectors = set()
        
        # Create word boundary patterns for multi-word sectors
        for sector in self.sector_tickers.keys():
            # Handle both "Banking" and "banking sector"
            patterns = [
                rf"\b{re.escape(sector)}\b",
                rf"\b{re.escape(sector.lower())} sector\b",
            ]
            
            for pattern in patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    sectors.add(sector)
                    break
        
        return sectors

    def _extract_events(self, text: str) -> Set[str]:
        """Extract event keywords with context"""
        events = set()
        lower_text = text.lower()
        
        for keyword in self.event_keywords:
            # Word boundary matching
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, lower_text):
                events.add(keyword)
        
        return events

    def _merge_entities_with_confidence(
        self, 
        entities_with_conf: Set[Tuple[str, float, str]]
    ) -> List[str]:
        """Merge duplicate entities, keeping highest confidence version"""
        entity_map = {}
        
        for entity, confidence, source in entities_with_conf:
            if entity not in entity_map or confidence > entity_map[entity][0]:
                entity_map[entity] = (confidence, source)
        
        return sorted(entity_map.keys())

    def extract_entities(
        self, 
        item: Union[NewsArticle, str],
        return_confidence: bool = False
    ) -> Union[Dict[str, List[str]], Dict[str, List[EntityConfidence]]]:
        """
        Extract entities from text
        
        Args:
            item: NewsArticle or text string
            return_confidence: If True, return EntityConfidence objects with scores
        
        Returns:
            Dictionary of entity types to lists of entities (or EntityConfidence objects)
        """
        text = self._as_text(item)
        
        # Initialize result containers
        companies_with_conf = set()
        regulators_with_conf = set()
        sectors = set()
        people = set()
        events = set()

        # Extract regulators (always regex-based for reliability)
        regulators_with_conf = self._extract_regulators(text)

        # Extract companies using multiple methods
        if self.nlp is not None:
            doc = self.nlp(text)
            
            # Method 1: PhraseMatcher (highest precision)
            matcher_companies = self._extract_companies_matcher(doc)
            companies_with_conf.update(matcher_companies)
            
            # Method 2: Regex (high precision with word boundaries)
            # Only use if matcher didn't find much
            if len(matcher_companies) < 2:
                regex_companies = self._extract_companies_regex(text)
                companies_with_conf.update(regex_companies)
            
            # Method 3: NER (good recall, but only if we haven't found via other methods)
            ner_companies = self._extract_companies_ner(doc)
            # Only add NER results that weren't found by matcher/regex
            existing_company_names = {c for c, _, _ in companies_with_conf}
            for company, conf, source in ner_companies:
                if company not in existing_company_names:
                    companies_with_conf.add((company, conf, source))
            
            # Extract people
            people = self._extract_people(doc)
        else:
            # Fallback: regex-based extraction
            companies_with_conf = self._extract_companies_regex(text)

        # Merge companies, removing duplicates and regulator conflicts
        companies_raw = self._merge_entities_with_confidence(companies_with_conf)
        
        # Final cleanup: remove any companies that are actually regulators
        regulator_names = {reg for reg, _, _ in regulators_with_conf}
        companies = [c for c in companies_raw if c not in regulator_names]

        # Extract sectors
        # 1. Infer from companies
        sectors.update(self._infer_sectors_from_companies(set(companies)))
        # 2. Explicit sector mentions
        sectors.update(self._extract_sectors_explicit(text))

        # Extract events
        events = self._extract_events(text)

        # Format output
        regulators = self._merge_entities_with_confidence(regulators_with_conf)
        
        if return_confidence:
            # Return with confidence scores
            return {
                "Companies": [
                    EntityConfidence(entity=e, entity_type="Company", confidence=c, source=s)
                    for e, c, s in companies_with_conf if e in companies
                ],
                "Sectors": [
                    EntityConfidence(entity=s, entity_type="Sector", confidence=0.9, source="inferred")
                    for s in sorted(sectors)
                ],
                "Regulators": [
                    EntityConfidence(entity=r, entity_type="Regulator", confidence=c, source=s)
                    for r, c, s in regulators_with_conf
                ],
                "People": [
                    EntityConfidence(entity=p, entity_type="Person", confidence=0.85, source="ner")
                    for p in sorted(people)
                ],
                "Events": [
                    EntityConfidence(entity=e, entity_type="Event", confidence=0.8, source="keyword")
                    for e in sorted(events)
                ],
            }
        else:
            # Return simple lists (backward compatible)
            return {
                "Companies": companies,
                "Sectors": sorted(sectors),
                "Regulators": regulators,
                "People": sorted(people),
                "Events": sorted(events),
            }


# Example usage and testing
if __name__ == "__main__":
    extractor = EntityExtractor()
    
    # Test cases from problem statement
    test_cases = [
        "RBI raises repo rate by 25 basis points to combat inflation",
        "HDFC Bank announces 15% dividend, board approves stock buyback",
        "Banking sector NPAs decline to 5-year low, credit growth at 16%",
        "Reserve Bank hikes interest rates by 0.25% in surprise move",
        "ICICI Bank opens 500 new branches across Tier-2 cities",
    ]
    
    print("=" * 80)
    print("Entity Extraction Tests")
    print("=" * 80)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n[Test {i}] {text}")
        entities = extractor.extract_entities(text)
        
        for entity_type, values in entities.items():
            if values:
                print(f"  {entity_type}: {values}")
        
        # Test with confidence scores
        entities_conf = extractor.extract_entities(text, return_confidence=True)
        print("\n  With Confidence:")
        for entity_type, conf_list in entities_conf.items():
            if conf_list:
                for ec in conf_list:
                    print(f"    {ec.entity} ({ec.confidence:.2f}, {ec.source})")