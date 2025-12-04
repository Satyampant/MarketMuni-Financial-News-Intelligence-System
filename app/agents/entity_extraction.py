from __future__ import annotations
from app.core.config import Paths
from pathlib import Path
import json
import re
from typing import Dict, List, Union, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from app.core.models import NewsArticle
from app.core.config_loader import get_config

try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False


@dataclass
class EntityConfidence:
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
        model_name: str = None,
        event_keywords: Optional[List[str]] = None,
    ):
        config = get_config()
    
        model_name = model_name or config.entity_extraction.spacy_model
        event_keywords = event_keywords or config.entity_extraction.event_keywords

        alias_path = Path(alias_path) if alias_path else Paths.COMPANY_ALIASES
        sector_path = Path(sector_path) if sector_path else Paths.SECTOR_TICKERS
        regulator_path = Path(regulator_path) if regulator_path else Paths.REGULATORS

        self.alias_table = {}
        if alias_path.exists():
            self.alias_table = json.loads(alias_path.read_text())
        
        # Reverse lookup: alias -> canonical name
        self.alias_to_canonical = {}
        for canonical, meta in self.alias_table.items():
            for alias in meta.get("aliases", [canonical]):
                self.alias_to_canonical[alias.lower()] = canonical

        self.sector_tickers = {}
        if sector_path.exists():
            self.sector_tickers = json.loads(sector_path.read_text())

        # Regulator bidirectional mapping
        self.regulator_map = {}
        self.canonical_regulators = set()
        if regulator_path.exists():
            reg_data = json.loads(regulator_path.read_text())
            for canonical, aliases in reg_data.items():
                self.canonical_regulators.add(canonical)
                for alias in aliases:
                    self.regulator_map[alias.lower()] = canonical

        self.event_keywords = [k.lower() for k in (event_keywords or [
            "dividend", "buyback", "stock buyback", "merger", "acquisition", 
            "ipo", "rates", "repo rate", "interest rate", "policy rate",
            "profit", "loss", "revenue", "earnings", "quarterly results",
            "rights issue", "delisting", "bonus", "split", "consolidation", 
            "restructuring", "capex", "guidance", "outlook", "forecast",
            "default", "downgrade", "upgrade", "rating", "credit rating",
            "stake sale", "divestment", "spin-off", "demerger", 
            "capacity addition", "plant closure", "debt restructuring",
            "working capital", "cash flow", "free cash flow"
        ])]

        self.nlp = None
        self.use_transformer = False
        
        # Load spaCy (Priority: Transformer -> Small -> Blank)
        if SPACY_AVAILABLE:
            for model in [model_name, "en_core_web_sm", "en_core_web_lg"]:
                try:
                    self.nlp = spacy.load(model)
                    self.use_transformer = ("trf" in model)
                    break
                except Exception:
                    continue
            
            if self.nlp is None:
                self.nlp = spacy.blank("en")

        # Setup PhraseMatcher for precise alias detection
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
        if isinstance(item, NewsArticle):
            return f"{item.title}. {item.content}"
        return item

    def _normalize_company_name(self, name: str) -> Optional[str]:
        """Map mention to canonical name, handling prefixes and exact matches."""
        normalized = name.strip()
        
        for prefix in ["the ", "The "]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        canonical = self.alias_to_canonical.get(normalized.lower())
        if canonical:
            return canonical
        
        # Conservative partial matching
        normalized_lower = normalized.lower()
        for alias, canon in self.alias_to_canonical.items():
            if normalized_lower == alias:
                return canon
        
        return normalized

    def _extract_regulators(self, text: str) -> Set[Tuple[str, float, str]]:
        regulators = set()
        lower_text = text.lower()
        
        for alias, canonical in self.regulator_map.items():
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, lower_text):
                # Higher confidence for exact canonical matches
                confidence = 1.0 if alias == canonical.lower() else 0.95
                regulators.add((canonical, confidence, "regex"))
        
        return regulators

    def _extract_companies_matcher(self, doc) -> Set[Tuple[str, float, str]]:
        """PhraseMatcher extraction (highest precision)."""
        companies = set()
        
        if self.matcher is None:
            return companies
        
        matched_spans = set()
        for match_id, start, end in self.matcher(doc):
            canonical = self.nlp.vocab.strings[match_id]
            span_text = doc[start:end].text
            
            # Double-check against known aliases for validity
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
        """SpaCy NER extraction with regulator filtering."""
        companies = set()
        
        for ent in getattr(doc, "ents", []):
            if ent.label_ == "ORG":
                canonical = self._normalize_company_name(ent.text)
                
                if canonical in self.canonical_regulators:
                    continue
                if canonical.lower() in self.regulator_map:
                    continue
                
                confidence = 0.85 if self.use_transformer else 0.75
                companies.add((canonical, confidence, "ner"))
        
        return companies

    def _extract_companies_regex(self, text: str) -> Set[Tuple[str, float, str]]:
        """Strict word-boundary regex fallback."""
        companies = set()
        
        for canonical, meta in self.alias_table.items():
            for alias in meta.get("aliases", [canonical]):
                pattern = rf"\b{re.escape(alias)}\b"
                if re.search(pattern, text, flags=re.IGNORECASE):
                    companies.add((canonical, 0.95, "regex"))
                    break 
        
        return companies

    def _extract_people(self, doc) -> Set[str]:
        people = set()
        
        for ent in getattr(doc, "ents", []):
            if ent.label_ == "PERSON":
                # Filter single-word names to reduce false positives
                name = ent.text.strip()
                if len(name.split()) >= 2:
                    people.add(name)
        
        return people

    def _infer_sectors_from_companies(self, companies: Set[str]) -> Set[str]:
        sectors = set()
        
        for company in companies:
            meta = self.alias_table.get(company)
            if meta and isinstance(meta, dict):
                sector = meta.get("sector")
                if sector:
                    sectors.add(sector)
        
        return sectors

    def _extract_sectors_explicit(self, text: str) -> Set[str]:
        sectors = set()
        
        for sector in self.sector_tickers.keys():
            patterns = [
                rf"\b{re.escape(sector)}\b",
                rf"\b{re.escape(sector.lower())} sector\b",
            ]
            # Handle singular forms (e.g., "Semiconductors" -> "semiconductor")
            if sector.endswith('s'):
                singular = sector[:-1]
                patterns.append(rf"\b{re.escape(singular)}\b")
                patterns.append(rf"\b{re.escape(singular.lower())} sector\b")

            for pattern in patterns:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    sectors.add(sector)
                    break
        
        return sectors

    def _extract_events(self, text: str) -> Set[str]:
        events = set()
        lower_text = text.lower()
        
        for keyword in self.event_keywords:
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, lower_text):
                events.add(keyword)
        
        return events

    def _merge_entities_with_confidence(self, entities_with_conf: Set[Tuple[str, float, str]]) -> List[str]:
        """Deduplicate entities, keeping highest confidence source."""
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
        Extract entities via pipeline: Matcher -> Regex -> NER.
        """
        text = self._as_text(item)
        
        companies_with_conf = set()
        regulators_with_conf = set()
        sectors = set()
        people = set()
        events = set()

        regulators_with_conf = self._extract_regulators(text)

        if self.nlp is not None:
            doc = self.nlp(text)
            
            # 1. Precise Matcher
            matcher_companies = self._extract_companies_matcher(doc)
            companies_with_conf.update(matcher_companies)
            
            # 2. Regex Fallback (if insufficient matches)
            if len(matcher_companies) < 2:
                regex_companies = self._extract_companies_regex(text)
                companies_with_conf.update(regex_companies)
            
            # 3. NER (Catch-all for unknown entities)
            ner_companies = self._extract_companies_ner(doc)
            existing_company_names = {c for c, _, _ in companies_with_conf}
            for company, conf, source in ner_companies:
                if company not in existing_company_names:
                    companies_with_conf.add((company, conf, source))
            
            people = self._extract_people(doc)
        else:
            companies_with_conf = self._extract_companies_regex(text)

        # Merge and remove regulator conflicts
        companies_raw = self._merge_entities_with_confidence(companies_with_conf)
        regulator_names = {reg for reg, _, _ in regulators_with_conf}
        companies = [c for c in companies_raw if c not in regulator_names]

        # Extract Sectors and Events
        sectors.update(self._infer_sectors_from_companies(set(companies)))
        sectors.update(self._extract_sectors_explicit(text))
        events = self._extract_events(text)

        regulators = self._merge_entities_with_confidence(regulators_with_conf)
        
        if return_confidence:
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
            return {
                "Companies": companies,
                "Sectors": sorted(sectors),
                "Regulators": regulators,
                "People": sorted(people),
                "Events": sorted(events),
            }