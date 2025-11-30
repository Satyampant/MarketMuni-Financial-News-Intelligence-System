from __future__ import annotations
from pathlib import Path
import json
import re
from typing import Dict, List, Union, Optional
from news_storage import NewsArticle  

try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

MODULE_DIR = Path(__file__).parent

class EntityExtractor:
    def __init__(
        self,
        alias_path: Optional[Union[str, Path]] = None,
        sector_path: Optional[Union[str, Path]] = None,
        regulator_path: Optional[Union[str, Path]] = None,
        model_name: str = "en_core_web_trf",
        event_keywords: Optional[List[str]] = None,
    ):
        # portable defaults (module-relative)
        alias_path = Path(alias_path) if alias_path else MODULE_DIR / "company_aliases.json"
        sector_path = Path(sector_path) if sector_path else MODULE_DIR / "sector_tickers.json"

        self.alias_table = {}
        if alias_path.exists():
            self.alias_table = json.loads(alias_path.read_text())
        self.sector_tickers = {}
        if sector_path.exists():
            self.sector_tickers = json.loads(sector_path.read_text())

        # regulators canonical map
        regulator_path = Path(regulator_path) if regulator_path else MODULE_DIR / "regulators.json"

        # Load Regulators
        self.regulator_map = {} # Maps alias -> Canonical Name
        if regulator_path.exists():
            reg_data = json.loads(regulator_path.read_text())
            for canonical, aliases in reg_data.items():
                for alias in aliases:
                    self.regulator_map[alias.lower()] = canonical
        else:
             # Fallback if file missing
             self.regulator_map = {"rbi": "RBI", "sebi": "SEBI"}

        # event keywords (lowercased)
        self.event_keywords = [k.lower() for k in (event_keywords or [
            "dividend", "buyback", "merger", "acquisition", "ipo", "rates", "repo rate", "profit", "loss"
        ])]

        # spaCy loading (preferred transformer -> small -> blank)
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
            except Exception:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    self.nlp = spacy.blank("en")

        # PhraseMatcher for alias detection (only if nlp has vocab)
        self.matcher = None
        if self.nlp is not None and self.alias_table:
            try:
                self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
                for canon, meta in self.alias_table.items():
                    aliases = meta.get("aliases", [canon])
                    patterns = [self.nlp.make_doc(a) for a in aliases]
                    self.matcher.add(canon, patterns)
            except Exception:
                self.matcher = None  # safe fallback

    def _as_text(self, item: Union[NewsArticle, str]) -> str:
        if isinstance(item, NewsArticle):
            return f"{item.title}. {item.content}"
        return item

    def extract_entities(self, item: Union[NewsArticle, str]) -> Dict[str, List[str]]:
        text = self._as_text(item)
        lower_text = text.lower()

        companies = set()
        sectors = set()
        regulators = set()
        people = set()
        events = set()

        # --- 1. Regulator Detection ---
        for alias, canonical in self.regulator_map.items():
            if re.search(rf"\b{re.escape(alias)}\b", lower_text):
                regulators.add(canonical)

        # --- 2. Company Detection ---
        if self.nlp is not None:
            doc = self.nlp(text)
            
            # A. Matcher
            if self.matcher is not None:
                for match_id, start, end in self.matcher(doc):
                    canon = self.nlp.vocab.strings[match_id]
                    companies.add(canon)
            
            # B. Standard NER (ORG)
            for ent in getattr(doc, "ents", []):
                if ent.label_ == "ORG":
                    ent_text = ent.text.strip()
                    mapped = None
                    for canon, meta in self.alias_table.items():
                        aliases = [a.lower() for a in meta.get("aliases", [canon])]
                        if ent_text.lower() in aliases:
                            mapped = canon
                            break
                    companies.add(mapped or ent_text)

                if ent.label_ == "PERSON":
                    people.add(ent.text.strip())
        else:
            # Fallback regex detection
            for canon, meta in self.alias_table.items():
                for a in meta.get("aliases", [canon]):
                    if re.search(rf"\b{re.escape(a)}\b", text, flags=re.IGNORECASE):
                        companies.add(canon)
                        break

        # --- 3. CLEANUP: Set Difference (Improved) ---
        final_companies = set()
        
        for co in companies:
            # Normalize for comparison: lower case, strip whitespace
            norm_co = co.lower().strip()
            
            # Remove "the " prefix if it exists (common in spaCy ORG entities)
            if norm_co.startswith("the "):
                norm_co = norm_co[4:].strip()

            # Check 1: Is this explicitly a known regulator alias?
            # (e.g. checks "reserve bank of india" against regulator map)
            if norm_co in self.regulator_map:
                continue
                
            # Check 2: Does it map to a Canonical Regulator we already found?
            # If map["reserve bank of india"] -> "RBI", and we found "RBI" in Step 1
            canonical_reg = self.regulator_map.get(norm_co)
            if canonical_reg and canonical_reg in regulators:
                continue

            # Check 3: Is the Company Name exactly the same as a found Regulator?
            # (e.g. If spaCy extracted "RBI" as ORG)
            if co in regulators:
                continue

            final_companies.add(co)

        # 4) Sector detection
        tokens = re.findall(r"\b\w+\b", text)
        token_set = set(tokens)
        for sector in self.sector_tickers.keys():
            if len(sector) <= 2:
                if sector in token_set:
                    sectors.add(sector)
            else:
                if re.search(rf"\b{re.escape(sector)}\b", text, flags=re.IGNORECASE):
                    sectors.add(sector)

        # 5) Event detection
        for kw in self.event_keywords:
            if re.search(rf"\b{re.escape(kw)}\b", lower_text):
                events.add(kw)

        return {
            "Companies": sorted(final_companies),  
            "Sectors": sorted(sectors),
            "Regulators": sorted(regulators),
            "People": sorted(people),
            "Events": sorted(events),
        }