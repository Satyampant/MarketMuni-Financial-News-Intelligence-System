from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import difflib

# Import from storage to avoid circular dependency
from news_storage import NewsArticle

MODULE_DIR = Path(__file__).parent


@dataclass
class StockImpact:
    symbol: str
    confidence: float
    impact_type: str  # "direct" | "sector" | "regulatory"
    source_entity: Optional[str] = None
    rule: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StockImpactMapper:
    def __init__(
        self,
        alias_path: Optional[Path] = None,
        sector_path: Optional[Path] = None,
        regulators_path: Optional[Path] = None,
        regulator_impact_path: Optional[Path] = None,
    ):
        alias_path = alias_path or MODULE_DIR / "company_aliases.json"
        sector_path = sector_path or MODULE_DIR / "sector_tickers.json"
        regulators_path = regulators_path or MODULE_DIR / "regulators.json"
        regulator_impact_path = regulator_impact_path or MODULE_DIR / "regulator_sector_impact.json"

        # 1. Load Company Aliases
        self.aliases = {}
        if alias_path.exists():
            try:
                self.aliases = json.loads(alias_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Build company -> ticker mapping (case-insensitive)
        self.co_map: Dict[str, str] = {}
        if isinstance(self.aliases, dict):
            for canon, data in self.aliases.items():
                if not isinstance(data, dict):
                    continue
                ticker = data.get("ticker")
                if not ticker:
                    continue
                
                # Add canonical name
                self.co_map[canon.lower()] = ticker
                
                # Add all aliases
                for alias in data.get("aliases", []):
                    self.co_map[alias.lower()] = ticker

        # 2. Load Sectors
        self.sectors: Dict[str, List[str]] = {}
        self.sector_map_lower: Dict[str, str] = {}  # lowercase sector -> original case
        
        if sector_path.exists():
            try:
                self.sectors = json.loads(sector_path.read_text(encoding="utf-8"))
                # Build case-insensitive lookup
                for sector_name in self.sectors.keys():
                    self.sector_map_lower[sector_name.lower()] = sector_name
            except Exception:
                pass

        # 3. Load Regulators
        self.regulator_alias_map: Dict[str, str] = {}  # alias -> canonical
        
        if regulators_path.exists():
            try:
                raw = json.loads(regulators_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    for canon, aliases in raw.items():
                        # Add canonical name
                        self.regulator_alias_map[canon.lower()] = canon
                        # Add all aliases
                        for alias in aliases:
                            self.regulator_alias_map[alias.lower()] = canon
            except Exception:
                pass

        # 4. Load Regulator -> Sector Impact Mapping
        self.reg_impact_map: Dict[str, List[str]] = {}
        
        if regulator_impact_path.exists():
            try:
                self.reg_impact_map = json.loads(regulator_impact_path.read_text(encoding="utf-8"))
            except Exception:
                # Fallback to basic mapping if file doesn't exist
                self.reg_impact_map = {
                    "RBI": ["Banking", "Finance"],
                    "SEBI": ["Banking", "Finance", "IT", "Auto", "Energy"],
                    "IRDAI": ["Insurance"],
                    "US FDA": ["Pharma"],
                    "DOT": ["Telecom"],
                    "TRAI": ["Telecom"]
                }

    def _best_ticker_for_company(self, company: str) -> Optional[Dict[str, Any]]:
        """Find the best matching ticker for a company name with confidence scoring."""
        if not company:
            return None
        
        key = company.lower().strip()
        
        # Remove common prefixes
        if key.startswith("the "):
            key = key[4:].strip()
        
        # 1. Exact match (highest confidence)
        if key in self.co_map:
            return {
                "ticker": self.co_map[key],
                "rule": "exact_alias",
                "score": 1.0
            }
        
        # 2. Fuzzy match with high threshold
        names = list(self.co_map.keys())
        matches = difflib.get_close_matches(key, names, n=1, cutoff=0.80)
        
        if matches:
            matched = matches[0]
            ratio = difflib.SequenceMatcher(None, key, matched).ratio()
            return {
                "ticker": self.co_map[matched],
                "rule": "fuzzy_alias",
                "score": round(ratio, 2)
            }
        
        return None

    @staticmethod
    def _combine_confidences(existing: float, new: float) -> float:
        """
        Combine two confidence scores using probability theory.
        P(A or B) = 1 - (1-A)*(1-B)
        This ensures the combined confidence is always higher than individual ones.
        """
        return round(1.0 - (1.0 - existing) * (1.0 - new), 3)

    @staticmethod
    def _impact_type_precedence(type1: str, type2: str) -> str:
        """
        Determine which impact type takes precedence.
        Direct > Sector > Regulatory
        """
        precedence = {"direct": 3, "sector": 2, "regulatory": 1}
        score1 = precedence.get(type1, 0)
        score2 = precedence.get(type2, 0)
        return type1 if score1 >= score2 else type2

    def _get_sector_tickers(self, sector_name: str) -> List[str]:
        """Get tickers for a sector (case-insensitive)."""
        # Try exact match first
        if sector_name in self.sectors:
            return self.sectors[sector_name]
        
        # Try case-insensitive match
        sector_lower = sector_name.lower()
        original_case = self.sector_map_lower.get(sector_lower)
        
        if original_case:
            return self.sectors[original_case]
        
        return []

    def map_to_stocks(self, entities: Dict[str, List[str]]) -> List[StockImpact]:
        """
        Map extracted entities to stock symbols with confidence scores.
        
        Args:
            entities: Dict with keys: Companies, Sectors, Regulators, People, Events
            
        Returns:
            List of StockImpact objects with deduplication and confidence combination
        """
        impacts_by_symbol: Dict[str, StockImpact] = {}

        def add_impact(symbol: str, conf: float, itype: str, source: str, rule: str):
            """Add or merge a stock impact."""
            existing = impacts_by_symbol.get(symbol)
            
            if existing is None:
                # New impact
                impacts_by_symbol[symbol] = StockImpact(
                    symbol=symbol,
                    confidence=round(conf, 3),
                    impact_type=itype,
                    source_entity=source,
                    rule=rule
                )
            else:
                # Merge with existing impact
                combined_conf = self._combine_confidences(existing.confidence, conf)
                best_type = self._impact_type_precedence(existing.impact_type, itype)
                
                # Combine sources intelligently (avoid duplication)
                sources = set(existing.source_entity.split(" | ")) if existing.source_entity else set()
                sources.add(source)
                
                rules = set(existing.rule.split(" | ")) if existing.rule else set()
                rules.add(rule)
                
                impacts_by_symbol[symbol] = StockImpact(
                    symbol=symbol,
                    confidence=combined_conf,
                    impact_type=best_type,
                    source_entity=" | ".join(sorted(sources)),
                    rule=" | ".join(sorted(rules))
                )

        # 1. Process Direct Company Mentions (Confidence: variable based on match quality)
        for company in entities.get("Companies", []):
            match_result = self._best_ticker_for_company(company)
            if match_result:
                add_impact(
                    symbol=match_result["ticker"],
                    conf=match_result["score"],
                    itype="direct",
                    source=f"Company:{company}",
                    rule=match_result["rule"]
                )

        # 2. Process Sector-Wide Impact (Confidence: 0.7)
        for sector in entities.get("Sectors", []):
            tickers = self._get_sector_tickers(sector)
            
            for ticker in tickers:
                add_impact(
                    symbol=ticker,
                    conf=0.7,
                    itype="sector",
                    source=f"Sector:{sector}",
                    rule="sector_match"
                )

        # 3. Process Regulatory Impact (Confidence: 0.5)
        for regulator in entities.get("Regulators", []):
            # Normalize regulator name via alias map
            canonical = self.regulator_alias_map.get(regulator.lower(), regulator)
            
            # Find impacted sectors
            impacted_sectors = self.reg_impact_map.get(canonical, [])
            
            for sector_name in impacted_sectors:
                tickers = self._get_sector_tickers(sector_name)
                
                for ticker in tickers:
                    add_impact(
                        symbol=ticker,
                        conf=0.5,
                        itype="regulatory",
                        source=f"Regulator:{canonical}",
                        rule=f"regulatory:{canonical}→{sector_name}"
                    )

        # Sort by confidence (descending) for consistent output
        result = sorted(
            impacts_by_symbol.values(),
            key=lambda x: (-x.confidence, x.symbol)
        )
        
        return result

    def map_article(
        self,
        article: NewsArticle,
        extractor: Optional[Any] = None
    ) -> NewsArticle:
        """
        Map entities to stocks and update the article.impacted_stocks field.
        
        Args:
            article: NewsArticle object to process
            extractor: Optional EntityExtractor instance for entity extraction
            
        Returns:
            Updated NewsArticle with impacted_stocks populated
        """
        entities = {}
        
        # Extract entities if extractor provided
        if extractor is not None:
            # Combine title and content for better context
            text = f"{article.title}. {article.content}"
            entities = extractor.extract_entities(text)
        # Or use pre-existing entities attribute
        elif hasattr(article, "entities") and article.entities:
            entities = article.entities
        
        # Map entities to stock impacts
        if entities:
            impacts = self.map_to_stocks(entities)
            # Convert to dict format for storage
            article.impacted_stocks = [imp.to_dict() for imp in impacts]
        
        return article