from app.core.config import Paths
from typing import List, Dict, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import difflib

from app.core.models import NewsArticle
from app.core.config_loader import get_config

@dataclass
class StockImpact:
    symbol: str
    confidence: float
    impact_type: str  # direct | sector | regulatory
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
        config = get_config()
        self.fuzzy_threshold = config.stock_impact.fuzzy_match_threshold
        alias_path = alias_path or Paths.COMPANY_ALIASES
        sector_path = sector_path or Paths.SECTOR_TICKERS
        regulators_path = regulators_path or Paths.REGULATORS
        regulator_impact_path = regulator_impact_path or Paths.REGULATOR_IMPACT

        # Load Company Aliases
        self.aliases = {}
        if alias_path.exists():
            try:
                self.aliases = json.loads(alias_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Build case-insensitive ticker map
        self.co_map: Dict[str, str] = {}
        if isinstance(self.aliases, dict):
            for canon, data in self.aliases.items():
                if not isinstance(data, dict):
                    continue
                if ticker := data.get("ticker"):
                    self.co_map[canon.lower()] = ticker
                    for alias in data.get("aliases", []):
                        self.co_map[alias.lower()] = ticker

        # Load Sectors
        self.sectors: Dict[str, List[str]] = {}
        self.sector_map_lower: Dict[str, str] = {}
        
        if sector_path.exists():
            try:
                self.sectors = json.loads(sector_path.read_text(encoding="utf-8"))
                for sector_name in self.sectors.keys():
                    self.sector_map_lower[sector_name.lower()] = sector_name
            except Exception:
                pass

        # Load Regulators
        self.regulator_alias_map: Dict[str, str] = {}
        
        if regulators_path.exists():
            try:
                raw = json.loads(regulators_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    for canon, aliases in raw.items():
                        self.regulator_alias_map[canon.lower()] = canon
                        for alias in aliases:
                            self.regulator_alias_map[alias.lower()] = canon
            except Exception:
                pass

        # Load Regulator -> Sector Impact Mapping
        self.reg_impact_map: Dict[str, List[str]] = {}
        
        if regulator_impact_path.exists():
            try:
                self.reg_impact_map = json.loads(regulator_impact_path.read_text(encoding="utf-8"))
            except Exception:
                # Fallback defaults
                self.reg_impact_map = {
                    "RBI": ["Banking", "Finance"],
                    "SEBI": ["Banking", "Finance", "IT", "Auto", "Energy"],
                    "IRDAI": ["Insurance"],
                    "US FDA": ["Pharma"],
                    "DOT": ["Telecom"],
                    "TRAI": ["Telecom"]
                }

    def _best_ticker_for_company(self, company: str) -> Optional[Dict[str, Any]]:
        """Find best matching ticker using exact or fuzzy matching."""
        if not company:
            return None
        
        key = company.lower().strip()
        if key.startswith("the "):
            key = key[4:].strip()
        
        # 1. Exact match
        if key in self.co_map:
            return {
                "ticker": self.co_map[key],
                "rule": "exact_alias",
                "score": 1.0
            }
        
        # 2. Fuzzy match
        names = list(self.co_map.keys())
        matches = difflib.get_close_matches(key, names, n=1, cutoff=self.fuzzy_threshold)
        
        if matches:
            ratio = difflib.SequenceMatcher(None, key, matches[0]).ratio()
            return {
                "ticker": self.co_map[matches[0]],
                "rule": "fuzzy_alias",
                "score": round(ratio, 2)
            }
        
        return None

    @staticmethod
    def _combine_confidences(existing: float, new: float) -> float:
        """Combine probabilities: P(A or B) = 1 - (1-A)*(1-B)."""
        return round(1.0 - (1.0 - existing) * (1.0 - new), 3)

    @staticmethod
    def _impact_type_precedence(type1: str, type2: str) -> str:
        """Priority: Direct > Sector > Regulatory."""
        precedence = {"direct": 3, "sector": 2, "regulatory": 1}
        score1 = precedence.get(type1, 0)
        score2 = precedence.get(type2, 0)
        return type1 if score1 >= score2 else type2

    def _get_sector_tickers(self, sector_name: str) -> List[str]:
        if sector_name in self.sectors:
            return self.sectors[sector_name]
        
        # Case-insensitive fallback
        original_case = self.sector_map_lower.get(sector_name.lower())
        return self.sectors[original_case] if original_case else []

    def map_to_stocks(self, entities: Dict[str, List[str]]) -> List[StockImpact]:
        """
        Map extracted entities (Companies, Sectors, Regulators) to stock symbols.
        Returns deduplicated list sorted by confidence.
        """
        impacts_by_symbol: Dict[str, StockImpact] = {}

        def add_impact(symbol: str, conf: float, itype: str, source: str, rule: str):
            existing = impacts_by_symbol.get(symbol)
            
            if existing is None:
                impacts_by_symbol[symbol] = StockImpact(
                    symbol=symbol,
                    confidence=round(conf, 3),
                    impact_type=itype,
                    source_entity=source,
                    rule=rule
                )
            else:
                # Merge strategy: Combine confidence, take highest precedence type, union sources
                combined_conf = self._combine_confidences(existing.confidence, conf)
                best_type = self._impact_type_precedence(existing.impact_type, itype)
                
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

        # 1. Direct Company Mentions
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

        # 2. Sector-Wide Impact (Fixed confidence: 0.7)
        for sector in entities.get("Sectors", []):
            for ticker in self._get_sector_tickers(sector):
                add_impact(
                    symbol=ticker,
                    conf=0.7,
                    itype="sector",
                    source=f"Sector:{sector}",
                    rule="sector_match"
                )

        # 3. Regulatory Impact (Fixed confidence: 0.5)
        for regulator in entities.get("Regulators", []):
            canonical = self.regulator_alias_map.get(regulator.lower(), regulator)
            
            for sector_name in self.reg_impact_map.get(canonical, []):
                for ticker in self._get_sector_tickers(sector_name):
                    add_impact(
                        symbol=ticker,
                        conf=0.5,
                        itype="regulatory",
                        source=f"Regulator:{canonical}",
                        rule=f"regulatory:{canonical}→{sector_name}"
                    )

        return sorted(impacts_by_symbol.values(), key=lambda x: (-x.confidence, x.symbol))

    def map_article(
        self,
        article: NewsArticle,
        extractor: Optional[Any] = None
    ) -> NewsArticle:
        """Orchestrate entity extraction (if needed) and stock mapping."""
        entities = {}
        
        if extractor is not None:
            # Context extraction from title + content
            text = f"{article.title}. {article.content}"
            entities = extractor.extract_entities(text)
        elif hasattr(article, "entities") and article.entities:
            entities = article.entities
        
        if entities:
            impacts = self.map_to_stocks(entities)
            article.impacted_stocks = [imp.to_dict() for imp in impacts]
        
        return article