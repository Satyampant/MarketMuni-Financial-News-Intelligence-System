from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import difflib

# Import from storage to avoid circular dependency
from news_storage import NewsArticle, StockImpact

MODULE_DIR = Path(__file__).parent

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
            except Exception: pass

        self.co_map: Dict[str, str] = {}
        if isinstance(self.aliases, dict):
            for canon, data in self.aliases.items():
                ticker = data.get("ticker") if isinstance(data, dict) else data
                if not ticker: continue
                self.co_map[canon.lower()] = ticker
                if isinstance(data, dict):
                    for a in (data.get("aliases") or []):
                        self.co_map[a.lower()] = ticker

        # 2. Load Sectors
        self.sectors: Dict[str, List[str]] = {}
        if sector_path.exists():
            try:
                self.sectors = json.loads(sector_path.read_text(encoding="utf-8"))
            except Exception: pass

        # 3. Load Regulators
        self.regulator_alias_map: Dict[str, str] = {}
        if regulators_path.exists():
            try:
                raw = json.loads(regulators_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    for canon, aliases in raw.items():
                        self.regulator_alias_map[canon.lower()] = canon
                        for a in aliases:
                            self.regulator_alias_map[a.lower()] = canon
            except Exception: pass

        # Load regulator sector impact
        self.reg_impact_map = {}
        # Override with file if it exists
        if regulator_impact_path.exists():
            try:
                file_map = json.loads(regulator_impact_path.read_text(encoding="utf-8"))
                self.reg_impact_map.update(file_map)
            except Exception: pass

    def _best_ticker_for_company(self, company: str) -> Optional[Dict[str, Any]]:
        if not company: return None
        key = company.lower().strip()
        
        # Exact match
        if key in self.co_map:
            return {"ticker": self.co_map[key], "rule": "exact_alias", "score": 1.0}
            
        # Fuzzy match
        names = list(self.co_map.keys())
        matches = difflib.get_close_matches(key, names, n=1, cutoff=0.85) # Stricter cutoff
        if matches:
            matched = matches[0]
            ratio = difflib.SequenceMatcher(None, key, matched).ratio()
            return {"ticker": self.co_map[matched], "rule": "fuzzy_alias", "score": round(ratio, 2)}
        return None

    @staticmethod
    def _combine_probs(p_old: float, p_new: float) -> float:
        # P(A or B) = 1 - (1-A)*(1-B)
        return 1.0 - (1.0 - p_old) * (1.0 - p_new)

    @staticmethod
    def _impact_type_precedence(existing: str, incoming: str) -> str:
        order = {"direct": 3, "sector": 2, "regulatory": 1}
        return existing if order.get(existing, 0) >= order.get(incoming, 0) else incoming

    def map_to_stocks(self, entities: Dict[str, List[str]]) -> List[StockImpact]:
        impacts_by_symbol: Dict[str, StockImpact] = {}

        def add(symbol: str, conf: float, itype: str, source: str, rule: str):
            prev = impacts_by_symbol.get(symbol)
            if prev is None:
                impacts_by_symbol[symbol] = StockImpact(
                    symbol=symbol, confidence=float(conf), impact_type=itype,
                    source_entity=source, rule=rule
                )
            else:
                combined_conf = self._combine_probs(prev.confidence, float(conf))
                new_type = self._impact_type_precedence(prev.impact_type, itype)
                impacts_by_symbol[symbol] = StockImpact(
                    symbol=symbol, confidence=combined_conf, impact_type=new_type,
                    source_entity=f"{prev.source_entity} | {source}",
                    rule=f"{prev.rule} | {rule}"
                )

        # 1. Direct Companies
        for comp in entities.get("Companies", []):
            res = self._best_ticker_for_company(comp)
            if res:
                add(res["ticker"], res["score"], "direct", f"Company:{comp}", res["rule"])

        # 2. Sectors
        for sector in entities.get("Sectors", []):
            # Try exact match first
            tickers = self.sectors.get(sector)
            if not tickers:
                # Fallback to simple case-insensitive match
                for s_key, s_tickers in self.sectors.items():
                    if sector.lower() == s_key.lower():
                        tickers = s_tickers
                        break
            
            if tickers:
                for t in tickers:
                    add(t, 0.7, "sector", f"Sector:{sector}", "sector_match")

        # 3. Regulators
        for reg in entities.get("Regulators", []):
            # Normalize regulator name via alias map
            canon = self.regulator_alias_map.get(reg.lower(), reg)
            
            # Find impacted sectors
            impacted_sectors = self.reg_impact_map.get(canon, [])
            
            for sect in impacted_sectors:
                tickers = self.sectors.get(sect, [])
                for t in tickers:
                    add(t, 0.5, "regulatory", f"Regulator:{reg}", f"reg_impact:{canon}")

        return list(impacts_by_symbol.values())

    def map_article(self, article: NewsArticle, extractor: Optional[Any] = None) -> NewsArticle:
        """
        Maps entities to stocks and UPDATES the article.impacted_stocks field.
        """
        entities = {}
        
        # Extract if extractor provided
        if extractor is not None:
            # Combine title and content for better context
            text = f"{article.title}. {article.content}"
            entities = extractor.extract_entities(text)
        # Or look for pre-existing entities attribute (if set by pipeline)
        elif hasattr(article, "entities"):
            entities = article.entities
            
        if entities:
            impacts = self.map_to_stocks(entities)
            article.impacted_stocks = impacts
            
        return article