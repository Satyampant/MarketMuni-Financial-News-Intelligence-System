from app.core.config import Paths
"""
Supply Chain Impact Mapper
Computes cross-sectoral impacts using supply chain relationships.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from app.core.models import NewsArticle
from app.core.config_loader import get_config


@dataclass
class CrossImpact:
    """Cross-sectoral impact prediction data model."""
    source_sector: str      # Origin of news
    target_sector: str      # Sector impacted
    relationship_type: str  # 'upstream_demand_shock' or 'downstream_supply_impact'
    impact_score: float     # 0-100 scale
    dependency_weight: float
    reasoning: str
    impacted_stocks: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SupplyChainImpactMapper:
    """
    Maps news sentiment to supply chain partners.
    
    Logic:
    1. Demand Shock: Downstream news -> Upstream suppliers (e.g., Auto sales -> Steel demand).
    2. Supply Impact: Upstream news -> Downstream customers (e.g., Steel price -> Auto costs).
    """
    
    def __init__(
        self,
        graph_path: Optional[Path] = None,
        sector_tickers_path: Optional[Path] = None
    ):
        config = get_config()
        self.min_impact_score = config.supply_chain.min_impact_score
        self.default_depth = config.supply_chain.traversal_depth

        graph_path = graph_path or Paths.SUPPLY_CHAIN_GRAPH
        self.supply_chain_graph = {}
        if graph_path.exists():
            self.supply_chain_graph = json.loads(graph_path.read_text())
        
        sector_tickers_path = sector_tickers_path or Paths.SECTOR_TICKERS
        self.sector_tickers = {}
        if sector_tickers_path.exists():
            self.sector_tickers = json.loads(sector_tickers_path.read_text())
        
    def find_upstream_impacts(self, sector: str, depth: int = None) -> List[Dict[str, Any]]:
        """Finds upstream suppliers that affect the given sector."""
        if depth is None:
            depth = self.default_depth
        if sector not in self.supply_chain_graph:
            return []
        
        upstream = []
        visited = set()
        
        def traverse(current_sector: str, current_depth: int, path_weight: float):
            if current_depth > depth or current_sector in visited:
                return
            
            visited.add(current_sector)
            
            relationships = self.supply_chain_graph.get(current_sector, {})
            dependencies = relationships.get("depends_on", [])
            
            for dep in dependencies:
                dep_sector = dep["sector"]
                combined_weight = path_weight * dep["weight"]
                
                upstream.append({
                    "sector": dep_sector,
                    "weight": combined_weight,
                    "depth": current_depth,
                    "direct": current_depth == 1
                })
                
                if current_depth < depth:
                    traverse(dep_sector, current_depth + 1, combined_weight)
        
        traverse(sector, 1, 1.0)
        
        # Deduplicate: keep the highest weight path for each sector
        sector_map = {}
        for item in upstream:
            s = item["sector"]
            if s not in sector_map or item["weight"] > sector_map[s]["weight"]:
                sector_map[s] = item
        
        return sorted(sector_map.values(), key=lambda x: -x["weight"])
    
    def find_downstream_impacts(self, sector: str, depth: int = 1) -> List[Dict[str, Any]]:
        """Finds downstream customers affected by the given sector."""
        if sector not in self.supply_chain_graph:
            return []
        
        downstream = []
        visited = set()
        
        def traverse(current_sector: str, current_depth: int, path_weight: float):
            if current_depth > depth or current_sector in visited:
                return
            
            visited.add(current_sector)
            
            relationships = self.supply_chain_graph.get(current_sector, {})
            impacts = relationships.get("impacts", [])
            
            for impact in impacts:
                impact_sector = impact["sector"]
                combined_weight = path_weight * impact["weight"]
                
                downstream.append({
                    "sector": impact_sector,
                    "weight": combined_weight,
                    "depth": current_depth,
                    "direct": current_depth == 1
                })
                
                if current_depth < depth:
                    traverse(impact_sector, current_depth + 1, combined_weight)
        
        traverse(sector, 1, 1.0)
        
        # Deduplicate: keep highest weight path
        sector_map = {}
        for item in downstream:
            s = item["sector"]
            if s not in sector_map or item["weight"] > sector_map[s]["weight"]:
                sector_map[s] = item
        
        return sorted(sector_map.values(), key=lambda x: -x["weight"])
    
    def calculate_propagation_score(
        self,
        source_sector: str,
        target_sector: str,
        sentiment_signal: float
    ) -> float:
        """Calculates impact score: dependency_weight * sentiment_signal."""
        dependency_weight = 0.0
        
        # Check if target depends on source (upstream)
        upstream = self.find_upstream_impacts(target_sector, depth=2)
        for item in upstream:
            if item["sector"] == source_sector:
                dependency_weight = item["weight"]
                break
        
        # Check if source impacts target (downstream) if not found in upstream
        if dependency_weight == 0.0:
            downstream = self.find_downstream_impacts(source_sector, depth=2)
            for item in downstream:
                if item["sector"] == target_sector:
                    dependency_weight = item["weight"]
                    break
        
        return round(dependency_weight * (sentiment_signal / 100.0) * 100.0, 2)
    
    def _get_sector_stocks(self, sector: str) -> List[str]:
        return self.sector_tickers.get(sector, [])
    
    def _generate_reasoning(
        self,
        source_sector: str,
        target_sector: str,
        relationship_type: str,
        dependency_weight: float,
        sentiment_classification: str
    ) -> str:
        """Generates natural language explanation based on sentiment direction and dependency strength."""
        
        strength = "strong" if dependency_weight > 0.75 else "moderate" if dependency_weight > 0.50 else "weak"
        
        sentiment_desc = {
            "Bullish": "positive",
            "Bearish": "negative",
            "Neutral": "neutral"
        }.get(sentiment_classification, "neutral")
        
        if relationship_type == "upstream_demand_shock":
            # Downstream news -> Upstream supplier demand
            demand_direction = "increased demand" if sentiment_classification == "Bullish" else "reduced demand" if sentiment_classification == "Bearish" else "demand changes"
            return (
                f"{source_sector} sector news creates {demand_direction} signal for {target_sector} sector. "
                f"{source_sector} has a {strength} dependency on {target_sector} (weight: {dependency_weight:.2f}). "
                f"{sentiment_desc.capitalize()} developments in {source_sector} will propagate upstream to {target_sector} suppliers."
            )
        else:
            # Upstream news -> Downstream customer supply
            supply_effect = "supply boost" if sentiment_classification == "Bullish" else "supply constraints" if sentiment_classification == "Bearish" else "supply changes"
            return (
                f"{source_sector} sector news creates {supply_effect} for {target_sector} sector. "
                f"{target_sector} has a {strength} dependency on {source_sector} (weight: {dependency_weight:.2f}). "
                f"{sentiment_desc.capitalize()} developments in {source_sector} will impact downstream {target_sector} customers."
            )
    
    def generate_cross_impact_insights(
        self,
        article: NewsArticle,
        entities: Dict[str, List[str]]
    ) -> List[CrossImpact]:
        """Generates impact predictions based on extracted entities and sentiment."""
        impacts = []
        source_sectors = entities.get("Sectors", [])
        
        if not source_sectors:
            return impacts
        
        sentiment_signal = 50.0
        sentiment_classification = "Neutral"
        
        if article.has_sentiment():
            sentiment_data = article.get_sentiment()
            sentiment_signal = sentiment_data.signal_strength
            sentiment_classification = sentiment_data.classification
        
        seen_pairs = set()
        
        for source_sector in source_sectors:
            # 1. Demand Shocks (Downstream Source -> Upstream Supplier)
            upstream = self.find_upstream_impacts(source_sector, depth=1)
            
            for upstream_item in upstream:
                supplier_sector = upstream_item["sector"]
                dependency_weight = upstream_item["weight"]
                
                pair_key = (source_sector, supplier_sector)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                impact_score = self.calculate_propagation_score(
                    source_sector, supplier_sector, sentiment_signal
                )
                
                if impact_score > self.min_impact_score:
                    reasoning = self._generate_reasoning(
                        source_sector, supplier_sector,
                        "upstream_demand_shock", dependency_weight, sentiment_classification
                    )
                    
                    impacts.append(CrossImpact(
                        source_sector=source_sector,
                        target_sector=supplier_sector,
                        relationship_type="upstream_demand_shock",
                        impact_score=impact_score,
                        dependency_weight=dependency_weight,
                        reasoning=reasoning,
                        impacted_stocks=self._get_sector_stocks(supplier_sector)
                    ))
            
            # 2. Supply Shocks (Upstream Source -> Downstream Customer)
            downstream = self.find_downstream_impacts(source_sector, depth=1)
            
            for downstream_item in downstream:
                customer_sector = downstream_item["sector"]
                dependency_weight = downstream_item["weight"]
                
                pair_key = (source_sector, customer_sector)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                impact_score = self.calculate_propagation_score(
                    source_sector, customer_sector, sentiment_signal
                )
                
                if impact_score > 25.0:
                    reasoning = self._generate_reasoning(
                        source_sector, customer_sector,
                        "downstream_supply_impact", dependency_weight, sentiment_classification
                    )
                    
                    impacts.append(CrossImpact(
                        source_sector=source_sector,
                        target_sector=customer_sector,
                        relationship_type="downstream_supply_impact",
                        impact_score=impact_score,
                        dependency_weight=dependency_weight,
                        reasoning=reasoning,
                        impacted_stocks=self._get_sector_stocks(customer_sector)
                    ))
        
        return sorted(impacts, key=lambda x: -x.impact_score)

