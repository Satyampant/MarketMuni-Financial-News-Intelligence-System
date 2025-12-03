from app.core.config import Paths
"""
Supply Chain Impact Mapper
Computes cross-sectoral impacts using supply chain relationships
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from app.core.models import NewsArticle

# MODULE_DIR replaced by Paths config


@dataclass
class CrossImpact:
    """Cross-sectoral impact prediction"""
    source_sector: str  # Sector where news originated
    target_sector: str  # Sector being impacted
    relationship_type: str  # "upstream_demand_shock" | "downstream_supply_impact"
    impact_score: float  # 0-100 scale
    dependency_weight: float
    reasoning: str
    impacted_stocks: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SupplyChainImpactMapper:
    """
    Maps sectoral news to cross-industry impacts using supply chain graph.
    
    Impact Types:
    - upstream_demand_shock: News in downstream sector creates demand signal for upstream suppliers
      Example: Auto sales surge → increased demand for Steel
    - downstream_supply_impact: News in upstream sector creates supply signal for downstream customers
      Example: Steel price increase → cost pressure on Auto manufacturers
    """
    
    def __init__(
        self,
        graph_path: Optional[Path] = None,
        sector_tickers_path: Optional[Path] = None
    ):
        """
        Initialize supply chain mapper.
        
        Args:
            graph_path: Path to supply_chain_graph.json
            sector_tickers_path: Path to sector_tickers.json
        """
        # Load supply chain graph
        graph_path = graph_path or Paths.SUPPLY_CHAIN_GRAPH
        self.supply_chain_graph = {}
        if graph_path.exists():
            self.supply_chain_graph = json.loads(graph_path.read_text())
        
        # Load sector tickers for stock mapping
        sector_tickers_path = sector_tickers_path or Paths.SECTOR_TICKERS
        self.sector_tickers = {}
        if sector_tickers_path.exists():
            self.sector_tickers = json.loads(sector_tickers_path.read_text())
        
    
    def find_upstream_impacts(
        self, 
        sector: str, 
        depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Find sectors that affect the given sector (upstream dependencies).
        
        Args:
            sector: Target sector name
            depth: How many levels to traverse (1 = direct dependencies only)
        
        Returns:
            List of upstream sectors with weights
        """
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
                dep_weight = dep["weight"]
                combined_weight = path_weight * dep_weight
                
                upstream.append({
                    "sector": dep_sector,
                    "weight": combined_weight,
                    "depth": current_depth,
                    "direct": current_depth == 1
                })
                
                # Recursive traversal for multi-hop dependencies
                if current_depth < depth:
                    traverse(dep_sector, current_depth + 1, combined_weight)
        
        traverse(sector, 1, 1.0)
        
        # Deduplicate and keep highest weight
        sector_map = {}
        for item in upstream:
            s = item["sector"]
            if s not in sector_map or item["weight"] > sector_map[s]["weight"]:
                sector_map[s] = item
        
        return sorted(sector_map.values(), key=lambda x: -x["weight"])
    
    def find_downstream_impacts(
        self, 
        sector: str, 
        depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Find sectors affected by the given sector (downstream impacts).
        
        Args:
            sector: Source sector name
            depth: How many levels to traverse
        
        Returns:
            List of downstream sectors with weights
        """
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
                impact_weight = impact["weight"]
                combined_weight = path_weight * impact_weight
                
                downstream.append({
                    "sector": impact_sector,
                    "weight": combined_weight,
                    "depth": current_depth,
                    "direct": current_depth == 1
                })
                
                # Recursive traversal
                if current_depth < depth:
                    traverse(impact_sector, current_depth + 1, combined_weight)
        
        traverse(sector, 1, 1.0)
        
        # Deduplicate and keep highest weight
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
        """
        Calculate impact propagation score.
        
        Formula: dependency_weight × sentiment_signal × 100
        
        Args:
            source_sector: Sector where news originated
            target_sector: Sector being impacted
            sentiment_signal: Sentiment signal strength (0-100)
        
        Returns:
            Impact score (0-100 scale)
        """
        # Find dependency weight between sectors
        dependency_weight = 0.0
        
        # Check upstream (target depends on source)
        upstream = self.find_upstream_impacts(target_sector, depth=2)
        for item in upstream:
            if item["sector"] == source_sector:
                dependency_weight = item["weight"]
                break
        
        # Check downstream (source impacts target)
        if dependency_weight == 0.0:
            downstream = self.find_downstream_impacts(source_sector, depth=2)
            for item in downstream:
                if item["sector"] == target_sector:
                    dependency_weight = item["weight"]
                    break
        
        # Calculate final score
        impact_score = dependency_weight * (sentiment_signal / 100.0) * 100.0
        
        return round(impact_score, 2)
    
    def _get_sector_stocks(self, sector: str) -> List[str]:
        """Get stock symbols for a sector"""
        return self.sector_tickers.get(sector, [])
    
    def _generate_reasoning(
        self,
        source_sector: str,
        target_sector: str,
        relationship_type: str,
        dependency_weight: float,
        sentiment_classification: str
    ) -> str:
        """Generate natural language explanation for the impact"""
        
        strength = "strong" if dependency_weight > 0.75 else "moderate" if dependency_weight > 0.50 else "weak"
        
        sentiment_desc = {
            "Bullish": "positive",
            "Bearish": "negative",
            "Neutral": "neutral"
        }.get(sentiment_classification, "neutral")
        
        if relationship_type == "upstream_demand_shock":
            # News in downstream sector creates demand signal for upstream suppliers
            demand_direction = "increased demand" if sentiment_classification == "Bullish" else "reduced demand" if sentiment_classification == "Bearish" else "demand changes"
            return (
                f"{source_sector} sector news creates {demand_direction} signal for {target_sector} sector. "
                f"{source_sector} has a {strength} dependency on {target_sector} (weight: {dependency_weight:.2f}). "
                f"{sentiment_desc.capitalize()} developments in {source_sector} will propagate upstream to {target_sector} suppliers."
            )
        else:  # downstream_supply_impact
            # News in upstream sector affects downstream customers
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
        """
        Generate cross-sectoral impact predictions from article.
        
        Args:
            article: NewsArticle with sentiment data
            entities: Extracted entities dict (must contain "Sectors")
        
        Returns:
            List of CrossImpact objects with predictions
        """
        impacts = []
        
        # Extract sectors from entities
        source_sectors = entities.get("Sectors", [])
        if not source_sectors:
            return impacts
        
        # Get sentiment data
        sentiment_signal = 50.0  # Default neutral
        sentiment_classification = "Neutral"
        
        if article.has_sentiment():
            sentiment_data = article.get_sentiment()
            sentiment_signal = sentiment_data.signal_strength
            sentiment_classification = sentiment_data.classification
        
        # For each source sector, find cross-impacts
        seen_pairs = set()
        
        for source_sector in source_sectors:
            # Find upstream dependencies (sectors that source depends on)
            # This models DEMAND SHOCKS: News in downstream creates demand signal for upstream
            upstream = self.find_upstream_impacts(source_sector, depth=1)
            
            for upstream_item in upstream:
                supplier_sector = upstream_item["sector"]
                dependency_weight = upstream_item["weight"]
                
                # Skip if already processed
                pair_key = (source_sector, supplier_sector)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                # Calculate impact score
                # News in source_sector creates demand signal for supplier_sector
                impact_score = self.calculate_propagation_score(
                    source_sector,
                    supplier_sector,
                    sentiment_signal
                )
                
                # Only include significant impacts (score > 25)
                if impact_score > 25.0:
                    reasoning = self._generate_reasoning(
                        source_sector,  # News sector (downstream)
                        supplier_sector,  # Impacted sector (upstream supplier)
                        "upstream_demand_shock",
                        dependency_weight,
                        sentiment_classification
                    )
                    
                    impacts.append(CrossImpact(
                        source_sector=source_sector,  # Where news originated
                        target_sector=supplier_sector,  # Who gets impacted
                        relationship_type="upstream_demand_shock",
                        impact_score=impact_score,
                        dependency_weight=dependency_weight,
                        reasoning=reasoning,
                        impacted_stocks=self._get_sector_stocks(supplier_sector)
                    ))
            
            # Find downstream impacts (sectors that source impacts)
            # This models SUPPLY SHOCKS: News in upstream creates supply signal for downstream
            downstream = self.find_downstream_impacts(source_sector, depth=1)
            
            for downstream_item in downstream:
                customer_sector = downstream_item["sector"]
                dependency_weight = downstream_item["weight"]
                
                # Skip if already processed
                pair_key = (source_sector, customer_sector)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                # Calculate impact score
                # News in source_sector creates supply signal for customer_sector
                impact_score = self.calculate_propagation_score(
                    source_sector,
                    customer_sector,
                    sentiment_signal
                )
                
                # Only include significant impacts
                if impact_score > 25.0:
                    reasoning = self._generate_reasoning(
                        source_sector,  # News sector (upstream)
                        customer_sector,  # Impacted sector (downstream customer)
                        "downstream_supply_impact",
                        dependency_weight,
                        sentiment_classification
                    )
                    
                    impacts.append(CrossImpact(
                        source_sector=source_sector,  # Where news originated
                        target_sector=customer_sector,  # Who gets impacted
                        relationship_type="downstream_supply_impact",
                        impact_score=impact_score,
                        dependency_weight=dependency_weight,
                        reasoning=reasoning,
                        impacted_stocks=self._get_sector_stocks(customer_sector)
                    ))
        
        # Sort by impact score (highest first)
        impacts.sort(key=lambda x: -x.impact_score)
        
        return impacts


# Testing
if __name__ == "__main__":
    from datetime import datetime
    from app.core.models import SentimentData
    
    # Initialize mapper
    mapper = SupplyChainImpactMapper()
    
    print("=" * 80)
    print("Supply Chain Impact Mapper Tests")
    print("=" * 80)
    
    # Test 1: Find upstream dependencies
    print("\n[Test 1] Upstream dependencies for Auto sector:")
    upstream = mapper.find_upstream_impacts("Auto", depth=1)
    for item in upstream[:5]:
        print(f"  {item['sector']}: weight={item['weight']:.2f}")
    
    # Test 2: Find downstream impacts
    print("\n[Test 2] Downstream impacts from Steel sector:")
    downstream = mapper.find_downstream_impacts("Steel", depth=1)
    for item in downstream[:5]:
        print(f"  {item['sector']}: weight={item['weight']:.2f}")
    
    # Test 3: Generate cross-impact insights
    print("\n[Test 3] Cross-impact analysis:")
    
    test_article = NewsArticle(
        id="TEST_SUPPLY_001",
        title="Steel prices surge 20% amid supply constraints",
        content="Steel manufacturers report record prices as global supply chains face disruptions.",
        source="Test",
        timestamp=datetime.now()
    )
    
    # Add sentiment data
    test_article.set_sentiment(SentimentData(
        classification="Bearish",
        confidence_score=85.0,
        signal_strength=75.0,
        sentiment_breakdown={"bullish": 10, "bearish": 85, "neutral": 5},
        analysis_method="test"
    ))
    
    entities = {
        "Companies": [],
        "Sectors": ["Steel"],
        "Regulators": [],
        "People": [],
        "Events": []
    }
    
    impacts = mapper.generate_cross_impact_insights(test_article, entities)
    
    print(f"\nFound {len(impacts)} cross-sectoral impacts:")
    for impact in impacts[:5]:
        print(f"\n  {impact.source_sector} → {impact.target_sector}")
        print(f"  Type: {impact.relationship_type}")
        print(f"  Impact Score: {impact.impact_score:.2f}")
        print(f"  Reasoning: {impact.reasoning}")
        print(f"  Stocks: {', '.join(impact.impacted_stocks[:3])}")