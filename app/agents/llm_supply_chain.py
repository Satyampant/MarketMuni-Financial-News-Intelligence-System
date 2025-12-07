"""
LLM-Based Supply Chain Impact Analyzer
Uses LLM reasoning to infer cross-sectoral impacts without hardcoded dependency graphs.
File: app/agents/llm_supply_chain.py
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from app.core.models import NewsArticle
from app.core.llm_schemas import (
    EntityExtractionSchema,
    SentimentAnalysisSchema,
    SupplyChainImpactSchema,
    CrossImpact,
    RelationshipType
)
from app.services.llm_client import GroqLLMClient, LLMServiceError
from app.core.config_loader import get_config


class LLMSupplyChainAnalyzer:
    """
    LLM-powered supply chain impact analysis.
    Infers upstream supplier impacts and downstream customer effects using reasoning.
    """
    
    def __init__(
        self,
        llm_client: Optional[GroqLLMClient] = None,
        min_impact_score: float = None
    ):
        """
        Initialize LLM supply chain analyzer.
        
        Args:
            llm_client: Optional pre-configured LLM client
            min_impact_score: Minimum impact score threshold (default from config)
        """
        config = get_config()
        
        # Use reasoning model for complex supply chain analysis
        if llm_client is None:
            self.llm_client = GroqLLMClient(
                model=config.llm.models.reasoning,
                temperature=0.1,
                max_tokens=4096
            )
        else:
            self.llm_client = llm_client
        
        self.min_impact_score = min_impact_score or config.supply_chain.min_impact_score
        
        print(f"✓ LLMSupplyChainAnalyzer initialized")
        print(f"  - Model: {self.llm_client.model}")
        print(f"  - Min impact score: {self.min_impact_score}")
    
    def _build_few_shot_examples(self) -> str:
        """
        Construct few-shot examples for supply chain impact analysis.
        """
        examples = """
**Example 1: Auto Sector Sales Surge**
Article: "Auto sector sales jump 20% in festive season driven by strong consumer demand"
Sectors: [Auto]
Sentiment: Bullish (Signal Strength: 85)

Analysis:
Upstream Impacts (Demand Shocks):
1. Steel Sector
   - Impact Score: 76.5
   - Dependency Weight: 0.85
   - Reasoning: Increased auto production creates higher demand for steel components (body panels, frames). Auto sector has strong dependency on steel suppliers. Positive sales translate to increased orders for upstream steel manufacturers.
   - Time Horizon: Short-term (1-3 months)

2. Semiconductors Sector
   - Impact Score: 67.5
   - Dependency Weight: 0.75
   - Reasoning: Modern vehicles require chips for infotainment, safety systems, and powertrains. Sales surge drives chip demand. Auto-semiconductor supply chain is critical with moderate-high dependency.
   - Time Horizon: Medium-term (3-6 months)

3. Rubber Sector
   - Impact Score: 58.5
   - Dependency Weight: 0.65
   - Reasoning: Tire manufacturers (rubber sector) benefit from increased vehicle production. Auto sector depends on rubber for tires and seals. Positive demand signal propagates upstream.
   - Time Horizon: Short-term (1-2 months)

Downstream Impacts (Supply Effects):
1. Logistics Sector
   - Impact Score: 54.0
   - Dependency Weight: 0.60
   - Reasoning: Higher auto sales increase vehicle availability for logistics fleets. Transportation companies benefit from expanded supply. Moderate dependency on auto sector for fleet expansion.
   - Time Horizon: Medium-term (3-6 months)

2. Insurance Sector
   - Impact Score: 40.5
   - Dependency Weight: 0.45
   - Reasoning: More vehicles on road means increased insurance policy demand. Insurance sector benefits from auto market growth through new policy sales.
   - Time Horizon: Long-term (6-12 months)

---

**Example 2: Steel Price Hike**
Article: "Steel prices increase 15% due to rising raw material costs and supply constraints"
Sectors: [Steel]
Sentiment: Bearish (Signal Strength: 78)

Analysis:
Upstream Impacts (Demand Shocks):
1. Mining Sector
   - Impact Score: 63.0
   - Dependency Weight: 0.90
   - Reasoning: Steel mills are major iron ore consumers. Price hikes may reduce steel production, lowering mining demand. Strong dependency creates direct upstream impact.
   - Time Horizon: Short-term (1-2 months)

2. Coal Sector
   - Impact Score: 52.5
   - Dependency Weight: 0.75
   - Reasoning: Steel production uses coking coal. Price pressures may reduce steel output, affecting coal demand. Moderate-high dependency on coal suppliers.
   - Time Horizon: Short-term (2-3 months)

Downstream Impacts (Supply Effects):
1. Auto Sector
   - Impact Score: 68.0
   - Dependency Weight: 0.85
   - Reasoning: Higher steel costs increase auto manufacturing expenses, compressing margins. Auto sector highly dependent on steel supplies. Price increases create cost pressure downstream.
   - Time Horizon: Short-term (1-3 months)

2. Construction Sector
   - Impact Score: 64.0
   - Dependency Weight: 0.80
   - Reasoning: Steel is critical for infrastructure and building projects. Price hikes increase construction costs, potentially delaying projects. Strong downstream dependency.
   - Time Horizon: Medium-term (3-6 months)

3. Infrastructure Sector
   - Impact Score: 60.0
   - Dependency Weight: 0.75
   - Reasoning: Infrastructure projects require large steel quantities. Price increases affect project economics and timelines. Moderate-high dependency creates supply cost impact.
   - Time Horizon: Medium-term (3-9 months)

---

**Example 3: RBI Rate Hike**
Article: "RBI raises repo rate by 25 basis points to control inflation"
Sectors: [Banking, Finance]
Sentiment: Neutral (Signal Strength: 50)

Analysis:
Upstream Impacts (Demand Shocks):
1. IT Sector
   - Impact Score: 35.0
   - Dependency Weight: 0.70
   - Reasoning: Banks depend on IT services for digital infrastructure. Rate hikes may constrain banking budgets, reducing IT spending. Moderate dependency creates upstream demand pressure.
   - Time Horizon: Medium-term (6-12 months)

Downstream Impacts (Supply Effects):
1. Real Estate Sector
   - Impact Score: 72.0
   - Dependency Weight: 0.90
   - Reasoning: Higher interest rates increase mortgage costs, reducing real estate demand. Real estate heavily dependent on banking credit availability. Rate hikes directly impact affordability.
   - Time Horizon: Medium-term (3-6 months)

2. Auto Sector
   - Impact Score: 56.0
   - Dependency Weight: 0.70
   - Reasoning: Auto loans become more expensive with rate hikes, dampening vehicle sales. Auto sector relies on consumer financing. Higher rates reduce purchase affordability.
   - Time Horizon: Short-term (1-3 months)

3. Consumer Durables Sector
   - Impact Score: 45.0
   - Dependency Weight: 0.60
   - Reasoning: EMI financing for appliances and electronics becomes costlier. Consumer durables depend on credit-driven demand. Rate increases reduce purchase capacity.
   - Time Horizon: Short-term (1-3 months)
"""
        return examples.strip()
    
    def _format_entity_context(
        self,
        entities: EntityExtractionSchema
    ) -> str:
        """
        Format entity context for prompt.
        
        Args:
            entities: Extracted entities from article
            
        Returns:
            Formatted entity string
        """
        context_parts = []
        
        if entities.companies:
            companies_str = ", ".join([c.name for c in entities.companies])
            context_parts.append(f"Companies: [{companies_str}]")
        
        if entities.sectors:
            sectors_str = ", ".join(entities.sectors)
            context_parts.append(f"Sectors: [{sectors_str}]")
        
        if entities.regulators:
            regulators_str = ", ".join([r.name for r in entities.regulators])
            context_parts.append(f"Regulators: [{regulators_str}]")
        
        if entities.events:
            events_str = ", ".join([e.event_type for e in entities.events])
            context_parts.append(f"Events: [{events_str}]")
        
        return "\n".join(context_parts) if context_parts else "No key entities identified"
    
    def _format_sentiment_context(
        self,
        sentiment: SentimentAnalysisSchema
    ) -> str:
        """
        Format sentiment context for prompt.
        
        Args:
            sentiment: Sentiment analysis result
            
        Returns:
            Formatted sentiment string
        """
        return f"""Sentiment Classification: {sentiment.classification.value}
Signal Strength: {sentiment.signal_strength}/100
Confidence: {sentiment.confidence_score}/100
Key Factors:
{chr(10).join(f'  - {factor}' for factor in sentiment.key_factors[:3])}"""
    
    def _build_analysis_prompt(
        self,
        article: NewsArticle,
        entities: EntityExtractionSchema,
        sentiment: SentimentAnalysisSchema
    ) -> str:
        """
        Construct supply chain impact analysis prompt.
        
        Args:
            article: News article to analyze
            entities: Extracted entities
            sentiment: Sentiment analysis result
            
        Returns:
            Formatted prompt string
        """
        entity_context = self._format_entity_context(entities)
        sentiment_context = self._format_sentiment_context(sentiment)
        
        prompt = f"""Analyze the supply chain impacts of this financial news across sectors.

**Article Title**: {article.title}

**Article Content**: {article.content}

**Extracted Entities**:
{entity_context}

**Sentiment Analysis**:
{sentiment_context}

---

**Your Task**: Identify cross-sectoral supply chain impacts using economic reasoning.

**Analysis Framework**:

1. **Upstream Impacts (Demand Shocks)**:
   - Identify suppliers/dependencies of the affected sector(s)
   - Assess how news creates demand changes for upstream sectors
   - Consider: How does sentiment direction affect supplier demand?
   - Example: Auto sales ↑ → Steel demand ↑ (upstream supplier benefits)

2. **Downstream Impacts (Supply Effects)**:
   - Identify customers/consumers of the affected sector(s)
   - Assess how news creates supply changes for downstream sectors
   - Consider: How does sentiment direction affect customer costs/availability?
   - Example: Steel prices ↑ → Auto costs ↑ (downstream customer pressured)

**For Each Impacted Sector, Provide**:
- **Target Sector**: Name of the impacted sector
- **Relationship Type**: upstream_demand_shock OR downstream_supply_impact
- **Impact Score** (0-100): Magnitude of impact based on:
  - Dependency strength between sectors (0.0-1.0 weight)
  - Sentiment signal strength ({sentiment.signal_strength}/100)
  - Formula: dependency_weight × signal_strength = impact_score
- **Dependency Weight** (0.0-1.0): How strongly sectors are connected
  - 0.9-1.0: Critical dependency (e.g., Auto→Steel, Real Estate→Banking)
  - 0.7-0.8: Strong dependency (e.g., IT→Telecom, Pharma→Chemicals)
  - 0.5-0.6: Moderate dependency (e.g., Auto→Logistics, Banking→IT)
  - Below 0.5: Weak/indirect dependency
- **Reasoning**: Clear explanation of the supply chain mechanism and impact direction
- **Time Horizon**: Expected timeframe (short-term: 1-3 months, medium-term: 3-6 months, long-term: 6-12 months)
- **Impacted Stocks**: List of major stock symbols in the target sector (if applicable)

**Guidelines**:
- Only include impacts with **impact_score >= {self.min_impact_score}**
- Be **conservative** with dependency weights - require clear economic linkage
- Consider **sentiment direction**:
  - Bullish: Demand shocks positive for suppliers, supply effects may benefit/pressure customers
  - Bearish: Demand shocks negative for suppliers, supply constraints for customers
  - Neutral: Minimal propagation effects
- Focus on **direct supply chain relationships** (1-2 degrees of separation)
- Limit analysis to **top 8 most significant impacts** (4 upstream + 4 downstream max)
- For **regulator/policy news**, focus on regulated sectors and their dependencies

**Important**: If the news has minimal supply chain implications (e.g., single company dividend), return empty upstream/downstream lists with appropriate reasoning.

Now analyze the supply chain impacts above."""
        
        return prompt
    
    def generate_cross_impact_insights(
        self,
        article: NewsArticle,
        entities: EntityExtractionSchema,
        sentiment: SentimentAnalysisSchema
    ) -> SupplyChainImpactSchema:
        """
        Generate supply chain impact predictions using LLM reasoning.
        
        Args:
            article: News article to analyze
            entities: Extracted entities from article
            sentiment: Sentiment analysis result
            
        Returns:
            SupplyChainImpactSchema with upstream and downstream impacts
            
        Raises:
            LLMServiceError: If supply chain analysis fails
        """
        # Build system message with few-shot examples
        system_message = f"""You are an expert supply chain economist specializing in cross-sectoral impact analysis.

Your task is to analyze financial news and predict how it affects upstream suppliers and downstream customers through supply chain relationships.

**Few-Shot Examples**:

{self._build_few_shot_examples()}

---

**Your Analysis Should**:
- Use economic reasoning to infer supply chain dependencies
- Assign realistic dependency weights based on sector relationships
- Calculate impact scores using: dependency_weight × sentiment_signal_strength
- Provide clear, evidence-based reasoning for each impact
- Consider time horizons for impact realization
- Only include impacts above the minimum threshold ({self.min_impact_score})

Always return your analysis in the specified structured JSON format matching SupplyChainImpactSchema."""
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(article, entities, sentiment)
        
        try:
            # Call LLM with structured output
            result_dict = self.llm_client.generate_structured_output(
                prompt=prompt,
                schema=SupplyChainImpactSchema,
                system_message=system_message
            )
            
            # Validate with Pydantic
            supply_chain_result = SupplyChainImpactSchema.model_validate(result_dict)
            
            # Post-process: Filter by minimum impact score
            supply_chain_result.upstream_impacts = [
                impact for impact in supply_chain_result.upstream_impacts
                if impact.impact_score >= self.min_impact_score
            ]
            
            supply_chain_result.downstream_impacts = [
                impact for impact in supply_chain_result.downstream_impacts
                if impact.impact_score >= self.min_impact_score
            ]
            
            # Sort by impact score (descending)
            supply_chain_result.upstream_impacts.sort(
                key=lambda x: x.impact_score,
                reverse=True
            )
            supply_chain_result.downstream_impacts.sort(
                key=lambda x: x.impact_score,
                reverse=True
            )
            
            # Update total sectors impacted count
            total_sectors = len(set(
                [i.target_sector for i in supply_chain_result.upstream_impacts] +
                [i.target_sector for i in supply_chain_result.downstream_impacts]
            ))
            supply_chain_result.total_sectors_impacted = total_sectors
            
            return supply_chain_result
            
        except Exception as e:
            raise LLMServiceError(f"Supply chain analysis failed for {article.id}: {e}")
    
    def analyze_and_attach(
        self,
        article: NewsArticle,
        entities: EntityExtractionSchema,
        sentiment: SentimentAnalysisSchema
    ) -> NewsArticle:
        """
        Convenience method: Analyze supply chain impacts and attach to article.
        
        Args:
            article: News article to analyze
            entities: Extracted entities
            sentiment: Sentiment analysis result
            
        Returns:
            Article with cross_impacts populated
        """
        # Perform LLM supply chain analysis
        supply_chain_result = self.generate_cross_impact_insights(
            article, entities, sentiment
        )
        
        # Convert to legacy format for backward compatibility
        all_impacts = (
            supply_chain_result.upstream_impacts +
            supply_chain_result.downstream_impacts
        )
        
        cross_impact_dicts = [
            {
                "source_sector": impact.source_sector,
                "target_sector": impact.target_sector,
                "relationship_type": impact.relationship_type.value,
                "impact_score": impact.impact_score,
                "dependency_weight": impact.dependency_weight,
                "reasoning": impact.reasoning,
                "impacted_stocks": impact.impacted_stocks,
                "time_horizon": impact.time_horizon
            }
            for impact in all_impacts
        ]
        
        article.set_cross_impacts(cross_impact_dicts)
        return article
    
    def get_impact_statistics(
        self,
        articles: List[NewsArticle]
    ) -> Dict[str, Any]:
        """
        Calculate supply chain impact statistics across articles.
        
        Args:
            articles: List of articles with supply chain impacts
            
        Returns:
            Dict with statistics
        """
        if not articles:
            return {
                "total_articles": 0,
                "articles_with_impacts": 0,
                "total_impacts": 0
            }
        
        total_articles = len(articles)
        articles_with_impacts = sum(1 for a in articles if a.has_cross_impacts())
        
        total_upstream = 0
        total_downstream = 0
        total_impacts = 0
        max_impact_score = 0.0
        all_target_sectors = set()
        
        for article in articles:
            if article.has_cross_impacts():
                for impact in article.cross_impacts:
                    total_impacts += 1
                    all_target_sectors.add(impact.get("target_sector"))
                    
                    impact_score = impact.get("impact_score", 0.0)
                    if impact_score > max_impact_score:
                        max_impact_score = impact_score
                    
                    rel_type = impact.get("relationship_type", "")
                    if rel_type == "upstream_demand_shock":
                        total_upstream += 1
                    elif rel_type == "downstream_supply_impact":
                        total_downstream += 1
        
        avg_impacts = total_impacts / articles_with_impacts if articles_with_impacts > 0 else 0.0
        
        return {
            "total_articles": total_articles,
            "articles_with_impacts": articles_with_impacts,
            "coverage_percentage": round(articles_with_impacts / total_articles * 100, 2) if total_articles > 0 else 0,
            "total_impacts": total_impacts,
            "upstream_impacts": total_upstream,
            "downstream_impacts": total_downstream,
            "avg_impacts_per_article": round(avg_impacts, 2),
            "max_impact_score": round(max_impact_score, 2),
            "unique_target_sectors": len(all_target_sectors),
            "analysis_method": "llm"
        }