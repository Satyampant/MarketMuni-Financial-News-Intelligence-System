"""
LLM-Based Supply Chain Impact Analyzer
Uses LLM reasoning to infer cross-sectoral impacts.
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
    """Infers upstream supplier impacts and downstream customer effects using LLM reasoning."""
    
    def __init__(
        self,
        llm_client: Optional[GroqLLMClient] = None,
        min_impact_score: float = None
    ):
        self.config = get_config()
        
        # Use reasoning model for complex supply chain analysis
        if llm_client is None:
            self.llm_client = GroqLLMClient(
                model=self.config.llm.models.reasoning,
                temperature=0.1,
                max_tokens=4096
            )
        else:
            self.llm_client = llm_client
        
        self.min_impact_score = min_impact_score or self.config.supply_chain.min_impact_score
        
        print(f"✓ LLMSupplyChainAnalyzer initialized")
        print(f"  - Model: {self.llm_client.model}")
        print(f"  - Min impact score: {self.min_impact_score}")
    
    def _build_few_shot_examples(self) -> str:
        return self.config.prompts.supply_chain.few_shot_examples
        
    def _format_entity_context(self, entities: EntityExtractionSchema) -> str:
        """Formats extracted entities (companies, sectors, regulators) for the prompt."""
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
    
    def _format_sentiment_context(self, sentiment: SentimentAnalysisSchema) -> str:
        """Formats sentiment metrics for the prompt."""
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
        prompt_template = self.config.prompts.supply_chain.task_prompt
        entity_context = self._format_entity_context(entities)
        sentiment_context = self._format_sentiment_context(sentiment)
        
        return prompt_template.format(
            title=article.title,
            content=article.content,
            entity_context=entity_context,
            sentiment_context=sentiment_context,
            signal_strength=sentiment.signal_strength,
            min_impact_score=self.min_impact_score
        )
    
    def generate_cross_impact_insights(
        self,
        article: NewsArticle,
        entities: EntityExtractionSchema,
        sentiment: SentimentAnalysisSchema
    ) -> SupplyChainImpactSchema:
        """Generates supply chain impact predictions using LLM reasoning."""
        
        system_message_template = self.config.prompts.supply_chain.system_message
        few_shot_examples = self._build_few_shot_examples()
        system_message = system_message_template.format(
            few_shot_examples=few_shot_examples,
            min_impact_score=self.min_impact_score
        )
        
        prompt = self._build_analysis_prompt(article, entities, sentiment)
        
        try:
            result_dict = self.llm_client.generate_structured_output(
                prompt=prompt,
                schema=SupplyChainImpactSchema,
                system_message=system_message
            )
            
            supply_chain_result = SupplyChainImpactSchema.model_validate(result_dict)
            
            # Filter results by minimum impact score
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
            
            supply_chain_result.total_sectors_impacted = len(set(
                [i.target_sector for i in supply_chain_result.upstream_impacts] +
                [i.target_sector for i in supply_chain_result.downstream_impacts]
            ))
            
            return supply_chain_result
            
        except Exception as e:
            raise LLMServiceError(f"Supply chain analysis failed for {article.id}: {e}")
    
    def analyze_and_attach(
        self,
        article: NewsArticle,
        entities: EntityExtractionSchema,
        sentiment: SentimentAnalysisSchema
    ) -> NewsArticle:
        """Analyzes supply chain impacts and attaches them to the article."""
        
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
    
    def get_impact_statistics(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Calculates statistics across a list of analyzed articles."""
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