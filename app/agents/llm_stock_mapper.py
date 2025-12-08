"""
LLM-Based Stock Impact Mapper
Maps financial news to affected stocks using LLM reasoning.
"""

from typing import Dict, List, Optional, Any
from dataclasses import asdict

from app.core.models import NewsArticle
from app.core.llm_schemas import EntityExtractionSchema, StockImpactSchema, StockImpact, ImpactType
from app.services.llm_client import GroqLLMClient, LLMServiceError
from app.core.config_loader import get_config


class LLMStockImpactMapper:
    """Reasoning engine that maps financial news to affected stocks based on extracted entities."""
    
    def __init__(
        self,
        llm_client: Optional[GroqLLMClient] = None,
        max_stocks_per_article: int = 15
    ):
        self.config = get_config()
        
        if llm_client is None:
            self.llm_client = GroqLLMClient(
                model=self.config.llm.models.reasoning,
                temperature=0.1,
                max_tokens=3072
            )
        else:
            self.llm_client = llm_client
        
        self.max_stocks = max_stocks_per_article
        
        print(f"✓ LLMStockImpactMapper initialized")
        print(f"  - Model: {self.llm_client.model}")
        print(f"  - Max stocks per article: {max_stocks_per_article}")
    
    def _build_impact_analysis_prompt(
        self,
        entities: EntityExtractionSchema,
        article: NewsArticle
    ) -> str:
        prompt_template = self.config.prompts.stock_impact.get('task_prompt', '')

        companies_str = ""
        if entities.companies:
            companies_str = "\n".join([
                f"  - {c.name}" + (f" (Ticker: {c.ticker_symbol})" if c.ticker_symbol else "") +
                (f" [Sector: {c.sector}]" if c.sector else "") +
                f" [Confidence: {c.confidence:.2f}]"
                for c in entities.companies
            ])
        else:
            companies_str = "  None explicitly mentioned"
        
        sectors_str = ", ".join(entities.sectors) if entities.sectors else "None"
        
        regulators_str = ""
        if entities.regulators:
            regulators_str = "\n".join([
                f"  - {r.name}" + (f" ({r.jurisdiction})" if r.jurisdiction else "") +
                f" [Confidence: {r.confidence:.2f}]"
                for r in entities.regulators
            ])
        else:
            regulators_str = "  None mentioned"
        
        events_str = ""
        if entities.events:
            events_str = "\n".join([
                f"  - {e.event_type}: {e.description} [Confidence: {e.confidence:.2f}]"
                for e in entities.events
            ])
        else:
            events_str = "  None identified"
        
        return prompt_template.format(
            title=article.title,
            content=article.content,
            companies=companies_str,
            sectors=sectors_str,
            regulators=regulators_str,
            events=events_str,
            max_stocks=self.max_stocks
        )
    
    def map_to_stocks(
        self,
        entities: EntityExtractionSchema,
        article: NewsArticle
    ) -> StockImpactSchema:
        """Map news article to affected stock symbols using LLM reasoning."""
        system_message = self.config.prompts.stock_impact.get('system_message', '')
        
        prompt = self._build_impact_analysis_prompt(entities, article)
        
        try:
            result_dict = self.llm_client.generate_structured_output(
                prompt=prompt,
                schema=StockImpactSchema,
                system_message=system_message
            )
            
            stock_impact_result = StockImpactSchema.model_validate(result_dict)
            
            # Sort by confidence and apply limit
            stock_impact_result.impacted_stocks = sorted(
                stock_impact_result.impacted_stocks,
                key=lambda s: s.confidence,
                reverse=True
            )[:self.max_stocks]
            
            return stock_impact_result
            
        except Exception as e:
            raise LLMServiceError(f"Stock impact mapping failed for {article.id}: {e}")
    
    def map_article(
        self,
        article: NewsArticle,
        entities: Optional[EntityExtractionSchema] = None
    ) -> NewsArticle:
        """Wrapper to map stocks and update the article object directly."""
        if entities is None:
            raise ValueError("entities parameter is required. Extract entities first using LLMEntityExtractor.")
        
        impact_result = self.map_to_stocks(entities, article)
        
        # Convert to legacy dict format for backward compatibility
        article.impacted_stocks = [
            {
                "symbol": stock.symbol,
                "company_name": stock.company_name,
                "confidence": stock.confidence,
                "impact_type": stock.impact_type.value,
                "reasoning": stock.reasoning
            }
            for stock in impact_result.impacted_stocks
        ]
        
        return article
    
    def get_impact_statistics(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Calculate aggregate statistics across a list of articles."""
        total_articles = len(articles)
        articles_with_impacts = sum(1 for a in articles if a.impacted_stocks)
        total_stocks = sum(len(a.impacted_stocks) for a in articles)
        
        if articles_with_impacts == 0:
            return {
                "total_articles": total_articles,
                "articles_with_impacts": 0,
                "avg_stocks_per_article": 0.0,
                "impact_type_distribution": {}
            }
        
        impact_types = {"direct": 0, "sector": 0, "regulatory": 0}
        for article in articles:
            for stock in article.impacted_stocks:
                impact_type = stock.get("impact_type", "unknown")
                if impact_type in impact_types:
                    impact_types[impact_type] += 1
        
        return {
            "total_articles": total_articles,
            "articles_with_impacts": articles_with_impacts,
            "coverage_percentage": round(articles_with_impacts / total_articles * 100, 2),
            "total_stocks_identified": total_stocks,
            "avg_stocks_per_article": round(total_stocks / articles_with_impacts, 2),
            "impact_type_distribution": impact_types
        }