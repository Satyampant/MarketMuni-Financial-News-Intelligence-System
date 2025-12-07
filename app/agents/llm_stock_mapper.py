"""
LLM-Based Stock Impact Mapper
Maps financial news to affected stocks using LLM reasoning.
File: app/agents/llm_stock_mapper.py
"""

from typing import Dict, List, Optional, Any
from dataclasses import asdict

from app.core.models import NewsArticle
from app.core.llm_schemas import EntityExtractionSchema, StockImpactSchema, StockImpact, ImpactType
from app.services.llm_client import GroqLLMClient, LLMServiceError
from app.core.config_loader import get_config


class LLMStockImpactMapper:
    """
    LLM-powered stock impact analysis.
    Reasons about which stocks are affected by news based on entities and content.
    """
    
    def __init__(
        self,
        llm_client: Optional[GroqLLMClient] = None,
        max_stocks_per_article: int = 15
    ):
        """
        Initialize LLM stock impact mapper.
        
        Args:
            llm_client: Optional pre-configured LLM client
            max_stocks_per_article: Maximum stocks to return per article
        """
        config = get_config()
        
        # Initialize LLM client
        if llm_client is None:
            self.llm_client = GroqLLMClient(
                model=config.llm.models.reasoning,  # Use reasoning model for complex analysis
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
        """
        Construct detailed prompt for stock impact analysis.
        
        Args:
            entities: Extracted entities from article
            article: News article with title and content
            
        Returns:
            Formatted prompt string
        """
        # Format companies with tickers
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
        
        # Format sectors
        sectors_str = ", ".join(entities.sectors) if entities.sectors else "None"
        
        # Format regulators
        regulators_str = ""
        if entities.regulators:
            regulators_str = "\n".join([
                f"  - {r.name}" + (f" ({r.jurisdiction})" if r.jurisdiction else "") +
                f" [Confidence: {r.confidence:.2f}]"
                for r in entities.regulators
            ])
        else:
            regulators_str = "  None mentioned"
        
        # Format events
        events_str = ""
        if entities.events:
            events_str = "\n".join([
                f"  - {e.event_type}: {e.description} [Confidence: {e.confidence:.2f}]"
                for e in entities.events
            ])
        else:
            events_str = "  None identified"
        
        prompt = f"""Analyze this financial news article and identify which stock symbols will be impacted.

**Article Title**: {article.title}

**Article Content**: {article.content}

**Extracted Entities**:

**Companies**:
{companies_str}

**Sectors**: {sectors_str}

**Regulators**:
{regulators_str}

**Events**:
{events_str}

---

**Task**: Identify stock symbols that will be impacted by this news. For each stock:

1. **Provide the exact stock ticker symbol** (e.g., HDFCBANK for NSE, RELIANCE for BSE, AAPL for NASDAQ)
2. **Provide the full company name** for clarity
3. **Assign a confidence score (0.0-1.0)** based on:
   - **Direct mention (0.90-1.00)**: Company explicitly named and central to the story
   - **Sector-wide impact (0.60-0.80)**: Company operates in mentioned sector and will be affected
   - **Regulatory/Indirect (0.40-0.60)**: Company affected through regulatory changes or supply chain
4. **Classify impact type**:
   - **direct**: Company explicitly mentioned in article
   - **sector**: Company in affected sector but not directly mentioned
   - **regulatory**: Impacted by regulatory changes mentioned in article
5. **Provide clear reasoning** explaining why this stock is impacted

**Guidelines**:
- Focus on **publicly traded companies** with valid stock ticker symbols
- Use correct exchange formats (NSE/BSE for Indian stocks, NYSE/NASDAQ for US stocks)
- For Indian stocks, use common ticker formats (e.g., HDFCBANK, TCS, RELIANCE, INFY)
- Prioritize **direct and sector-wide impacts** over distant connections
- Be conservative with confidence scores - only assign 0.9+ for explicit mentions
- Limit results to top {self.max_stocks} most impacted stocks
- If a company is mentioned but ticker is uncertain, still include it with reasoning
- Consider upstream/downstream supply chain effects for sector analysis

**Important**: If no companies are mentioned or impacted, return an empty list with explanation in overall_market_impact.

**Output Format**: Return structured JSON matching the StockImpactSchema format."""
        
        return prompt
    
    def map_to_stocks(
        self,
        entities: EntityExtractionSchema,
        article: NewsArticle
    ) -> StockImpactSchema:
        """
        Map news article to affected stock symbols using LLM reasoning.
        
        Args:
            entities: Extracted entities from article
            article: News article to analyze
            
        Returns:
            StockImpactSchema with impacted stocks and reasoning
            
        Raises:
            LLMServiceError: If LLM analysis fails
        """
        system_message = """You are an expert stock market analyst specializing in impact assessment.
Your task is to identify which stocks will be affected by financial news based on:
- Direct company mentions
- Sector-wide implications
- Regulatory/policy changes
- Supply chain relationships

Always provide clear reasoning and conservative confidence scores.
Return results in strict JSON format matching the StockImpactSchema."""
        
        prompt = self._build_impact_analysis_prompt(entities, article)
        
        try:
            # Call LLM with structured output
            result_dict = self.llm_client.generate_structured_output(
                prompt=prompt,
                schema=StockImpactSchema,
                system_message=system_message
            )
            
            # Validate with Pydantic
            stock_impact_result = StockImpactSchema.model_validate(result_dict)
            
            # Post-process: Sort by confidence and limit results
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
        """
        Convenience method: Map article and attach results.
        
        Args:
            article: News article to process
            entities: Optional pre-extracted entities (will extract if None)
            
        Returns:
            Article with impacted_stocks populated
        """
        if entities is None:
            raise ValueError("entities parameter is required. Extract entities first using LLMEntityExtractor.")
        
        # Perform impact mapping
        impact_result = self.map_to_stocks(entities, article)
        
        # Convert to legacy format for backward compatibility
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
        """
        Calculate statistics across multiple articles.
        
        Args:
            articles: List of articles with stock impacts
            
        Returns:
            Dict with impact statistics
        """
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
        
        # Count impact types
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