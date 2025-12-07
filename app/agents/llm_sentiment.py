"""
LLM-Based Sentiment Analysis Agent
Replaces rule-based lexicon and FinBERT with LLM few-shot prompting.
File: app/agents/llm_sentiment.py
"""

from typing import Dict, Any, Optional
from datetime import datetime

from app.core.models import NewsArticle, SentimentData
from app.core.llm_schemas import (
    EntityExtractionSchema, 
    SentimentAnalysisSchema,
    SentimentClassification
)
from app.services.llm_client import GroqLLMClient, LLMServiceError
from app.core.config_loader import get_config


class LLMSentimentAnalyzer:
    """
    LLM-powered sentiment analysis using few-shot prompting.
    Replaces rule-based lexicon and FinBERT approaches with context-aware LLM reasoning.
    """
    
    def __init__(
        self,
        llm_client: Optional[GroqLLMClient] = None,
        use_entity_context: bool = True
    ):
        """
        Initialize LLM sentiment analyzer.
        
        Args:
            llm_client: Optional pre-configured LLM client
            use_entity_context: Whether to include entity context in prompts
        """
        config = get_config()
        
        # Use reasoning model for sentiment analysis
        if llm_client is None:
            self.llm_client = GroqLLMClient(
                model=config.llm.models.reasoning,
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=2048
            )
        else:
            self.llm_client = llm_client
        
        self.use_entity_context = use_entity_context
        
        print(f"✓ LLMSentimentAnalyzer initialized")
        print(f"  - Model: {self.llm_client.model}")
        print(f"  - Entity context: {'enabled' if use_entity_context else 'disabled'}")
    
    def _build_few_shot_examples(self) -> str:
        """
        Construct few-shot examples for sentiment analysis.
        These examples teach the LLM the task format and expected reasoning.
        """
        examples = """
**Example 1:**
Article: "HDFC Bank announces 15% dividend increase and approves Rs 5,000 crore stock buyback program."
Entities: Companies: [HDFC Bank], Events: [dividend, stock buyback]
Analysis:
- Classification: Bullish
- Confidence Score: 92
- Key Factors:
  * Significant dividend increase (15%) signals strong profitability
  * Stock buyback demonstrates management confidence in valuation
  * Capital return to shareholders is positive market signal
- Signal Strength: 95

**Example 2:**
Article: "TCS announces layoffs affecting 5,000 employees amid cost-cutting measures and declining revenue growth."
Entities: Companies: [TCS], Events: [layoffs]
Analysis:
- Classification: Bearish
- Confidence Score: 88
- Key Factors:
  * Large-scale layoffs indicate operational challenges
  * Cost-cutting suggests margin pressure
  * Declining revenue growth is negative fundamental signal
- Signal Strength: 85

**Example 3:**
Article: "RBI maintains repo rate at 6.5%, citing balanced inflation and growth outlook."
Entities: Regulators: [RBI], Events: [policy_decision]
Analysis:
- Classification: Neutral
- Confidence Score: 75
- Key Factors:
  * No change in policy rate maintains status quo
  * Balanced economic outlook (neither hawkish nor dovish)
  * Market expectations were already aligned with this decision
- Signal Strength: 50

**Example 4:**
Article: "Reliance Industries Q3 profit surges 25% beating analyst estimates, driven by strong retail and telecom performance."
Entities: Companies: [Reliance Industries], Events: [earnings]
Analysis:
- Classification: Bullish
- Confidence Score: 90
- Key Factors:
  * Earnings beat analyst consensus (positive surprise)
  * Strong growth rate (25% YoY) across key segments
  * Multiple business units performing well (diversification strength)
- Signal Strength: 93

**Example 5:**
Article: "Steel prices decline 8% this quarter due to weak demand from construction and auto sectors."
Entities: Sectors: [Steel, Construction, Auto], Events: [price_decline]
Analysis:
- Classification: Bearish
- Confidence Score: 82
- Key Factors:
  * Price decline directly impacts steel sector margins
  * Weak demand from major customers (construction, auto)
  * Sector-wide headwinds suggest broader economic slowdown
- Signal Strength: 78

**Example 6:**
Article: "Asian Paints announces 3% price increase across product portfolio citing higher raw material costs."
Entities: Companies: [Asian Paints], Events: [price_increase]
Analysis:
- Classification: Neutral
- Confidence Score: 65
- Key Factors:
  * Price increase can improve margins (positive)
  * Rising input costs compress profitability (negative)
  * Impact on demand from price hike is uncertain
- Signal Strength: 55
"""
        return examples.strip()
    
    def _format_entity_context(self, entities: EntityExtractionSchema) -> str:
        """
        Format entity context for inclusion in prompt.
        
        Args:
            entities: Extracted entities from article
            
        Returns:
            Formatted entity string
        """
        if not entities:
            return "No entities extracted"
        
        context_parts = []
        
        # Companies
        if entities.companies:
            companies_str = ", ".join([
                f"{c.name}" + (f" ({c.ticker_symbol})" if c.ticker_symbol else "")
                for c in entities.companies
            ])
            context_parts.append(f"Companies: [{companies_str}]")
        
        # Sectors
        if entities.sectors:
            sectors_str = ", ".join(entities.sectors)
            context_parts.append(f"Sectors: [{sectors_str}]")
        
        # Regulators
        if entities.regulators:
            regulators_str = ", ".join([r.name for r in entities.regulators])
            context_parts.append(f"Regulators: [{regulators_str}]")
        
        # Events
        if entities.events:
            events_str = ", ".join([e.event_type for e in entities.events])
            context_parts.append(f"Events: [{events_str}]")
        
        return ", ".join(context_parts) if context_parts else "No key entities identified"
    
    def _build_analysis_prompt(
        self,
        article: NewsArticle,
        entities: Optional[EntityExtractionSchema] = None
    ) -> str:
        """
        Construct sentiment analysis prompt with few-shot examples.
        
        Args:
            article: News article to analyze
            entities: Optional extracted entities for context
            
        Returns:
            Formatted prompt string
        """
        # Entity context
        entity_context = ""
        if self.use_entity_context and entities:
            entity_context = f"\n**Entities**: {self._format_entity_context(entities)}"
        
        prompt = f"""Analyze the sentiment of this financial news article and determine its market impact.

**Article Title**: {article.title}

**Article Content**: {article.content}{entity_context}

---

**Your Task**: Provide a comprehensive sentiment analysis following this structure:

1. **Classification**: Choose one - Bullish, Bearish, or Neutral
2. **Confidence Score**: Rate 0-100 based on signal clarity
3. **Key Factors**: List 3-5 specific reasons supporting your classification
4. **Signal Strength**: Rate 0-100 for trading signal quality (higher = stronger actionable signal)

**Guidelines**:
- **Bullish**: Positive news (earnings beats, dividends, expansion, strong growth, upgrades)
- **Bearish**: Negative news (losses, layoffs, regulatory issues, downgrades, missed targets)
- **Neutral**: Mixed signals, status quo, or unclear impact

- **Confidence Score**: Based on clarity and magnitude of impact
  - 90-100: Crystal clear sentiment with strong evidence
  - 70-90: Clear sentiment with good supporting factors
  - 50-70: Moderate clarity with some ambiguity
  - Below 50: Unclear or highly mixed signals

- **Signal Strength**: Actionable trading signal quality
  - Consider: Entity importance, event magnitude, market relevance
  - Direct company news (earnings, dividends) = higher signal
  - Sector-wide or regulatory news = moderate signal
  - Indirect effects or minor news = lower signal

**Important**: 
- Focus on **factual analysis**, not speculation
- Consider **entity context** when evaluating impact
- Be **conservative** with confidence scores for ambiguous news
- Distinguish between **sentiment direction** (classification) and **signal quality** (strength)

Now analyze the article above and provide your structured assessment."""
        
        return prompt
    
    def analyze_sentiment(
        self,
        article: NewsArticle,
        entities: Optional[EntityExtractionSchema] = None
    ) -> SentimentAnalysisSchema:
        """
        Analyze article sentiment using LLM with few-shot prompting.
        
        Args:
            article: News article to analyze
            entities: Optional extracted entities for enhanced context
            
        Returns:
            SentimentAnalysisSchema with classification, confidence, and reasoning
            
        Raises:
            LLMServiceError: If sentiment analysis fails
        """
        # Build system message with few-shot examples
        system_message = f"""You are an expert financial sentiment analyst specializing in market impact assessment.

Your task is to analyze financial news articles and determine their sentiment (Bullish, Bearish, or Neutral) along with confidence scores and key factors supporting your analysis.

**Few-Shot Examples**:

{self._build_few_shot_examples()}

---

**Your Analysis Should**:
- Provide clear, actionable sentiment classification
- Include specific, evidence-based key factors
- Assign realistic confidence and signal strength scores
- Consider entity context and market implications

Always return your analysis in the specified structured JSON format."""
        
        # Build analysis prompt
        prompt = self._build_analysis_prompt(article, entities)
        
        try:
            # Call LLM with structured output
            result_dict = self.llm_client.generate_structured_output(
                prompt=prompt,
                schema=SentimentAnalysisSchema,
                system_message=system_message
            )
            
            # Validate with Pydantic
            sentiment_result = SentimentAnalysisSchema.model_validate(result_dict)
            
            return sentiment_result
            
        except Exception as e:
            raise LLMServiceError(f"Sentiment analysis failed for {article.id}: {e}")
    
    def analyze_and_attach(
        self,
        article: NewsArticle,
        entities: Optional[EntityExtractionSchema] = None
    ) -> NewsArticle:
        """
        Convenience method: Analyze sentiment and attach to article.
        
        Args:
            article: News article to analyze
            entities: Optional extracted entities
            
        Returns:
            Article with sentiment data populated
        """
        # Perform LLM sentiment analysis
        sentiment_result = self.analyze_sentiment(article, entities)
        
        # Convert to legacy SentimentData format for backward compatibility
        sentiment_data = SentimentData(
            classification=sentiment_result.classification.value,
            confidence_score=sentiment_result.confidence_score,
            signal_strength=sentiment_result.signal_strength,
            sentiment_breakdown={
                "key_factors": sentiment_result.key_factors,
                "sentiment_percentages": sentiment_result.sentiment_breakdown or {},
                "entity_influence": sentiment_result.entity_influence or {}
            },
            analysis_method="llm",
            timestamp=datetime.now().isoformat()
        )
        
        article.set_sentiment(sentiment_data)
        return article
    
    def batch_analyze(
        self,
        articles: list[NewsArticle],
        entities_list: Optional[list[EntityExtractionSchema]] = None
    ) -> list[NewsArticle]:
        """
        Analyze sentiment for multiple articles.
        
        Args:
            articles: List of articles to analyze
            entities_list: Optional list of entity schemas (must match article order)
            
        Returns:
            List of articles with sentiment attached
        """
        if entities_list and len(entities_list) != len(articles):
            raise ValueError("entities_list must match articles length")
        
        results = []
        for i, article in enumerate(articles):
            entities = entities_list[i] if entities_list else None
            analyzed_article = self.analyze_and_attach(article, entities)
            results.append(analyzed_article)
        
        return results
    
    def get_sentiment_statistics(
        self,
        articles: list[NewsArticle]
    ) -> Dict[str, Any]:
        """
        Calculate sentiment distribution across articles.
        
        Args:
            articles: List of articles with sentiment data
            
        Returns:
            Dict with sentiment statistics
        """
        if not articles:
            return {
                "total_articles": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0
            }
        
        sentiment_counts = {
            "Bullish": 0,
            "Bearish": 0,
            "Neutral": 0
        }
        
        total_confidence = 0.0
        total_signal_strength = 0.0
        analyzed_count = 0
        
        for article in articles:
            if article.has_sentiment():
                sentiment = article.get_sentiment()
                sentiment_counts[sentiment.classification] += 1
                total_confidence += sentiment.confidence_score
                total_signal_strength += sentiment.signal_strength
                analyzed_count += 1
        
        avg_confidence = total_confidence / analyzed_count if analyzed_count > 0 else 0.0
        avg_signal = total_signal_strength / analyzed_count if analyzed_count > 0 else 0.0
        
        return {
            "total_articles": len(articles),
            "analyzed_count": analyzed_count,
            "bullish_count": sentiment_counts["Bullish"],
            "bearish_count": sentiment_counts["Bearish"],
            "neutral_count": sentiment_counts["Neutral"],
            "bullish_percentage": round(sentiment_counts["Bullish"] / analyzed_count * 100, 2) if analyzed_count > 0 else 0,
            "bearish_percentage": round(sentiment_counts["Bearish"] / analyzed_count * 100, 2) if analyzed_count > 0 else 0,
            "neutral_percentage": round(sentiment_counts["Neutral"] / analyzed_count * 100, 2) if analyzed_count > 0 else 0,
            "avg_confidence_score": round(avg_confidence, 2),
            "avg_signal_strength": round(avg_signal, 2),
            "analysis_method": "llm"
        }