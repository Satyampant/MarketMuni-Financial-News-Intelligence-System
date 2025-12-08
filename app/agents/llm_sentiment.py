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
    """LLM-powered sentiment analysis using few-shot prompting."""
    
    def __init__(
        self,
        llm_client: Optional[GroqLLMClient] = None,
        use_entity_context: bool = True
    ):
        self.config = get_config()
        
        # specific reasoning model setup if client not provided
        if llm_client is None:
            self.llm_client = GroqLLMClient(
                model=self.config.llm.models.reasoning,
                temperature=0.1,
                max_tokens=2048
            )
        else:
            self.llm_client = llm_client
        
        self.use_entity_context = use_entity_context
        
        print(f"✓ LLMSentimentAnalyzer initialized")
        print(f"  - Model: {self.llm_client.model}")
        print(f"  - Entity context: {'enabled' if use_entity_context else 'disabled'}")
    
    def _build_few_shot_examples(self) -> str:
        return self.config.prompts.sentiment_analysis.get('few_shot_examples', '')
    
    def _format_entity_context(self, entities: EntityExtractionSchema) -> str:
        """Format entity extraction results into a context string."""
        if not entities:
            return "No entities extracted"
        
        context_parts = []
        
        if entities.companies:
            companies_str = ", ".join([
                f"{c.name}" + (f" ({c.ticker_symbol})" if c.ticker_symbol else "")
                for c in entities.companies
            ])
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
        
        return ", ".join(context_parts) if context_parts else "No key entities identified"
    
    def _build_analysis_prompt(
        self,
        article: NewsArticle,
        entities: Optional[EntityExtractionSchema] = None
    ) -> str:
        """Construct the main prompt using article content and entity context."""
        prompt_template = self.config.prompts.sentiment_analysis.get('task_prompt', '')
        
        entity_context = ""
        if self.use_entity_context and entities:
            entity_context = f"\n**Entities**: {self._format_entity_context(entities)}"
        
        return prompt_template.format(
            title=article.title,
            content=article.content,
            entity_context=entity_context
        )
    
    def analyze_sentiment(
        self,
        article: NewsArticle,
        entities: Optional[EntityExtractionSchema] = None
    ) -> SentimentAnalysisSchema:
        """Execute LLM analysis and validate against Pydantic schema."""
        system_message_template = self.config.prompts.sentiment_analysis.get('system_message', '')
        few_shot_examples = self._build_few_shot_examples()
        system_message = system_message_template.format(few_shot_examples=few_shot_examples)
        
        prompt = self._build_analysis_prompt(article, entities)
        
        try:
            result_dict = self.llm_client.generate_structured_output(
                prompt=prompt,
                schema=SentimentAnalysisSchema,
                system_message=system_message
            )
            
            sentiment_result = SentimentAnalysisSchema.model_validate(result_dict)
            return sentiment_result
            
        except Exception as e:
            raise LLMServiceError(f"Sentiment analysis failed for {article.id}: {e}")
    
    def analyze_and_attach(
        self,
        article: NewsArticle,
        entities: Optional[EntityExtractionSchema] = None
    ) -> NewsArticle:
        """Analyze sentiment and attach result to the article object."""
        sentiment_result = self.analyze_sentiment(article, entities)
        
        # Map to legacy format for backward compatibility
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
        """Run sentiment analysis on a list of articles."""
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
        """Calculate sentiment distribution and average confidence."""
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