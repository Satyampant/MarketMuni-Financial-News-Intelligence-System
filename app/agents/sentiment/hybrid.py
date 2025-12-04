from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from app.core.models import NewsArticle, SentimentData
from app.agents.entity_extraction import EntityExtractor
from app.agents.sentiment.classifier import RuleBasedSentimentClassifier
from app.core.config_loader import get_config
import math

try:
    from app.agents.sentiment.finbert import FinBERTSentimentClassifier, FINBERT_AVAILABLE
except ImportError:
    FINBERT_AVAILABLE = False
    FinBERTSentimentClassifier = None

SentimentMethod = Literal["rule_based", "finbert", "hybrid"]

@dataclass
class HybridSentimentResult:
    """Combined sentiment result from multiple methods."""
    classification: str
    confidence_score: float
    signal_strength: float
    sentiment_breakdown: Dict[str, float]
    analysis_method: str
    rule_based_result: Optional[Dict[str, Any]] = None
    finbert_result: Optional[Dict[str, Any]] = None
    agreement_score: Optional[float] = None

class HybridSentimentClassifier:
    
    def __init__(
        self,
        method: SentimentMethod = None,
        entity_extractor: Optional[EntityExtractor] = None,
        finbert_model: str = None,
        finbert_weight: float = None,
        rule_weight: float = None
    ):
        config = get_config()
        self.method = method or config.sentiment_analysis.method
        finbert_model = finbert_model or config.sentiment_analysis.finbert.model_name
        self.entity_extractor = entity_extractor or EntityExtractor()

        if finbert_weight is None:
            finbert_weight = config.sentiment_analysis.hybrid_weights.finbert_weight
        if rule_weight is None:
            rule_weight = config.sentiment_analysis.hybrid_weights.rule_weight
        
        self.finbert_weight = finbert_weight
        self.rule_weight = rule_weight
        
        self.rule_classifier = RuleBasedSentimentClassifier(
            entity_extractor=self.entity_extractor
        )
        
        self.finbert_classifier = None
        if method in ["finbert", "hybrid"] and FINBERT_AVAILABLE:
            try:
                self.finbert_classifier = FinBERTSentimentClassifier(
                    model_name=finbert_model,
                    entity_extractor=self.entity_extractor
                )
                print(f"✓ Hybrid classifier initialized with method: {method}")
            except Exception as e:
                print(f"⚠ FinBERT initialization failed: {e}")
                print("  Falling back to rule-based only")
                self.method = "rule_based"
        elif method in ["finbert", "hybrid"]:
            print(f"⚠ FinBERT not available (missing transformers/torch)")
            print("  Falling back to rule-based only")
            self.method = "rule_based"
    
    def _calculate_agreement_score(
        self,
        rule_result: Dict[str, Any],
        finbert_result: Dict[str, Any]
    ) -> float:
        """Returns agreement score (0-1) based on classification match and vector cosine similarity."""
        rule_class = rule_result["classification"]
        finbert_class = finbert_result["classification"]
        
        classification_match = 1.0 if rule_class == finbert_class else 0.0
        
        rule_breakdown = rule_result["sentiment_breakdown"]
        finbert_breakdown = finbert_result["sentiment_breakdown"]
        
        def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
            keys = ["bullish", "bearish", "neutral"]
            v1 = [vec1.get(k, 0) for k in keys]
            v2 = [vec2.get(k, 0) for k in keys]
            
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        score_similarity = cosine_similarity(rule_breakdown, finbert_breakdown)
        
        # Weighted combination: 60% hard match, 40% vector similarity
        agreement = 0.6 * classification_match + 0.4 * score_similarity
        
        return round(agreement, 3)
    
    def _combine_results(
        self,
        rule_result: Dict[str, Any],
        finbert_result: Dict[str, Any]
    ) -> HybridSentimentResult:
        """Merges results using weighted averaging; confidence is boosted by model agreement."""
        agreement = self._calculate_agreement_score(rule_result, finbert_result)
        
        rule_bd = rule_result["sentiment_breakdown"]
        finbert_bd = finbert_result["sentiment_breakdown"]
        
        combined_scores = {
            "bullish": (
                self.rule_weight * rule_bd["bullish"] +
                self.finbert_weight * finbert_bd["bullish"]
            ),
            "bearish": (
                self.rule_weight * rule_bd["bearish"] +
                self.finbert_weight * finbert_bd["bearish"]
            ),
            "neutral": (
                self.rule_weight * rule_bd["neutral"] +
                self.finbert_weight * finbert_bd["neutral"]
            )
        }
        
        max_score = max(combined_scores.values())
        if combined_scores["bullish"] == max_score:
            classification = "Bullish"
        elif combined_scores["bearish"] == max_score:
            classification = "Bearish"
        else:
            classification = "Neutral"
        
        # Agreement acts as a multiplier (0.9x to 1.1x) on base confidence
        agreement_boost = 0.9 + (agreement * 0.2)
        final_confidence = min(max_score * agreement_boost, 100.0)
        
        rule_signal = rule_result["signal_strength"]
        finbert_signal = finbert_result["signal_strength"]
        combined_signal = (
            self.rule_weight * rule_signal +
            self.finbert_weight * finbert_signal
        ) * agreement_boost
        combined_signal = min(combined_signal, 100.0)
        
        enhanced_breakdown = {
            **combined_scores,
            "agreement_score": agreement,
            "rule_weight": self.rule_weight,
            "finbert_weight": self.finbert_weight,
            "rule_classification": rule_result["classification"],
            "finbert_classification": finbert_result["classification"]
        }
        
        return HybridSentimentResult(
            classification=classification,
            confidence_score=round(final_confidence, 2),
            signal_strength=round(combined_signal, 2),
            sentiment_breakdown=enhanced_breakdown,
            analysis_method="hybrid",
            rule_based_result=rule_result,
            finbert_result=finbert_result,
            agreement_score=agreement
        )
    
    def analyze_sentiment(self, article: NewsArticle) -> Dict[str, Any]:
        """Analyzes article using rule-based, FinBERT, or hybrid approach depending on config."""
        if self.method == "rule_based":
            return self.rule_classifier.analyze_sentiment(article)
        
        elif self.method == "finbert":
            if self.finbert_classifier is None:
                result = self.rule_classifier.analyze_sentiment(article)
                result["analysis_method"] = "rule_based_fallback"
                return result
            
            return self.finbert_classifier.analyze_sentiment(article)
        
        else:  # hybrid
            rule_result = self.rule_classifier.analyze_sentiment(article)
            
            if self.finbert_classifier is None:
                rule_result["analysis_method"] = "rule_based_fallback"
                return rule_result
            
            finbert_result = self.finbert_classifier.analyze_sentiment(article)
            hybrid_result = self._combine_results(rule_result, finbert_result)
            
            return {
                "classification": hybrid_result.classification,
                "confidence_score": hybrid_result.confidence_score,
                "signal_strength": hybrid_result.signal_strength,
                "sentiment_breakdown": hybrid_result.sentiment_breakdown,
                "analysis_method": hybrid_result.analysis_method,
                "agreement_score": hybrid_result.agreement_score
            }
    
    def analyze_and_attach(self, article: NewsArticle) -> NewsArticle:
        """Runs analysis and updates the article object with sentiment data."""
        result = self.analyze_sentiment(article)
        
        sentiment_data = SentimentData(
            classification=result["classification"],
            confidence_score=result["confidence_score"],
            signal_strength=result["signal_strength"],
            sentiment_breakdown=result["sentiment_breakdown"],
            analysis_method=result["analysis_method"]
        )
        
        article.set_sentiment(sentiment_data)
        return article
    
    def get_method_info(self) -> Dict[str, Any]:
        return {
            "configured_method": self.method,
            "rule_based_available": True,
            "finbert_available": self.finbert_classifier is not None,
            "weights": {
                "rule_based": self.rule_weight,
                "finbert": self.finbert_weight
            } if self.method == "hybrid" else None
        }