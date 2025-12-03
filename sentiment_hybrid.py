"""
Hybrid Sentiment Classifier - Combines Rule-Based + FinBERT
Production-grade sentiment analysis with fallback mechanisms
"""

from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
from news_storage import NewsArticle, SentimentData
from entity_extraction import EntityExtractor

# Import rule-based classifier
from sentiment_classifier import RuleBasedSentimentClassifier

# Import FinBERT with graceful fallback
try:
    from sentiment_finbert import FinBERTSentimentClassifier, FINBERT_AVAILABLE
except ImportError:
    FINBERT_AVAILABLE = False
    FinBERTSentimentClassifier = None


SentimentMethod = Literal["rule_based", "finbert", "hybrid"]


@dataclass
class HybridSentimentResult:
    """Combined sentiment result from multiple methods"""
    classification: str
    confidence_score: float
    signal_strength: float
    sentiment_breakdown: Dict[str, float]
    analysis_method: str
    rule_based_result: Optional[Dict[str, Any]] = None
    finbert_result: Optional[Dict[str, Any]] = None
    agreement_score: Optional[float] = None  # How much do both methods agree?


class HybridSentimentClassifier:
    """
    Hybrid sentiment classifier that combines rule-based and FinBERT approaches.
    
    Strategy:
    - Rule-Based: Fast, interpretable, works immediately (baseline)
    - FinBERT: High accuracy, understands context (enhancement)
    - Hybrid: Combines both for maximum accuracy
    
    Fallback: If FinBERT unavailable, uses rule-based only
    """
    
    def __init__(
        self,
        method: SentimentMethod = "hybrid",
        entity_extractor: Optional[EntityExtractor] = None,
        finbert_model: str = "ProsusAI/finbert",
        finbert_weight: float = 0.7,  # Weight for FinBERT in hybrid mode
        rule_weight: float = 0.3  # Weight for rule-based in hybrid mode
    ):
        """
        Initialize hybrid classifier.
        
        Args:
            method: "rule_based", "finbert", or "hybrid"
            entity_extractor: EntityExtractor instance
            finbert_model: FinBERT model name
            finbert_weight: Weight for FinBERT scores in hybrid mode (0-1)
            rule_weight: Weight for rule-based scores in hybrid mode (0-1)
        """
        self.method = method
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.finbert_weight = finbert_weight
        self.rule_weight = rule_weight
        
        # Initialize rule-based classifier (always available)
        self.rule_classifier = RuleBasedSentimentClassifier(
            entity_extractor=self.entity_extractor
        )
        
        # Initialize FinBERT classifier (if available and needed)
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
        """
        Calculate agreement between rule-based and FinBERT.
        
        Returns:
            Agreement score (0-1), where 1 = perfect agreement
        """
        # Check if classifications match
        rule_class = rule_result["classification"]
        finbert_class = finbert_result["classification"]
        
        classification_match = 1.0 if rule_class == finbert_class else 0.0
        
        # Check score distribution similarity
        rule_breakdown = rule_result["sentiment_breakdown"]
        finbert_breakdown = finbert_result["sentiment_breakdown"]
        
        # Calculate cosine similarity of score vectors
        import math
        
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
        
        # Weighted combination
        agreement = 0.6 * classification_match + 0.4 * score_similarity
        
        return round(agreement, 3)
    
    def _combine_results(
        self,
        rule_result: Dict[str, Any],
        finbert_result: Dict[str, Any]
    ) -> HybridSentimentResult:
        """
        Combine rule-based and FinBERT results using weighted averaging.
        
        Strategy:
        - If both agree: High confidence, use weighted average
        - If disagree: Lower confidence, favor FinBERT slightly
        """
        # Calculate agreement
        agreement = self._calculate_agreement_score(rule_result, finbert_result)
        
        # Extract breakdowns
        rule_bd = rule_result["sentiment_breakdown"]
        finbert_bd = finbert_result["sentiment_breakdown"]
        
        # Weighted combination of scores
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
        
        # Determine classification from combined scores
        max_score = max(combined_scores.values())
        if combined_scores["bullish"] == max_score:
            classification = "Bullish"
        elif combined_scores["bearish"] == max_score:
            classification = "Bearish"
        else:
            classification = "Neutral"
        
        # Confidence is the max score, adjusted by agreement
        base_confidence = max_score
        
        # High agreement boosts confidence, low agreement reduces it
        agreement_boost = 0.9 + (agreement * 0.2)  # Range: 0.9-1.1
        final_confidence = min(base_confidence * agreement_boost, 100.0)
        
        # Signal strength combines both methods
        rule_signal = rule_result["signal_strength"]
        finbert_signal = finbert_result["signal_strength"]
        combined_signal = (
            self.rule_weight * rule_signal +
            self.finbert_weight * finbert_signal
        ) * agreement_boost
        combined_signal = min(combined_signal, 100.0)
        
        # Enhanced breakdown with metadata
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
        """
        Analyze sentiment using configured method.
        
        Args:
            article: NewsArticle to analyze
        
        Returns:
            Dictionary with sentiment analysis results
        """
        if self.method == "rule_based":
            # Use rule-based only
            result = self.rule_classifier.analyze_sentiment(article)
            return result
        
        elif self.method == "finbert":
            # Use FinBERT only
            if self.finbert_classifier is None:
                # Fallback to rule-based
                result = self.rule_classifier.analyze_sentiment(article)
                result["analysis_method"] = "rule_based_fallback"
                return result
            
            result = self.finbert_classifier.analyze_sentiment(article)
            return result
        
        else:  # method == "hybrid"
            # Use both and combine
            rule_result = self.rule_classifier.analyze_sentiment(article)
            
            if self.finbert_classifier is None:
                # Fallback to rule-based
                rule_result["analysis_method"] = "rule_based_fallback"
                return rule_result
            
            finbert_result = self.finbert_classifier.analyze_sentiment(article)
            
            # Combine results
            hybrid_result = self._combine_results(rule_result, finbert_result)
            
            # Convert to dict format
            return {
                "classification": hybrid_result.classification,
                "confidence_score": hybrid_result.confidence_score,
                "signal_strength": hybrid_result.signal_strength,
                "sentiment_breakdown": hybrid_result.sentiment_breakdown,
                "analysis_method": hybrid_result.analysis_method,
                "agreement_score": hybrid_result.agreement_score
            }
    
    def analyze_and_attach(self, article: NewsArticle) -> NewsArticle:
        """
        Analyze sentiment and attach to article.
        
        Args:
            article: NewsArticle to process
        
        Returns:
            Article with sentiment data attached
        """
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
        """Get information about active sentiment analysis method"""
        return {
            "configured_method": self.method,
            "rule_based_available": True,
            "finbert_available": self.finbert_classifier is not None,
            "weights": {
                "rule_based": self.rule_weight,
                "finbert": self.finbert_weight
            } if self.method == "hybrid" else None
        }

