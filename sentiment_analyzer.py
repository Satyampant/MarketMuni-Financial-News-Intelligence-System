from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Any
import os


class SentimentClassification(Enum):
    """Sentiment classification for financial news"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class SentimentScore:
    """
    Comprehensive sentiment analysis result for financial news.
    
    Attributes:
        classification: Overall sentiment direction (BULLISH/BEARISH/NEUTRAL)
        confidence: Model confidence in classification (0-100 scale)
        signal_strength: Trader actionability score (0-10 scale)
        reasoning: Explainability text for the sentiment decision
        price_impact_probability: Predicted likelihood of price movement (0-1)
        sentiment_factors: Optional dict of contributing factors
    """
    classification: SentimentClassification
    confidence: float  # 0-100 scale
    signal_strength: float  # 0-10 scale (trader actionability)
    reasoning: str  # Explainability text
    price_impact_probability: float  # 0-1 scale
    sentiment_factors: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate score ranges"""
        if not 0 <= self.confidence <= 100:
            raise ValueError(f"Confidence must be 0-100, got {self.confidence}")
        
        if not 0 <= self.signal_strength <= 10:
            raise ValueError(f"Signal strength must be 0-10, got {self.signal_strength}")
        
        if not 0 <= self.price_impact_probability <= 1:
            raise ValueError(f"Price impact probability must be 0-1, got {self.price_impact_probability}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = asdict(self)
        result['classification'] = self.classification.value
        return result
    
    def is_actionable(self, min_confidence: float = 70.0, min_signal: float = 6.0) -> bool:
        """
        Determine if sentiment is actionable for trading decisions.
        
        Args:
            min_confidence: Minimum confidence threshold (default 70%)
            min_signal: Minimum signal strength threshold (default 6/10)
        
        Returns:
            True if sentiment meets actionability criteria
        """
        return (
            self.confidence >= min_confidence and
            self.signal_strength >= min_signal and
            self.classification != SentimentClassification.NEUTRAL
        )
    
    def get_sentiment_label(self) -> str:
        """Get human-readable sentiment label with strength"""
        if self.classification == SentimentClassification.NEUTRAL:
            return "Neutral"
        
        # Add strength modifier based on signal strength
        if self.signal_strength >= 8:
            modifier = "Strongly"
        elif self.signal_strength >= 6:
            modifier = "Moderately"
        else:
            modifier = "Weakly"
        
        return f"{modifier} {self.classification.value.capitalize()}"


@dataclass
class SentimentConfig:
    """
    Configuration for sentiment analysis using Groq LLM.
    
    Attributes:
        groq_model: Groq model identifier
        groq_api_key: API key for Groq (loaded from environment)
        temperature: Sampling temperature for consistency
        max_tokens: Maximum tokens for LLM response
        prompt_template: System prompt template for sentiment analysis
    """
    groq_model: str = "mixtral-8x7b-32768"
    groq_api_key: str = None
    temperature: float = 0.1  # Low temperature for consistency
    max_tokens: int = 1024
    prompt_template: str = None
    
    def __post_init__(self):
        """Load API key from environment and set default prompt"""
        # Load API key from environment if not provided
        if self.groq_api_key is None:
            self.groq_api_key = os.getenv("GROQ_API_KEY", "")
            if not self.groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY not found. Set it via environment variable or pass it explicitly."
                )
        
        # Set default prompt template if not provided
        if self.prompt_template is None:
            self.prompt_template = self._get_default_prompt()
    
    @staticmethod
    def _get_default_prompt() -> str:
        """Get default system prompt for sentiment analysis"""
        return """You are a professional financial sentiment analyst. Analyze the given financial news article and provide a comprehensive sentiment assessment.

Your analysis must include:
1. **Classification**: BULLISH, BEARISH, or NEUTRAL
2. **Confidence**: Your confidence in the classification (0-100 scale)
3. **Signal Strength**: Trader actionability score (0-10 scale)
   - 0-3: Low actionability (minor news, unclear impact)
   - 4-6: Medium actionability (notable news, moderate clarity)
   - 7-10: High actionability (major news, clear directional impact)
4. **Reasoning**: Clear explanation of your sentiment decision, citing specific facts from the article
5. **Price Impact Probability**: Likelihood of near-term price movement (0-1 scale)

**Guidelines:**
- BULLISH: Positive developments (profit growth, expansion, regulatory approval, dividends, buybacks)
- BEARISH: Negative developments (losses, regulatory issues, downgrades, scandals)
- NEUTRAL: Mixed signals, routine updates, or unclear directional impact
- Consider both direct mentions and sector-wide implications
- Weight recent, material events higher than generic statements
- Factor in the news source credibility and specificity

**Output Format (JSON):**
{
  "classification": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <0-100>,
  "signal_strength": <0-10>,
  "reasoning": "<detailed explanation>",
  "price_impact_probability": <0-1>,
  "sentiment_factors": {
    "primary_driver": "<main reason>",
    "supporting_factors": ["<factor1>", "<factor2>"],
    "risk_factors": ["<risk1>", "<risk2>"]
  }
}

Be precise, data-driven, and conservative in your assessments. Only assign high signal strength when the news clearly indicates actionable price movement."""
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if not self.groq_api_key:
            return False
        
        if not 0 <= self.temperature <= 2:
            return False
        
        if self.max_tokens < 100:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (excluding API key for security)"""
        return {
            "groq_model": self.groq_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "has_api_key": bool(self.groq_api_key)
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Sentiment Data Model & Configuration Tests")
    print("=" * 60)
    
    # Test 1: Create sentiment score
    print("\n[Test 1] Create SentimentScore")
    score = SentimentScore(
        classification=SentimentClassification.BULLISH,
        confidence=85.5,
        signal_strength=7.5,
        reasoning="Strong earnings growth and dividend announcement indicate positive momentum",
        price_impact_probability=0.75,
        sentiment_factors={
            "primary_driver": "15% profit growth",
            "supporting_factors": ["dividend increase", "sector outperformance"],
            "risk_factors": ["market volatility"]
        }
    )
    
    print(f"Classification: {score.classification.value}")
    print(f"Confidence: {score.confidence}%")
    print(f"Signal Strength: {score.signal_strength}/10")
    print(f"Label: {score.get_sentiment_label()}")
    print(f"Actionable: {score.is_actionable()}")
    print(f"Price Impact: {score.price_impact_probability}")
    
    # Test 2: Convert to dictionary
    print("\n[Test 2] Convert to Dictionary")
    score_dict = score.to_dict()
    print(f"Dict keys: {list(score_dict.keys())}")
    print(f"Classification type: {type(score_dict['classification'])}")
    
    # Test 3: Validation
    print("\n[Test 3] Validation Tests")
    try:
        invalid_score = SentimentScore(
            classification=SentimentClassification.NEUTRAL,
            confidence=150,  # Invalid
            signal_strength=5.0,
            reasoning="Test",
            price_impact_probability=0.5
        )
    except ValueError as e:
        print(f"✓ Caught invalid confidence: {e}")
    
    # Test 4: Create configuration (requires GROQ_API_KEY in environment)
    print("\n[Test 4] Create SentimentConfig")
    try:
        # Set a dummy key for testing if not in environment
        if not os.getenv("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = "dummy_key_for_testing"
        
        config = SentimentConfig()
        print(f"Model: {config.groq_model}")
        print(f"Temperature: {config.temperature}")
        print(f"Max Tokens: {config.max_tokens}")
        print(f"Has API Key: {bool(config.groq_api_key)}")
        print(f"Valid Config: {config.validate()}")
        print(f"\nPrompt Template Length: {len(config.prompt_template)} chars")
        print(f"Prompt Preview: {config.prompt_template[:200]}...")
        
    except ValueError as e:
        print(f"Config error: {e}")
    
    # Test 5: Different sentiment classifications
    print("\n[Test 5] Different Classifications")
    
    test_cases = [
        ("BULLISH", 90, 8.5, "Strong buy signal"),
        ("BEARISH", 75, 7.0, "Clear sell indicator"),
        ("NEUTRAL", 60, 3.0, "No clear direction"),
    ]
    
    for classification, conf, signal, reason in test_cases:
        score = SentimentScore(
            classification=SentimentClassification[classification],
            confidence=conf,
            signal_strength=signal,
            reasoning=reason,
            price_impact_probability=signal / 10
        )
        print(f"{classification}: {score.get_sentiment_label()} (Actionable: {score.is_actionable()})")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)