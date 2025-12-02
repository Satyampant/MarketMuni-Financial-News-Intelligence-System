
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from news_storage import NewsArticle, SentimentData
from entity_extraction import EntityExtractor

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False


@dataclass
class FinBERTScore:
    positive: float
    negative: float
    neutral: float
    
    def to_bullish_bearish_neutral(self) -> Dict[str, float]:
        """Convert FinBERT labels to our domain labels (0-100 scale)"""
        return {
            "bullish": round(self.positive * 100, 2),
            "bearish": round(self.negative * 100, 2),
            "neutral": round(self.neutral * 100, 2)
        }


class FinBERTSentimentClassifier:
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        entity_extractor: Optional[EntityExtractor] = None,
        device: Optional[str] = None
    ):
        if not FINBERT_AVAILABLE:
            raise ImportError(
                "FinBERT requires transformers and torch. "
                "Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.entity_extractor = entity_extractor or EntityExtractor()
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading FinBERT model: {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Label mapping (FinBERT uses: positive, negative, neutral)
        self.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    
    def _predict_sentiment(self, text: str, max_length: int = 512) -> FinBERTScore:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]  # Move to CPU and get first batch
        
        # Map to labels (FinBERT order: positive=0, negative=1, neutral=2)
        return FinBERTScore(
            positive=float(probs[0]),
            negative=float(probs[1]),
            neutral=float(probs[2])
        )
    
    def _extract_entities_from_article(self, article: NewsArticle) -> Dict[str, Any]:
        if hasattr(article, 'entities') and article.entities:
            return article.entities
        
        if self.entity_extractor:
            return self.entity_extractor.extract_entities(article)
        
        return {
            "Companies": [],
            "Sectors": [],
            "Regulators": [],
            "People": [],
            "Events": []
        }
    
    def _calculate_entity_boost(self, entities: Dict[str, Any]) -> float:
        boost = 1.0
        
        # Company mentions increase confidence
        companies = entities.get("Companies", [])
        if companies:
            boost += len(companies) * 0.05
        
        # Regulatory mentions increase confidence
        regulators = entities.get("Regulators", [])
        if regulators:
            boost += 0.1
        
        # Event keywords increase confidence
        events = entities.get("Events", [])
        if events:
            boost += len(events) * 0.05
        
        return min(boost, 1.5)  # Cap at 1.5x
    
    def _classify_from_scores(
        self,
        scores: Dict[str, float]
    ) -> Tuple[str, float]:

        bullish = scores["bullish"]
        bearish = scores["bearish"]
        neutral = scores["neutral"]
        
        # Determine winner
        if bullish > bearish and bullish > neutral:
            classification = "Bullish"
            confidence = bullish
        elif bearish > bullish and bearish > neutral:
            classification = "Bearish"
            confidence = bearish
        else:
            classification = "Neutral"
            confidence = neutral
        
        # Adjust confidence based on margin
        max_score = max(bullish, bearish, neutral)
        second_max = sorted([bullish, bearish, neutral])[-2]
        margin = max_score - second_max
        
        # Higher margin = higher confidence
        confidence_multiplier = 1.0 + (margin / 100.0)
        final_confidence = min(confidence * confidence_multiplier, 100.0)
        
        return classification, final_confidence
    
    def analyze_sentiment(self, article: NewsArticle) -> Dict[str, Any]:
        # Combine title and content (title weighted more)
        text = f"{article.title}. {article.content}"
        
        # Truncate if too long (FinBERT max = 512 tokens ≈ 400 words)
        if len(text) > 2000:
            text = text[:2000]
        
        # Run FinBERT inference
        finbert_scores = self._predict_sentiment(text)
        
        # Convert to 0-100 scale with our labels
        normalized_scores = finbert_scores.to_bullish_bearish_neutral()
        
        # Extract entities for context boosting
        entities = self._extract_entities_from_article(article)
        entity_boost = self._calculate_entity_boost(entities)
        
        # Classify sentiment
        classification, confidence_score = self._classify_from_scores(normalized_scores)
        
        # Calculate signal strength (confidence * entity boost)
        signal_strength = min(confidence_score * entity_boost, 100.0)
        
        # Build detailed breakdown
        sentiment_breakdown = {
            "bullish": normalized_scores["bullish"],
            "bearish": normalized_scores["bearish"],
            "neutral": normalized_scores["neutral"],
            "entity_boost": round(entity_boost, 2),
            "raw_positive_prob": round(finbert_scores.positive, 4),
            "raw_negative_prob": round(finbert_scores.negative, 4),
            "raw_neutral_prob": round(finbert_scores.neutral, 4)
        }
        
        return {
            "classification": classification,
            "confidence_score": round(confidence_score, 2),
            "signal_strength": round(signal_strength, 2),
            "sentiment_breakdown": sentiment_breakdown,
            "analysis_method": "finbert"
        }
    
    def analyze_and_attach(self, article: NewsArticle) -> NewsArticle:
    
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


# Testing
if __name__ == "__main__":
    from datetime import datetime
    
    if not FINBERT_AVAILABLE:
        print("ERROR: FinBERT requires transformers and torch")
        print("Install with: pip install transformers torch")
        exit(1)
    
    classifier = FinBERTSentimentClassifier()
    
    test_articles = [
        NewsArticle(
            id="TEST_FINBERT_001",
            title="HDFC Bank announces 15% dividend, board approves stock buyback",
            content="HDFC Bank Ltd announced a 15% dividend payout and approved a stock buyback program worth Rs 5,000 crores.",
            source="Test",
            timestamp=datetime.now()
        ),
        NewsArticle(
            id="TEST_FINBERT_002",
            title="Banking sector losses narrow significantly",
            content="The banking sector reported losses that narrowed by 40% compared to last quarter, signaling recovery.",
            source="Test",
            timestamp=datetime.now()
        ),
        NewsArticle(
            id="TEST_FINBERT_003",
            title="Tech layoffs surge as companies cut costs",
            content="Major IT companies announced massive layoffs, with revenues plummeting amid economic uncertainty.",
            source="Test",
            timestamp=datetime.now()
        ),
        NewsArticle(
            id="TEST_FINBERT_004",
            title="RBI policy meeting scheduled for next week",
            content="The Reserve Bank of India will hold its monetary policy committee meeting next week to review rates.",
            source="Test",
            timestamp=datetime.now()
        )
    ]
    
    print("=" * 80)
    print("FinBERT Sentiment Analysis Tests")
    print("=" * 80)
    
    for article in test_articles:
        print(f"\n[{article.id}]")
        print(f"Title: {article.title}")
        result = classifier.analyze_sentiment(article)
        
        print(f"Classification: {result['classification']}")
        print(f"Confidence: {result['confidence_score']:.2f}")
        print(f"Signal Strength: {result['signal_strength']:.2f}")
        print(f"Breakdown: {result['sentiment_breakdown']}")