from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from app.core.models import NewsArticle, SentimentData
from app.agents.entity_extraction import EntityExtractor
from app.core.config_loader import get_config

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
        """Convert FinBERT raw probs to domain-specific percentage (0-100)."""
        return {
            "bullish": round(self.positive * 100, 2),
            "bearish": round(self.negative * 100, 2),
            "neutral": round(self.neutral * 100, 2)
        }


class FinBERTSentimentClassifier:
    
    def __init__(self, model_name: str = None, entity_extractor: Optional[EntityExtractor] = None, device: Optional[str] = None):
        config = get_config()
        if not FINBERT_AVAILABLE:
            raise ImportError("FinBERT requires transformers and torch.")
        
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.model_name = model_name or config.sentiment_analysis.finbert.model_name
        device = device or config.sentiment_analysis.finbert.device
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading FinBERT model: {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # FinBERT defaults: 0=positive, 1=negative, 2=neutral
        self.id2label = {0: "positive", 1: "negative", 2: "neutral"}
    
    def _predict_sentiment(self, text: str, max_length: int = 512) -> FinBERTScore:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Apply softmax and move to CPU for processing
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]
        
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
            "Companies": [], "Sectors": [], "Regulators": [], "People": [], "Events": []
        }
    
    def _calculate_entity_boost(self, entities: Dict[str, Any]) -> float:
        boost = 1.0
        
        # Increase confidence based on relevant entity presence
        companies = entities.get("Companies", [])
        if companies:
            boost += len(companies) * 0.05
        
        if entities.get("Regulators", []):
            boost += 0.1
        
        events = entities.get("Events", [])
        if events:
            boost += len(events) * 0.05
        
        return min(boost, 1.5)  # Cap boost at 1.5x
    
    def _classify_from_scores(self, scores: Dict[str, float]) -> Tuple[str, float]:
        bullish = scores["bullish"]
        bearish = scores["bearish"]
        neutral = scores["neutral"]
        
        if bullish > bearish and bullish > neutral:
            classification = "Bullish"
            confidence = bullish
        elif bearish > bullish and bearish > neutral:
            classification = "Bearish"
            confidence = bearish
        else:
            classification = "Neutral"
            confidence = neutral
        
        # Adjust confidence based on the margin between the top two scores
        max_score = max(bullish, bearish, neutral)
        second_max = sorted([bullish, bearish, neutral])[-2]
        margin = max_score - second_max
        
        confidence_multiplier = 1.0 + (margin / 100.0)
        final_confidence = min(confidence * confidence_multiplier, 100.0)
        
        return classification, final_confidence
    
    def analyze_sentiment(self, article: NewsArticle) -> Dict[str, Any]:
        text = f"{article.title}. {article.content}"
        
        # Truncate strictly to avoid tokenizer errors (FinBERT limit ~512 tokens)
        if len(text) > 2000:
            text = text[:2000]
        
        finbert_scores = self._predict_sentiment(text)
        normalized_scores = finbert_scores.to_bullish_bearish_neutral()
        
        entities = self._extract_entities_from_article(article)
        entity_boost = self._calculate_entity_boost(entities)
        
        classification, confidence_score = self._classify_from_scores(normalized_scores)
        
        # Signal strength combines raw confidence with entity context relevance
        signal_strength = min(confidence_score * entity_boost, 100.0)
        
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