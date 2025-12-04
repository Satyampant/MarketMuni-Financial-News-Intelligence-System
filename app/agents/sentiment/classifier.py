from app.core.config import Paths
"""
Rule-Based Sentiment Classifier for Financial News
"""

from typing import Dict, Any, Set, List, Tuple, Optional
import re
from dataclasses import dataclass
from pathlib import Path
from app.core.models import NewsArticle, SentimentData
from app.agents.entity_extraction import EntityExtractor
from app.core.config_loader import get_config

try:
    import spacy
    from spacy.tokens import Token, Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass
class SentimentScore:
    """Internal scoring structure."""
    bullish: float = 0.0
    bearish: float = 0.0
    neutral: float = 0.0
    
    @property
    def total(self) -> float:
        return self.bullish + self.bearish + self.neutral
    
    def normalize(self) -> Dict[str, float]:
        if self.total == 0:
            return {"bullish": 33.33, "bearish": 33.33, "neutral": 33.34}
        
        return {
            "bullish": round((self.bullish / self.total) * 100, 2),
            "bearish": round((self.bearish / self.total) * 100, 2),
            "neutral": round((self.neutral / self.total) * 100, 2)
        }


class RuleBasedSentimentClassifier:
    
    # Financial lexicons (Loughran-McDonald + Domain Knowledge)
    
    # Weight: 3.0
    STRONG_BULLISH = {
        "surge", "soar", "skyrocket", "boom", "breakthrough", "record",
        "outperform", "exceed", "beat", "stellar", "robust", "exceptional",
        "blockbuster", "rally", "upside", "gain", "achieve"
    }
    
    # Weight: 3.0
    STRONG_BEARISH = {
        "plunge", "crash", "collapse", "bankruptcy", "fraud", "scandal",
        "default", "plummet", "nosedive", "catastrophic", "crisis",
        "downgrade", "miss", "fail", "violation", "penalty"
    }
    
    # Weight: 2.0
    MODERATE_BULLISH = {
        "growth", "profit", "gain", "expansion", "increase", "rise",
        "dividend", "buyback", "acquisition", "upgrade", "positive",
        "strong", "improve", "recover", "opportunity", "favorable",
        "strength", "advance", "progress", "success", "enhance",
        "accelerate", "outpace", "momentum", "rebound", "resilient", "windfall", 
        "margin expansion", "market share gain", "capacity expansion"
    }
    
    # Weight: 2.0
    MODERATE_BEARISH = {
        "decline", "loss", "fall", "drop", "layoff", "concern",
        "risk", "slowdown", "contraction", "cut", "reduce", "lower",
        "deteriorate", "struggle", "weak", "negative", "downside",
        "pressure", "challenge", "headwind", "uncertainty",
        "decelerate", "underperform", "headwinds", "writedown", "provision", 
        "margin compression", "market share loss", "capacity constraint"
    }
    
    # Weight: 1.0
    WEAK_BULLISH = {
        "optimistic", "hopeful", "potential", "stable", "steady",
        "maintain", "hold", "sustain", "support", "modest"
    }
    
    # Weight: 1.0
    WEAK_BEARISH = {
        "cautious", "uncertain", "volatile", "mixed", "soften",
        "ease", "temper", "moderate", "restrain", "limit"
    }
    
    # Weight: 1.0
    NEUTRAL_KEYWORDS = {
        "announce", "report", "schedule", "plan", "state",
        "say", "indicate", "reveal", "disclose", "confirm",
        "meeting", "conference", "presentation", "statement",
        "file", "submit", "publish", "release"
    }
    
    # Event modifiers (multipliers)
    EVENT_MODIFIERS = {
        "dividend": 1.3, "buyback": 1.3, "stock buyback": 1.3,
        "merger": 1.1, "acquisition": 1.1, "ipo": 1.2,
        "earnings": 1.2, "profit": 1.2, "loss": 1.2,
        "layoff": 1.4, "bankruptcy": 1.5, "repo rate": 1.3,
        "interest rate": 1.3, "rates": 1.2, "policy rate": 1.3,
        "quarterly results": 1.2, "revenue": 1.1, "rights issue": 1.2,
        "delisting": 1.4, "share buyback": 1.3, "bonus issue": 1.2,
        "split": 1.1, "consolidation": 1.1, "restructuring": 1.2,
        "divestment": 1.1, "capital raising": 1.2
    }
    
    ENTITY_WEIGHTS = {
        "company": 1.3,
        "sector": 1.1,
        "regulator": 1.2
    }
    
    def __init__(self, entity_extractor: EntityExtractor = None, use_spacy: bool = None, model_name: str = None):
        config = get_config()
        
        self.entity_weights = config.sentiment_analysis.entity_weights or ENTITY_WEIGHTS
        self.event_modifiers = config.sentiment_analysis.event_modifiers or ENTITY_MODIFIERS
        
        if use_spacy is None:
            use_spacy = config.sentiment_analysis.rule_based.use_spacy
        
        if model_name is None:
            model_name = config.sentiment_analysis.rule_based.spacy_model

        self.extractor = entity_extractor or EntityExtractor()
        self.nlp = None
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        if self.use_spacy:
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    self.nlp = spacy.blank("en")
                    self.use_spacy = False
        
        self._build_inverted_index()
    
    def _build_inverted_index(self):
        """Build O(n) lookup map for keywords."""
        self.inverted_index = {}
        
        lexicon_map = [
            (self.STRONG_BULLISH, "bullish", 3.0),
            (self.STRONG_BEARISH, "bearish", 3.0),
            (self.MODERATE_BULLISH, "bullish", 2.0),
            (self.MODERATE_BEARISH, "bearish", 2.0),
            (self.WEAK_BULLISH, "bullish", 1.0),
            (self.WEAK_BEARISH, "bearish", 1.0),
            (self.NEUTRAL_KEYWORDS, "neutral", 1.0)
        ]
        
        for lexicon, sentiment, weight in lexicon_map:
            for keyword in lexicon:
                self.inverted_index[keyword.lower()] = (sentiment, weight)
                
                # Handle common variations manually
                if keyword.endswith('e'):
                    self.inverted_index[keyword[:-1] + 'ing'] = (sentiment, weight)
                    self.inverted_index[keyword + 'd'] = (sentiment, weight)
    
    def _detect_negation_spacy(self, token: Token) -> bool:
        """Check for negation dependencies or ancestors."""
        if token.dep_ == "neg":
            return True
        
        for child in token.children:
            if child.dep_ == "neg":
                return True
        
        if token.head.dep_ == "neg":
            return True
        
        # Check ancestors for common negation patterns (within 4 tokens)
        for ancestor in token.ancestors:
            if ancestor.lemma_ in {"not", "no", "never", "neither", "nor", "none"}:
                if token.i > ancestor.i and token.i - ancestor.i <= 4:
                    return True
        
        return False
    
    def _detect_negation_simple(self, text: str, pos: int) -> bool:
        # Check previous ~20 chars (roughly 3-4 words) for negation terms
        window_start = max(0, pos - 20)
        window = text[window_start:pos].lower()
        
        negation_words = {"not", "no", "never", "neither", "nor", "none", "without"}
        
        for neg_word in negation_words:
            if f" {neg_word} " in f" {window} ":
                return True
        
        return False
    
    def _analyze_with_spacy(self, text: str) -> SentimentScore:
        doc = self.nlp(text)
        score = SentimentScore()
        
        for token in doc:
            # Check lemma or exact text against index
            sentiment_info = self.inverted_index.get(token.lemma_.lower()) or \
                             self.inverted_index.get(token.text.lower())
            
            if sentiment_info:
                sentiment_type, weight = sentiment_info
                is_negated = self._detect_negation_spacy(token)
                
                if is_negated:
                    if sentiment_type == "bullish":
                        score.bearish += weight
                    elif sentiment_type == "bearish":
                        score.bullish += weight
                    else:
                        score.neutral += weight
                else:
                    if sentiment_type == "bullish":
                        score.bullish += weight
                    elif sentiment_type == "bearish":
                        score.bearish += weight
                    else:
                        score.neutral += weight
        
        return score
    
    def _analyze_simple(self, text: str) -> SentimentScore:
        score = SentimentScore()
        text_lower = text.lower()
        tokens = re.findall(r'\b\w+\b', text_lower)
        
        for token in tokens:
            sentiment_info = self.inverted_index.get(token)
            
            if sentiment_info:
                sentiment_type, weight = sentiment_info
                
                pos = text_lower.find(token)
                is_negated = self._detect_negation_simple(text_lower, pos)
                
                if is_negated:
                    if sentiment_type == "bullish":
                        score.bearish += weight
                    elif sentiment_type == "bearish":
                        score.bullish += weight
                    else:
                        score.neutral += weight
                else:
                    if sentiment_type == "bullish":
                        score.bullish += weight
                    elif sentiment_type == "bearish":
                        score.bearish += weight
                    else:
                        score.neutral += weight
        
        return score
    
    def _extract_entities_from_article(self, article: NewsArticle) -> Dict[str, List[str]]:
        if hasattr(article, 'entities') and article.entities:
            return article.entities
        
        if self.extractor:
            return self.extractor.extract_entities(article)
        
        return {
            "Companies": [], "Sectors": [], "Regulators": [],
            "People": [], "Events": []
        }
    
    def _calculate_entity_weight(self, entities: Dict[str, List[str]]) -> float:
        weight = 1.0
        
        if companies := entities.get("Companies", []):
            weight *= (1.0 + (len(companies) * 0.1))
            weight *= self.entity_weights["company"]
        
        if entities.get("Sectors", []):
            weight *= self.entity_weights["sector"]
        
        if entities.get("Regulators", []):
            weight *= self.entity_weights["regulator"]
        
        return min(weight, 2.0)  # Cap weight at 2x
    
    def _calculate_event_modifier(self, entities: Dict[str, List[str]]) -> float:
        modifier = 1.0
        events = entities.get("Events", [])
        
        if not events:
            return modifier
        
        # Find highest applicable modifier
        event_modifiers = []
        for event in events:
            event_lower = event.lower()
            for key, mod in self.event_modifiers.items():
                if key in event_lower:
                    event_modifiers.append(mod)
        
        if event_modifiers:
            modifier = max(event_modifiers)
        
        return modifier
    
    def _calculate_keyword_density(self, text: str, score: SentimentScore) -> float:
        tokens = re.findall(r'\b\w+\b', text.lower())
        total_words = len(tokens)
        
        if total_words == 0:
            return 0.5
        
        # Normalize density to 0.5-1.5 range
        density = score.total / total_words
        
        if density < 0.01:
            return 0.5
        elif density > 0.05:
            return 1.5
        else:
            return 0.5 + (density / 0.05) * 1.0
    
    def _classify_sentiment(self, normalized_scores: Dict[str, float]) -> Tuple[str, float]:
        bullish = normalized_scores["bullish"]
        bearish = normalized_scores["bearish"]
        neutral = normalized_scores["neutral"]
        
        if bullish > bearish and bullish > neutral:
            classification = "Bullish"
            confidence = bullish
        elif bearish > bullish and bearish > neutral:
            classification = "Bearish"
            confidence = bearish
        else:
            classification = "Neutral"
            confidence = neutral
        
        # Adjust confidence based on margin of victory
        max_score = max(bullish, bearish, neutral)
        second_max = sorted([bullish, bearish, neutral])[-2]
        margin = max_score - second_max
        
        confidence_multiplier = 1.0 + (margin / 100.0)
        final_confidence = min(confidence * confidence_multiplier, 100.0)
        
        return classification, final_confidence
    
    def _calculate_signal_strength(
        self,
        confidence: float,
        entity_weight: float,
        event_modifier: float,
        density: float
    ) -> float:
        signal = confidence * entity_weight * event_modifier * density
        return round(min(signal, 100.0), 2)
    
    def analyze_sentiment(self, article: NewsArticle) -> Dict[str, Any]:
        # Weighted title 1.5x via repetition
        text = f"{article.title}. {article.title}. {article.content}".lower()
        
        if self.use_spacy and self.nlp:
            base_score = self._analyze_with_spacy(text)
        else:
            base_score = self._analyze_simple(text)
        
        entities = self._extract_entities_from_article(article)
        
        # Calculate contextual weights
        entity_weight = self._calculate_entity_weight(entities)
        event_modifier = self._calculate_event_modifier(entities)
        keyword_density = self._calculate_keyword_density(text, base_score)
        
        # Apply modifiers to base score
        weighted_score = SentimentScore(
            bullish=base_score.bullish * entity_weight * event_modifier,
            bearish=base_score.bearish * entity_weight * event_modifier,
            neutral=base_score.neutral
        )
        
        normalized_scores = weighted_score.normalize()
        classification, confidence_score = self._classify_sentiment(normalized_scores)
        
        signal_strength = self._calculate_signal_strength(
            confidence_score, entity_weight, event_modifier, keyword_density
        )
        
        return {
            "classification": classification,
            "confidence_score": round(confidence_score, 2),
            "signal_strength": signal_strength,
            "sentiment_breakdown": {
                "bullish": normalized_scores["bullish"],
                "bearish": normalized_scores["bearish"],
                "neutral": normalized_scores["neutral"],
                "entity_weight": round(entity_weight, 2),
                "event_modifier": round(event_modifier, 2),
                "keyword_density": round(keyword_density, 2)
            },
            "analysis_method": "rule_based"
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