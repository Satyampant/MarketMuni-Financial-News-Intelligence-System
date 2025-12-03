from app.core.config import Paths
"""
Rule-Based Sentiment Classifier for Financial News
Production-grade implementation with spaCy integration and optimized algorithms
"""

from typing import Dict, Any, Set, List, Tuple, Optional
import re
from dataclasses import dataclass
from pathlib import Path
from app.core.models import NewsArticle, SentimentData
from app.agents.entity_extraction import EntityExtractor

try:
    import spacy
    from spacy.tokens import Token, Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# MODULE_DIR replaced by Paths config


@dataclass
class SentimentScore:
    """Internal scoring structure for sentiment calculation"""
    bullish: float = 0.0
    bearish: float = 0.0
    neutral: float = 0.0
    
    @property
    def total(self) -> float:
        return self.bullish + self.bearish + self.neutral
    
    def normalize(self) -> Dict[str, float]:
        """Normalize scores to 0-100 scale"""
        if self.total == 0:
            return {"bullish": 33.33, "bearish": 33.33, "neutral": 33.34}
        
        return {
            "bullish": round((self.bullish / self.total) * 100, 2),
            "bearish": round((self.bearish / self.total) * 100, 2),
            "neutral": round((self.neutral / self.total) * 100, 2)
        }


class RuleBasedSentimentClassifier:
    
    # Financial lexicons based on Loughran-McDonald + domain knowledge
    # Strong bullish signals (weight: 3.0)
    STRONG_BULLISH = {
        "surge", "soar", "skyrocket", "boom", "breakthrough", "record",
        "outperform", "exceed", "beat", "stellar", "robust", "exceptional",
        "blockbuster", "rally", "upside", "gain", "achieve"
    }
    
    # Strong bearish signals (weight: 3.0)
    STRONG_BEARISH = {
        "plunge", "crash", "collapse", "bankruptcy", "fraud", "scandal",
        "default", "plummet", "nosedive", "catastrophic", "crisis",
        "downgrade", "miss", "fail", "violation", "penalty"
    }
    
    # Moderate bullish signals (weight: 2.0)
    MODERATE_BULLISH = {
        "growth", "profit", "gain", "expansion", "increase", "rise",
        "dividend", "buyback", "acquisition", "upgrade", "positive",
        "strong", "improve", "recover", "opportunity", "favorable",
        "strength", "advance", "progress", "success", "enhance",
        "accelerate", "outpace", "momentum", "rebound", "resilient", "windfall", 
        "margin expansion", "market share gain", "capacity expansion"
    }
    
    # Moderate bearish signals (weight: 2.0)
    MODERATE_BEARISH = {
        "decline", "loss", "fall", "drop", "layoff", "concern",
        "risk", "slowdown", "contraction", "cut", "reduce", "lower",
        "deteriorate", "struggle", "weak", "negative", "downside",
        "pressure", "challenge", "headwind", "uncertainty",
        "decelerate", "underperform", "headwinds", "writedown", "provision", 
        "margin compression", "market share loss", "capacity constraint"
    }
    
    # Weak bullish signals (weight: 1.0)
    WEAK_BULLISH = {
        "optimistic", "hopeful", "potential", "stable", "steady",
        "maintain", "hold", "sustain", "support", "modest"
    }
    
    # Weak bearish signals (weight: 1.0)
    WEAK_BEARISH = {
        "cautious", "uncertain", "volatile", "mixed", "soften",
        "ease", "temper", "moderate", "restrain", "limit"
    }
    
    # Neutral indicators (weight: 1.0)
    NEUTRAL_KEYWORDS = {
        "announce", "report", "schedule", "plan", "state",
        "say", "indicate", "reveal", "disclose", "confirm",
        "meeting", "conference", "presentation", "statement",
        "file", "submit", "publish", "release"
    }
    
    # Event type modifiers (multiply sentiment score)
    EVENT_MODIFIERS = {
        "dividend": 1.3,
        "buyback": 1.3,
        "stock buyback": 1.3,
        "merger": 1.1,
        "acquisition": 1.1,
        "ipo": 1.2,
        "earnings": 1.2,
        "profit": 1.2,
        "loss": 1.2,
        "layoff": 1.4,
        "bankruptcy": 1.5,
        "repo rate": 1.3,
        "interest rate": 1.3,
        "rates": 1.2,
        "policy rate": 1.3,
        "quarterly results": 1.2,
        "revenue": 1.1,
        "rights issue": 1.2,
        "delisting": 1.4,
        "share buyback": 1.3,
        "bonus issue": 1.2,
        "split": 1.1,
        "consolidation": 1.1,
        "restructuring": 1.2,
        "divestment": 1.1,
        "capital raising": 1.2
    }
    
    # Entity type weights
    ENTITY_WEIGHTS = {
        "company": 1.3,
        "sector": 1.1,
        "regulator": 1.2
    }
    
    def __init__(self, entity_extractor: EntityExtractor = None, use_spacy: bool = True, model_name: str = "en_core_web_sm"):
        self.extractor = entity_extractor or EntityExtractor()
        self.nlp = None
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        # Load spaCy model
        if self.use_spacy:
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    self.nlp = spacy.blank("en")
                    self.use_spacy = False
        
        # Build inverted index for O(n) lookup
        self._build_inverted_index()
    
    def _build_inverted_index(self):
        self.inverted_index = {}
        
        # Map each keyword to its sentiment and weight
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
                # Store both original and lemmatized forms
                self.inverted_index[keyword.lower()] = (sentiment, weight)
                
                # Also store common variations manually
                if keyword.endswith('e'):
                    # surge -> surging, surged
                    self.inverted_index[keyword[:-1] + 'ing'] = (sentiment, weight)
                    self.inverted_index[keyword + 'd'] = (sentiment, weight)
    
    def _detect_negation_spacy(self, token: Token) -> bool:
        # Check if token has negation dependency
        if token.dep_ == "neg":
            return True
        
        # Check if any child has negation
        for child in token.children:
            if child.dep_ == "neg":
                return True
        
        # Check if parent is negated
        if token.head.dep_ == "neg":
            return True
        
        # Check for common negation patterns in ancestors
        for ancestor in token.ancestors:
            if ancestor.lemma_ in {"not", "no", "never", "neither", "nor", "none"}:
                # Check if token is in scope of negation
                if token.i > ancestor.i and token.i - ancestor.i <= 4:
                    return True
        
        return False
    
    def _detect_negation_simple(self, text: str, pos: int) -> bool:
        # Look at previous 20 characters (roughly 3-4 words)
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
            # Get lemmatized form
            lemma = token.lemma_.lower()
            text_lower = token.text.lower()
            
            # Check inverted index for both lemma and original text
            sentiment_info = self.inverted_index.get(lemma) or self.inverted_index.get(text_lower)
            
            if sentiment_info:
                sentiment_type, weight = sentiment_info
                
                # Detect negation using dependency parsing
                is_negated = self._detect_negation_spacy(token)
                
                # Apply sentiment with negation handling
                if is_negated:
                    # Flip sentiment
                    if sentiment_type == "bullish":
                        score.bearish += weight
                    elif sentiment_type == "bearish":
                        score.bullish += weight
                    # Neutral stays neutral
                    else:
                        score.neutral += weight
                else:
                    # Normal sentiment
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
        
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', text_lower)
        
        for i, token in enumerate(tokens):
            sentiment_info = self.inverted_index.get(token)
            
            if sentiment_info:
                sentiment_type, weight = sentiment_info
                
                # Find position in original text for negation check
                pos = text_lower.find(token)
                is_negated = self._detect_negation_simple(text_lower, pos)
                
                # Apply sentiment with negation handling
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
        # Check if entities already attached to article
        if hasattr(article, 'entities') and article.entities:
            return article.entities
        
        if self.extractor:
            return self.extractor.extract_entities(article)
        # Return empty entities if extraction not available
        return {
            "Companies": [],
            "Sectors": [],
            "Regulators": [],
            "People": [],
            "Events": []
        }
    
    def _calculate_entity_weight(self, entities: Dict[str, List[str]]) -> float:
        weight = 1.0
        
        # Company weight
        companies = entities.get("Companies", [])
        if companies:
            weight *= (1.0 + (len(companies) * 0.1))
            weight *= self.ENTITY_WEIGHTS["company"]
        
        # Sector weight
        sectors = entities.get("Sectors", [])
        if sectors:
            weight *= self.ENTITY_WEIGHTS["sector"]
        
        # Regulator weight
        regulators = entities.get("Regulators", [])
        if regulators:
            weight *= self.ENTITY_WEIGHTS["regulator"]
        
        return min(weight, 2.0)  # Cap at 2x
    
    def _calculate_event_modifier(self, entities: Dict[str, List[str]]) -> float:
        modifier = 1.0
        events = entities.get("Events", [])
        
        if not events:
            return modifier
        
        # Apply highest event modifier
        event_modifiers = []
        for event in events:
            event_lower = event.lower()
            for key, mod in self.EVENT_MODIFIERS.items():
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
        
        total_keywords = score.total
        density = total_keywords / total_words
        
        # Normalize to 0.5-1.5 range
        if density < 0.01:
            return 0.5
        elif density > 0.05:
            return 1.5
        else:
            return 0.5 + (density / 0.05) * 1.0
    
    def _classify_sentiment(
        self, 
        normalized_scores: Dict[str, float]
    ) -> Tuple[str, float]:
        bullish = normalized_scores["bullish"]
        bearish = normalized_scores["bearish"]
        neutral = normalized_scores["neutral"]
        
        # Determine classification
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
        
        # Higher margin = higher confidence
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
        # Combine title and content (title weighted 1.5x)
        text = f"{article.title}. {article.title}. {article.content}"
        text = text.lower()
        
        # Step 1: Calculate base sentiment scores using optimal method
        if self.use_spacy and self.nlp:
            base_score = self._analyze_with_spacy(text)
        else:
            base_score = self._analyze_simple(text)
        
        # Step 2: Extract or retrieve entities
        entities = self._extract_entities_from_article(article)
        
        # Step 3: Calculate contextual weights
        entity_weight = self._calculate_entity_weight(entities)
        event_modifier = self._calculate_event_modifier(entities)
        keyword_density = self._calculate_keyword_density(text, base_score)
        
        # Step 4: Apply weights to scores
        weighted_score = SentimentScore(
            bullish=base_score.bullish * entity_weight * event_modifier,
            bearish=base_score.bearish * entity_weight * event_modifier,
            neutral=base_score.neutral
        )
        
        # Step 5: Normalize scores to 0-100 scale
        normalized_scores = weighted_score.normalize()
        
        # Step 6: Classify sentiment and calculate confidence
        classification, confidence_score = self._classify_sentiment(normalized_scores)
        
        # Step 7: Calculate signal strength
        signal_strength = self._calculate_signal_strength(
            confidence_score,
            entity_weight,
            event_modifier,
            keyword_density
        )
        
        # Step 8: Build detailed breakdown
        sentiment_breakdown = {
            "bullish": normalized_scores["bullish"],
            "bearish": normalized_scores["bearish"],
            "neutral": normalized_scores["neutral"],
            "entity_weight": round(entity_weight, 2),
            "event_modifier": round(event_modifier, 2),
            "keyword_density": round(keyword_density, 2)
        }
        
        return {
            "classification": classification,
            "confidence_score": round(confidence_score, 2),
            "signal_strength": signal_strength,
            "sentiment_breakdown": sentiment_breakdown,
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

