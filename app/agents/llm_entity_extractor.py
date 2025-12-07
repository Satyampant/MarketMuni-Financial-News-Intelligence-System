"""
LLM-Based Entity Extraction Agent with Redis Caching
Shared cache across multiple workers with persistence.
"""

from typing import Dict, List, Optional, Any
from dataclasses import asdict
import re

from app.core.models import NewsArticle
from app.core.llm_schemas import EntityExtractionSchema
from app.services.llm_client import GroqLLMClient, LLMServiceError
from app.services.redis_cache import RedisCacheService, get_redis_cache
from app.core.config_loader import get_config

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


class LLMEntityExtractor:
    """
    LLM-powered entity extraction with Redis caching.
    Cache is shared across workers and persists across restarts.
    """
    
    def __init__(
        self,
        llm_client: Optional[GroqLLMClient] = None,
        redis_cache: Optional[RedisCacheService] = None,
        reference_companies: Optional[List[str]] = None,
        reference_sectors: Optional[List[str]] = None,
        reference_regulators: Optional[List[str]] = None,
        enable_caching: bool = True
    ):
        """
        Initialize LLM entity extractor with Redis caching.
        
        Args:
            llm_client: Optional pre-configured LLM client
            redis_cache: Optional Redis cache service (uses singleton if None)
            reference_companies: List of known companies for validation
            reference_sectors: List of known sectors for validation
            reference_regulators: List of known regulators for validation
            enable_caching: Enable Redis caching
        """
        config = get_config()
        
        # Initialize LLM client
        if llm_client is None:
            self.llm_client = GroqLLMClient(
                model=config.llm.models.fast,
                temperature=0.1,
                max_tokens=2048
            )
        else:
            self.llm_client = llm_client
        
        # Initialize Redis cache
        self.enable_caching = enable_caching
        if enable_caching:
            self.redis_cache = redis_cache or get_redis_cache()
            if self.redis_cache.is_connected:
                print("✓ Redis caching enabled for entity extraction")
            else:
                print("⚠ Redis not available - caching disabled")
        else:
            self.redis_cache = None
            print("✓ Caching disabled by configuration")
        
        # Reference lists for validation (dependency injection)
        self.reference_companies = reference_companies or self._load_default_companies()
        self.reference_sectors = reference_sectors or self._load_default_sectors()
        self.reference_regulators = reference_regulators or self._load_default_regulators()
        
        print(f"✓ LLMEntityExtractor initialized")
        print(f"  - Reference companies: {len(self.reference_companies)}")
        print(f"  - Reference sectors: {len(self.reference_sectors)}")
        print(f"  - Reference regulators: {len(self.reference_regulators)}")
    
    def _load_default_companies(self) -> List[str]:
        """Load minimal reference list of top companies."""
        return [
            "HDFC Bank", "ICICI Bank", "State Bank of India", "Reliance Industries",
            "TCS", "Infosys", "Wipro", "Adani Green Energy", "Mahindra & Mahindra",
            "Bharti Airtel", "Sun Pharmaceutical", "Maruti Suzuki", "Power Grid",
            "UltraTech Cement", "Tata Motors", "Tech Mahindra", "HUL", "ITC",
            "Asian Paints", "Bajaj Finance", "Axis Bank", "Kotak Mahindra Bank",
            "NTPC", "Coal India", "Larsen & Toubro", "HCL Technologies"
        ]
    
    def _load_default_sectors(self) -> List[str]:
        """Load known sectors for validation."""
        return [
            "Banking", "Finance", "NBFC", "Insurance", "IT", "Energy", "Oil & Gas",
            "Auto", "Telecom", "Pharma", "Healthcare", "Cement", "Utilities",
            "Steel", "Semiconductors", "Rubber", "Logistics", "Mining", "Coal",
            "Construction", "Infrastructure", "Real Estate", "Agriculture", "FMCG",
            "Textiles", "Aviation", "Chemicals", "Retail", "Consumer Durables",
            "Media", "E-commerce", "Railways", "Defense"
        ]
    
    def _load_default_regulators(self) -> List[str]:
        """Load known regulators for validation."""
        return [
            "RBI", "SEBI", "IRDAI", "CCI", "TRAI", "DOT", "DGCA", "RERA", "FSSAI",
            "US FDA", "Federal Reserve", "Ministry of Finance", "Ministry of Commerce"
        ]
    
    def _build_extraction_prompt(self, article: NewsArticle) -> str:
        """Construct detailed prompt for entity extraction."""
        return f"""Extract financial entities from this news article with high precision:

**Title**: {article.title}

**Content**: {article.content}

**Task**: Extract the following entities:

1. **Companies**:
   - Full company name as mentioned
   - Stock ticker symbol (e.g., HDFCBANK for NSE, RELIANCE for BSE, AAPL for NASDAQ)
   - Industry sector (e.g., Banking, IT, Auto)
   - Confidence score (0.0-1.0) - how certain you are this is a company

2. **Sectors**:
   - Broad industry categories mentioned or implied
   - Examples: Banking, IT, Pharma, Auto, Energy, FMCG, Telecom

3. **Regulators**:
   - Regulatory bodies or government entities
   - Include full name, acronym, and jurisdiction
   - Examples: RBI (India), SEBI (India), US FDA (United States)
   - Confidence score (0.0-1.0)

4. **People**:
   - Key individuals mentioned (CEOs, ministers, analysts)
   - Full names only (avoid single names)

5. **Events**:
   - Market events or corporate actions
   - Event type (e.g., dividend, merger, rate_hike, earnings, ipo)
   - Brief description
   - Confidence score (0.0-1.0)

**Guidelines**:
- Ticker symbols: Use correct exchange format (NSE/BSE for Indian stocks, NYSE/NASDAQ for US)
- Sectors: Use standard industry classifications
- Confidence: 1.0 = explicit mention, 0.7-0.9 = strong inference, 0.5-0.6 = weak inference
- Events: Normalize to lowercase with underscores (e.g., "stock buyback" → "stock_buyback")

**Output**: Return structured JSON matching the EntityExtractionSchema format."""
    
    def _validate_ticker_symbol(self, ticker: str) -> bool:
        """Validate ticker symbol format."""
        if not ticker or len(ticker) < 1:
            return False
        pattern = r'^[A-Z]{1,10}$'
        return bool(re.match(pattern, ticker.upper()))
    
    def _normalize_company_name(self, name: str) -> str:
        """Normalize company name using fuzzy matching."""
        if not name:
            return name
        
        normalized = name.strip()
        for prefix in ["The ", "the ", "THE "]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        if RAPIDFUZZ_AVAILABLE and self.reference_companies:
            match = process.extractOne(
                normalized,
                self.reference_companies,
                scorer=fuzz.ratio,
                score_cutoff=85
            )
            if match:
                return match[0]
        
        return normalized
    
    def _normalize_sector(self, sector: str) -> Optional[str]:
        """Normalize sector name against reference list."""
        if not sector:
            return None
        
        for ref_sector in self.reference_sectors:
            if sector.lower() == ref_sector.lower():
                return ref_sector
        
        if RAPIDFUZZ_AVAILABLE:
            match = process.extractOne(
                sector,
                self.reference_sectors,
                scorer=fuzz.ratio,
                score_cutoff=80
            )
            if match:
                return match[0]
        
        return sector
    
    def _normalize_regulator(self, regulator_name: str) -> str:
        """Normalize regulator name against reference list."""
        if not regulator_name:
            return regulator_name
        
        if regulator_name in self.reference_regulators:
            return regulator_name
        
        if RAPIDFUZZ_AVAILABLE:
            match = process.extractOne(
                regulator_name,
                self.reference_regulators,
                scorer=fuzz.ratio,
                score_cutoff=85
            )
            if match:
                return match[0]
        
        return regulator_name
    
    def _post_process_entities(self, raw_result: EntityExtractionSchema) -> EntityExtractionSchema:
        """Post-process and validate LLM extracted entities."""
        validated_companies = []
        seen_companies = set()
        
        for company in raw_result.companies:
            normalized_name = self._normalize_company_name(company.name)
            
            if normalized_name.lower() in seen_companies:
                continue
            seen_companies.add(normalized_name.lower())
            
            if company.ticker_symbol:
                if self._validate_ticker_symbol(company.ticker_symbol):
                    company.ticker_symbol = company.ticker_symbol.upper()
                else:
                    company.ticker_symbol = None
            
            if company.sector:
                company.sector = self._normalize_sector(company.sector)
            
            company.name = normalized_name
            validated_companies.append(company)
        
        validated_sectors = []
        seen_sectors = set()
        
        for sector in raw_result.sectors:
            normalized_sector = self._normalize_sector(sector)
            if normalized_sector and normalized_sector.lower() not in seen_sectors:
                validated_sectors.append(normalized_sector)
                seen_sectors.add(normalized_sector.lower())
        
        validated_regulators = []
        seen_regulators = set()
        
        for regulator in raw_result.regulators:
            normalized_name = self._normalize_regulator(regulator.name)
            
            if normalized_name.lower() in seen_regulators:
                continue
            seen_regulators.add(normalized_name.lower())
            
            regulator.name = normalized_name
            validated_regulators.append(regulator)
        
        validated_people = list(set(raw_result.people))
        
        validated_events = []
        seen_events = set()
        
        for event in raw_result.events:
            if event.event_type.lower() not in seen_events:
                validated_events.append(event)
                seen_events.add(event.event_type.lower())
        
        raw_result.companies = validated_companies
        raw_result.sectors = validated_sectors
        raw_result.regulators = validated_regulators
        raw_result.people = validated_people
        raw_result.events = validated_events
        
        return raw_result
    
    def extract_entities(
        self,
        article: NewsArticle,
        use_cache: bool = True
    ) -> EntityExtractionSchema:
        """
        Extract entities from article using LLM with Redis caching.
        
        Args:
            article: News article to process
            use_cache: Whether to use Redis cache
            
        Returns:
            EntityExtractionSchema with companies, sectors, regulators, people, events
        """
        # Try Redis cache first
        if use_cache and self.enable_caching and self.redis_cache and self.redis_cache.is_connected:
            cached_result = self.redis_cache.get(article.id)
            if cached_result:
                # Reconstruct EntityExtractionSchema from cached dict
                try:
                    return EntityExtractionSchema.model_validate(cached_result)
                except Exception as e:
                    print(f"⚠ Cache deserialization error for {article.id}: {e}")
                    # Fall through to fresh extraction
        
        # LLM extraction
        system_message = """You are a financial entity extraction expert specializing in Indian and global markets.
Extract companies, sectors, regulators, people, and market events with high precision.
Always provide confidence scores and ticker symbols when identifiable.
Return results in strict JSON format matching the EntityExtractionSchema."""
        
        prompt = self._build_extraction_prompt(article)
        
        try:
            # Call LLM with structured output
            result_dict = self.llm_client.generate_structured_output(
                prompt=prompt,
                schema=EntityExtractionSchema,
                system_message=system_message
            )
            
            # Validate with Pydantic
            raw_result = EntityExtractionSchema.model_validate(result_dict)
            
            # Post-process and validate
            validated_result = self._post_process_entities(raw_result)
            
            # Cache result in Redis
            if self.enable_caching and self.redis_cache and self.redis_cache.is_connected:
                cache_data = validated_result.model_dump()
                self.redis_cache.set(article.id, cache_data)
            
            return validated_result
            
        except Exception as e:
            raise LLMServiceError(f"Entity extraction failed for {article.id}: {e}")
    
    def clear_cache(self, article_id: Optional[str] = None) -> int:
        """
        Clear Redis cache.
        
        Args:
            article_id: Specific article to clear (None = clear all)
            
        Returns:
            Number of entries cleared
        """
        if not self.enable_caching or not self.redis_cache or not self.redis_cache.is_connected:
            return 0
        
        if article_id:
            deleted = self.redis_cache.delete(article_id)
            return 1 if deleted else 0
        else:
            return self.redis_cache.clear_all()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from Redis."""
        if not self.enable_caching or not self.redis_cache:
            return {
                "cache_enabled": False,
                "cache_type": "none"
            }
        
        if not self.redis_cache.is_connected:
            return {
                "cache_enabled": True,
                "cache_type": "redis",
                "connected": False
            }
        
        redis_stats = self.redis_cache.get_stats()
        return {
            "cache_enabled": True,
            "cache_type": "redis",
            **redis_stats
        }