import re
from typing import Dict, List, Optional, Any
from dataclasses import asdict

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from app.core.models import NewsArticle
from app.core.llm_schemas import EntityExtractionSchema
from app.services.llm_client import GroqLLMClient, LLMServiceError
from app.services.redis_cache import RedisCacheService, get_redis_cache
from app.core.config_loader import get_config

class LLMEntityExtractor:
    def __init__(
        self,
        llm_client: Optional[GroqLLMClient] = None,
        redis_cache: Optional[RedisCacheService] = None,
        reference_companies: Optional[List[str]] = None,
        reference_sectors: Optional[List[str]] = None,
        reference_regulators: Optional[List[str]] = None,
        enable_caching: bool = True
    ):
        self.config = get_config()
        
        if llm_client is None:
            self.llm_client = GroqLLMClient(
                model=self.config.llm.models.fast,
                temperature=0.1,
                max_tokens=2048
            )
        else:
            self.llm_client = llm_client
        
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
        
        # Load reference lists for validation/normalization
        self.reference_companies = reference_companies or self._load_default_companies()
        self.reference_sectors = reference_sectors or self._load_default_sectors()
        self.reference_regulators = reference_regulators or self._load_default_regulators()
        
        print(f"✓ LLMEntityExtractor initialized")
        print(f"  - Reference companies: {len(self.reference_companies)}")
        print(f"  - Reference sectors: {len(self.reference_sectors)}")
        print(f"  - Reference regulators: {len(self.reference_regulators)}")
    
    def _load_default_companies(self) -> List[str]:
        return [
            "HDFC Bank", "ICICI Bank", "State Bank of India", "Reliance Industries",
            "TCS", "Infosys", "Wipro", "Adani Green Energy", "Mahindra & Mahindra",
            "Bharti Airtel", "Sun Pharmaceutical", "Maruti Suzuki", "Power Grid",
            "UltraTech Cement", "Tata Motors", "Tech Mahindra", "HUL", "ITC",
            "Asian Paints", "Bajaj Finance", "Axis Bank", "Kotak Mahindra Bank",
            "NTPC", "Coal India", "Larsen & Toubro", "HCL Technologies"
        ]
    
    def _load_default_sectors(self) -> List[str]:
        return [
            "Banking", "Finance", "NBFC", "Insurance", "IT", "Energy", "Oil & Gas",
            "Auto", "Telecom", "Pharma", "Healthcare", "Cement", "Utilities",
            "Steel", "Semiconductors", "Rubber", "Logistics", "Mining", "Coal",
            "Construction", "Infrastructure", "Real Estate", "Agriculture", "FMCG",
            "Textiles", "Aviation", "Chemicals", "Retail", "Consumer Durables",
            "Media", "E-commerce", "Railways", "Defense"
        ]
    
    def _load_default_regulators(self) -> List[str]:
        return [
            "RBI", "SEBI", "IRDAI", "CCI", "TRAI", "DOT", "DGCA", "RERA", "FSSAI",
            "US FDA", "Federal Reserve", "Ministry of Finance", "Ministry of Commerce"
        ]
    
    def _build_extraction_prompt(self, article: NewsArticle) -> str:
        template = self.config.prompts.entity_extraction.task_prompt
        return template.format(
            title=article.title,
            content=article.content
        )
    
    def _validate_ticker_symbol(self, ticker: str) -> bool:
        if not ticker or len(ticker) < 1:
            return False
        # Ensures uppercase alphanumeric, usually 1-12 chars
        pattern = r'^(?=.*[A-Z])[A-Z0-9]{1,12}$'
        return bool(re.match(pattern, ticker))
    
    def _normalize_company_name(self, name: str) -> str:
        if not name:
            return name
        
        normalized = name.strip()
        for prefix in ["The ", "the ", "THE "]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        # Fuzzy match against reference list if available
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
        """Post-process: normalize names, deduplicate, and validate tickers."""
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
        
        # Check Redis Cache
        if use_cache and self.enable_caching and self.redis_cache and self.redis_cache.is_connected:
            cached_result = self.redis_cache.get(article.id)
            if cached_result:
                try:
                    return EntityExtractionSchema.model_validate(cached_result)
                except Exception as e:
                    print(f"⚠ Cache deserialization error for {article.id}: {e}")
                    # Fall through to fresh extraction
        
        # Perform Extraction
        system_message = self.config.prompts.entity_extraction.system_message
        prompt = self._build_extraction_prompt(article)
        
        try:
            result_dict = self.llm_client.generate_structured_output(
                prompt=prompt,
                schema=EntityExtractionSchema,
                system_message=system_message
            )
            
            raw_result = EntityExtractionSchema.model_validate(result_dict)
            validated_result = self._post_process_entities(raw_result)
            
            if self.enable_caching and self.redis_cache and self.redis_cache.is_connected:
                cache_data = validated_result.model_dump()
                self.redis_cache.set(article.id, cache_data)
            
            return validated_result
            
        except Exception as e:
            raise LLMServiceError(f"Entity extraction failed for {article.id}: {e}")
    
    def clear_cache(self, article_id: Optional[str] = None) -> int:
        if not self.enable_caching or not self.redis_cache or not self.redis_cache.is_connected:
            return 0
        
        if article_id:
            deleted = self.redis_cache.delete(article_id)
            return 1 if deleted else 0
        else:
            return self.redis_cache.clear_all()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        if not self.enable_caching or not self.redis_cache:
            return {"cache_enabled": False, "cache_type": "none"}
        
        if not self.redis_cache.is_connected:
            return {"cache_enabled": True, "cache_type": "redis", "connected": False}
        
        redis_stats = self.redis_cache.get_stats()
        return {
            "cache_enabled": True,
            "cache_type": "redis",
            **redis_stats
        }