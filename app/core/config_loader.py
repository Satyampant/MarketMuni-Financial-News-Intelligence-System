"""
Configuration Loader for MarketMuni
Loads and validates configuration from YAML file.
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

# Resolve project root relative to this file
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = ROOT_DIR / "config.yaml"
PROMPTS_FILE = ROOT_DIR / "prompts.yaml"

@dataclass
class EntityExtractionPrompts:
    system_message: str = ""
    task_prompt: str = ""
    entity_context_format: str = ""

@dataclass
class SentimentAnalysisPrompts:
    system_message: str = ""
    task_prompt: str = ""
    few_shot_examples: str = ""

@dataclass
class StockMappingPrompts:
    system_message: str = ""
    task_prompt: str = ""

@dataclass
class SupplyChainPrompts:
    system_message: str = ""
    few_shot_examples: str = ""
    task_prompt: str = ""

@dataclass
class QueryRoutingPrompts:
    system_message: str = ""
    few_shot_examples: str = ""
    task_prompt: str = ""

@dataclass
class PromptConfig:
    """Prompt templates configuration."""
    entity_extraction: EntityExtractionPrompts = field(default_factory=dict)
    sentiment_analysis: SentimentAnalysisPrompts = field(default_factory=dict)
    stock_impact: StockMappingPrompts = field(default_factory=dict)
    supply_chain: SupplyChainPrompts = field(default_factory=dict)
    query_routing: QueryRoutingPrompts = field(default_factory=dict)

@dataclass
class MongoDBConfig:
    """MongoDB configuration for article storage."""
    connection_string: str = "mongodb://localhost:27017/"
    database_name: str = "marketmuni"
    collection_name: str = "articles"
    max_pool_size: int = 100
    timeout_ms: int = 5000
    max_filter_ids: int = 1000  # Threshold for broad filter optimization

@dataclass
class RedisConfig:
    """Redis cache configuration."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ttl_seconds: int = 86400  # 24 hours
    max_connections: int = 50
    socket_timeout: int = 5
    socket_connect_timeout: int = 5

@dataclass
class DeduplicationConfig:
    bi_encoder_threshold: float = 0.50
    cross_encoder_threshold: float = 0.70
    cross_encoder_model: str = "cross-encoder/stsb-distilroberta-base"

@dataclass
class EntityExtractionConfig:
    spacy_model: str = "en_core_web_sm"
    use_spacy: bool = True
    event_keywords: list = field(default_factory=list)

@dataclass
class VectorStoreConfig:
    collection_name: str = "financial_news"
    persist_directory: str = "data/chroma_db"
    embedding_model: str = "all-mpnet-base-v2"
    distance_metric: str = "cosine"

@dataclass
class LLMRoutingConfig:
    """LLM-based query routing configuration."""
    enabled: bool = True
    confidence_threshold: float = 0.6
    fallback_strategy: str = "semantic_search"
    max_entities_per_query: int = 10
    enable_query_expansion: bool = True

@dataclass
class MultiQueryConfig:
    max_context_queries: int = 3
    initial_retrieval_multiplier: int = 2

@dataclass
class RerankingWeights:
    strategy_weight: float = 0.5
    semantic_weight: float = 0.5

@dataclass
class SentimentBoostConfig:
    enabled: bool = True
    max_multiplier: float = 1.5

@dataclass
class QueryProcessingConfig:
    default_top_k: int = 10
    min_similarity: float = 0.3
    llm_routing: LLMRoutingConfig = field(default_factory=LLMRoutingConfig)
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    multi_query: MultiQueryConfig = field(default_factory=MultiQueryConfig)
    reranking_weights: Dict[str, RerankingWeights] = field(default_factory=dict)
    sentiment_boost: SentimentBoostConfig = field(default_factory=SentimentBoostConfig)

@dataclass
class StockImpactConfig:
    confidence_thresholds: Dict[str, float] = field(default_factory=dict)
    fuzzy_match_threshold: float = 0.80

@dataclass
class SupplyChainConfig:
    traversal_depth: int = 1
    min_impact_score: float = 25.0
    weight_decay: float = 0.8

@dataclass
class LLMModelsConfig:
    """Alternative models for specific tasks."""
    fast: str = "llama-3.1-8b-instant"
    reasoning: str = "llama-3.3-70b-versatile"
    structured: str = "llama-3.3-70b-versatile"

@dataclass
class LLMFeaturesConfig:
    """Feature flags for LLM-based components."""
    entity_extraction: bool = True
    stock_mapping: bool = True
    sentiment_analysis: bool = True
    supply_chain: bool = True
    query_expansion: bool = True

@dataclass
class LLMConfig:
    """LLM configuration for Groq integration."""
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 30
    max_retries: int = 3
    models: LLMModelsConfig = field(default_factory=LLMModelsConfig)
    features: LLMFeaturesConfig = field(default_factory=LLMFeaturesConfig)

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    title: str = "Financial News Intelligence API"
    description: str = "Multi-agent AI system for processing and querying financial news"
    version: str = "1.0.0"

@dataclass
class ResourcesConfig:
    company_aliases: str = "company_aliases.json"
    sector_tickers: str = "sector_tickers.json"
    regulators: str = "regulators.json"
    regulator_impact: str = "regulator_sector_impact.json"
    supply_chain_graph: str = "supply_chain_graph.json"

@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

@dataclass
class PerformanceConfig:
    cache_embeddings: bool = True
    batch_size: int = 32
    num_workers: int = 4

@dataclass
class DevelopmentConfig:
    debug: bool = False
    use_mock_data: bool = False
    mock_data_path: str = "mock_news_data.json"
    enable_profiling: bool = False

@dataclass
class Config:
    """Main configuration class containing all settings."""
    mongodb: MongoDBConfig = field(default_factory=MongoDBConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    entity_extraction: EntityExtractionConfig = field(default_factory=EntityExtractionConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    query_processing: QueryProcessingConfig = field(default_factory=QueryProcessingConfig)
    stock_impact: StockImpactConfig = field(default_factory=StockImpactConfig)
    supply_chain: SupplyChainConfig = field(default_factory=SupplyChainConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)  
    redis: RedisConfig = field(default_factory=RedisConfig)
    api: APIConfig = field(default_factory=APIConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file, falling back to defaults if missing."""
    if config_path is None:
        config_path = CONFIG_FILE
    
    try:
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        config = Config()
        
        # --- MongoDB Configuration ---
        if 'mongodb' in yaml_data:
            mongo = yaml_data['mongodb']
            config.mongodb = MongoDBConfig(
                connection_string=mongo.get('connection_string', 'mongodb://localhost:27017/'),
                database_name=mongo.get('database_name', 'marketmuni'),
                collection_name=mongo.get('collection_name', 'articles'),
                max_pool_size=mongo.get('max_pool_size', 100),
                timeout_ms=mongo.get('timeout_ms', 5000),
                max_filter_ids=mongo.get('max_filter_ids', 1000)
            )
        
        # --- Deduplication ---
        dedup = yaml_data['deduplication']
        config.deduplication = DeduplicationConfig(
            bi_encoder_threshold=dedup.get('bi_encoder_threshold', 0.50),
            cross_encoder_threshold=dedup.get('cross_encoder_threshold', 0.70),
            cross_encoder_model=dedup.get('cross_encoder_model', 'cross-encoder/stsb-distilroberta-base')
        )
        
        # --- Entity Extraction ---
        entity = yaml_data['entity_extraction']
        config.entity_extraction = EntityExtractionConfig(
            spacy_model=entity.get('spacy_model', 'en_core_web_sm'),
            use_spacy=entity.get('use_spacy', True),
            event_keywords=entity.get('event_keywords', [])
        )
        
        # --- Vector Store ---
        vs = yaml_data['vector_store']
        config.vector_store = VectorStoreConfig(
            collection_name=vs.get('collection_name', 'financial_news'),
            persist_directory=vs.get('persist_directory', 'data/chroma_db'),
            embedding_model=vs.get('embedding_model', 'all-mpnet-base-v2'),
            distance_metric=vs.get('distance_metric', 'cosine')
        )
        
        # --- Query Processing  ---
        qp = yaml_data['query_processing']
        
        # LLM Routing Config
        llm_routing = LLMRoutingConfig()
        lr = qp['llm_routing']
        llm_routing = LLMRoutingConfig(
            enabled=lr.get('enabled', True),
            confidence_threshold=lr.get('confidence_threshold', 0.6),
            fallback_strategy=lr.get('fallback_strategy', 'semantic_search'),
            max_entities_per_query=lr.get('max_entities_per_query', 10),
            enable_query_expansion=lr.get('enable_query_expansion', True)
        )
        
        # Strategy Weights
        strategy_weights = qp.get('strategy_weights', {})
        
        multi_query = MultiQueryConfig()
        mq = qp['multi_query']
        multi_query = MultiQueryConfig(
            max_context_queries=mq.get('max_context_queries', 3),
            initial_retrieval_multiplier=mq.get('initial_retrieval_multiplier', 2)
        )
        
        reranking_weights = {}
        for strategy, weights in qp['reranking_weights'].items():
            reranking_weights[strategy] = RerankingWeights(
                strategy_weight=weights.get('strategy_weight', 0.5),
                semantic_weight=weights.get('semantic_weight', 0.5)
            )
        
        sentiment_boost = SentimentBoostConfig()
        sb = qp['sentiment_boost']
        sentiment_boost = SentimentBoostConfig(
            enabled=sb.get('enabled', True),
            max_multiplier=sb.get('max_multiplier', 1.5)
        )
        
        config.query_processing = QueryProcessingConfig(
            default_top_k=qp.get('default_top_k', 10),
            min_similarity=qp.get('min_similarity', 0.3),
            llm_routing=llm_routing,
            strategy_weights=strategy_weights,
            multi_query=multi_query,
            reranking_weights=reranking_weights,
            sentiment_boost=sentiment_boost
        )
        
        # --- Stock Impact ---
        si = yaml_data['stock_impact']
        config.stock_impact = StockImpactConfig(
            confidence_thresholds=si.get('confidence_thresholds', {}),
            fuzzy_match_threshold=si.get('fuzzy_match_threshold', 0.80)
        )
        
        # --- Supply Chain ---
        sc = yaml_data['supply_chain']
        config.supply_chain = SupplyChainConfig(
            traversal_depth=sc.get('traversal_depth', 1),
            min_impact_score=sc.get('min_impact_score', 25.0),
            weight_decay=sc.get('weight_decay', 0.8)
        )
        
        # --- LLM Configuration ---
        llm = yaml_data['llm']
        
        models_config = LLMModelsConfig()
        m = llm['models']
        models_config = LLMModelsConfig(
            fast=m.get('fast', 'llama-3.1-8b-instant'),
            reasoning=m.get('reasoning', 'llama-3.3-70b-versatile'),
            structured=m.get('structured', 'llama-3.3-70b-versatile')
        )
        
        features_config = LLMFeaturesConfig()
        f = llm['features']
        features_config = LLMFeaturesConfig(
            entity_extraction=f.get('entity_extraction', True),
            stock_mapping=f.get('stock_mapping', True),
            sentiment_analysis=f.get('sentiment_analysis', True),
            supply_chain=f.get('supply_chain', True),
            query_expansion=f.get('query_expansion', True)
        )
        
        config.llm = LLMConfig(
            provider=llm.get('provider', 'groq'),
            model=llm.get('model', 'llama-3.3-70b-versatile'),
            temperature=llm.get('temperature', 0.1),
            max_tokens=llm.get('max_tokens', 4096),
            timeout=llm.get('timeout', 30),
            max_retries=llm.get('max_retries', 3),
            models=models_config,
            features=features_config
        )

        # --- Redis Configuration ---
        redis_config = yaml_data['redis']
        
        # Support environment variable override for password
        redis_password = redis_config.get('password')
        if redis_password is None:
            redis_password = os.getenv('REDIS_PASSWORD')
        
        config.redis = RedisConfig(
            enabled=redis_config.get('enabled', True),
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            password=redis_password,
            ttl_seconds=redis_config.get('ttl_seconds', 86400),
            max_connections=redis_config.get('connection_pool', {}).get('max_connections', 50),
            socket_timeout=redis_config.get('connection_pool', {}).get('socket_timeout', 5),
            socket_connect_timeout=redis_config.get('connection_pool', {}).get('socket_connect_timeout', 5)
        )
        
        # --- API ---
        api = yaml_data['api']
        config.api = APIConfig(
            host=api.get('host', '0.0.0.0'),
            port=api.get('port', 8000),
            reload=api.get('reload', True),
            title=api.get('title', 'Financial News Intelligence API'),
            description=api.get('description', 'Multi-agent AI system'),
            version=api.get('version', '1.0.0')
        )
        
        # --- Logging ---
        log = yaml_data['logging']
        config.logging = LoggingConfig(
            level=log.get('level', 'INFO'),
            format=log.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # --- Performance ---
        perf = yaml_data['performance']
        config.performance = PerformanceConfig(
            cache_embeddings=perf.get('cache_embeddings', True),
            batch_size=perf.get('batch_size', 32),
            num_workers=perf.get('num_workers', 4)
        )
        
        # --- Development ---
        dev = yaml_data['development']
        config.development = DevelopmentConfig(
            debug=dev.get('debug', False),
            use_mock_data=dev.get('use_mock_data', False),
            mock_data_path=dev.get('mock_data_path', 'mock_news_data.json'),
            enable_profiling=dev.get('enable_profiling', False)
        )
        
        print(f"✓ Configuration loaded from {config_path}")
        
    except Exception as e:
        print(f"⚠ Error loading config file: {e}")
        print("  Using default configuration values")
        config = Config()

    # Load Prompts
    prompts_config = PromptConfig()
    try:
        with open(PROMPTS_FILE, 'r') as f:
            prompts_data = yaml.safe_load(f) or {}
        
        p = prompts_data['entity_extraction']
        prompts_config.entity_extraction = EntityExtractionPrompts(
            system_message=p.get('system_message', ''),
            task_prompt=p.get('task_prompt', ''),
            entity_context_format=p.get('entity_context_format', '')
        )
        
        p = prompts_data['sentiment_analysis']
        prompts_config.sentiment_analysis = SentimentAnalysisPrompts(
            system_message=p.get('system_message', ''),
            task_prompt=p.get('task_prompt', ''),
            few_shot_examples=p.get('few_shot_examples', '')
        )

        p = prompts_data['stock_impact']
        prompts_config.stock_impact = StockMappingPrompts(
            system_message=p.get('system_message', ''),
            task_prompt=p.get('task_prompt', '')
        )

        p = prompts_data['supply_chain']
        prompts_config.supply_chain = SupplyChainPrompts(
            system_message=p.get('system_message', ''),
            task_prompt=p.get('task_prompt', ''),
            few_shot_examples=p.get('few_shot_examples', '')
        )

        p = prompts_data['query_routing']
        prompts_config.query_routing = QueryRoutingPrompts(
            system_message=p.get('system_message', ''),
            task_prompt=p.get('task_prompt', ''),
            few_shot_examples=p.get('few_shot_examples', '')
        )
            
        print(f"✓ Prompts loaded from {PROMPTS_FILE}")
    except Exception as e:
        print(f"⚠ Error loading prompts file: {e}")
    
    config.prompts = prompts_config
    return config


# Singleton instance
_config_instance: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """Get the global configuration instance."""
    global _config_instance
    
    if _config_instance is None or reload:
        _config_instance = load_config()
    
    return _config_instance


# Auto-load on module import
config = get_config()